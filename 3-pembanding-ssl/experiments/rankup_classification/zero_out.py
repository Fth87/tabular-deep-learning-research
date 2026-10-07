"""Standalone RankUp-CLS classification experiment using zero_out augmentation."""

from __future__ import annotations

METHOD = "zero_out"

import argparse
import json
import random
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabm import TabM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def feature_mask(x, rate):
    width = x.shape[1]
    count = min(width, max(1, int(round(rate * width))))
    order = torch.rand((len(x), width), device=x.device).argsort(dim=1)
    return torch.zeros_like(x, dtype=torch.bool).scatter_(1, order[:, :count], True)


def augment(x, strength, train_features, pseudo_labels, confidence, donor_confidence,
            quantile_edges, feature_groups):
    return torch.where(feature_mask(x, strength), torch.zeros_like(x), x)


def suggest_strengths(trial, n_features):
    weak = trial.suggest_float("weak_strength", 0.01, 0.95)
    gap = trial.suggest_float("relative_gap", 0.001, 1.0)
    return weak, weak + (1.0 - weak) * gap

class TabMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = TabM.make(n_num_features=n_features, d_out=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score_validation(y_true: np.ndarray, predictions: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, predictions, average="macro", zero_division=0))
    return float(accuracy_score(y_true, predictions))


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.from_numpy(x[start:start + 1024]).to(device)
            logits.append(model(batch).mean(dim=1).cpu())
    return torch.cat(logits).argmax(dim=1).numpy()


def fit_model(args, x_lab, y_lab, x_unlab, x_val, y_val,
              n_classes, primary_metric, weak_strength, strong_strength,
              epochs, device, trial=None):
    seed_everything(args.seed)
    model = TabMClassifier(x_lab.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
    labeled_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_lab), torch.from_numpy(y_lab.astype(np.int64))),
        batch_size=args.batch_size, shuffle=True,
    )
    unlabeled_loader = DataLoader(TensorDataset(torch.from_numpy(x_unlab)),
                                  batch_size=args.batch_size, shuffle=True)

    x_train = torch.from_numpy(np.concatenate([x_lab, x_unlab])).to(device)
    quantile_edges = make_quantile_edges(x_train) if METHOD == "range_limited" else None
    feature_groups = make_feature_groups(x_train) if METHOD == "tabcutmix_plus" else None
    best_metric, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        labeled_batches = cycle(labeled_loader)
        total_losses, supervised_losses, unsupervised_losses, accept_rates = [], [], [], []
        for (x_u_cpu,) in islice(unlabeled_loader, args.max_steps):
            x_l_cpu, y_l_cpu = next(labeled_batches)
            x_l, y_l = x_l_cpu.to(device), y_l_cpu.to(device)
            x_u = x_u_cpu.to(device)

            with torch.no_grad():
                initial_logits = model(x_u).mean(dim=1)
                weak_probabilities = initial_logits.softmax(dim=-1)
                confidence, pseudo_labels = weak_probabilities.max(dim=-1)

            weak = augment(
                x_u, weak_strength, x_train, pseudo_labels, confidence,
                args.donor_confidence, quantile_edges, feature_groups,
            )
            strong = augment(
                x_u, strong_strength, x_train, pseudo_labels, confidence,
                args.donor_confidence, quantile_edges, feature_groups,
            )
            with torch.no_grad():
                weak_logits = model(weak).mean(dim=1)
                weak_probabilities = weak_logits.softmax(dim=-1)
                confidence, pseudo_labels = weak_probabilities.max(dim=-1)

            logits_l = model(x_l)
            supervised_loss = F.cross_entropy(
                logits_l.reshape(-1, n_classes),
                y_l.repeat_interleave(logits_l.shape[1]),
            )
            logits_s = model(strong)
            unsupervised_each = F.cross_entropy(
                logits_s.reshape(-1, n_classes),
                pseudo_labels.repeat_interleave(logits_s.shape[1]),
                reduction="none",
            ).reshape(len(x_u), -1).mean(dim=1)
            accepted = confidence >= args.confidence_cutoff
            unsupervised_loss = (
                unsupervised_each[accepted].mean()
                if accepted.any() else unsupervised_each.sum() * 0
            )
            loss = supervised_loss + args.unlabeled_weight * unsupervised_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_losses.append(float(loss.detach().cpu()))
            supervised_losses.append(float(supervised_loss.detach().cpu()))
            unsupervised_losses.append(float(unsupervised_loss.detach().cpu()))
            accept_rates.append(float(accepted.float().mean().detach().cpu()))

        val_predictions = predict(model, x_val, device)
        val_metric = score_validation(y_val, val_predictions, primary_metric)
        history.append({
            "epoch": epoch, "loss": float(np.mean(total_losses)),
            "supervised_loss": float(np.mean(supervised_losses)),
            "unsupervised_loss": float(np.mean(unsupervised_losses)),
            "pseudo_accept_rate": float(np.mean(accept_rates)),
            f"validation_{primary_metric}": val_metric,
        })
        if trial is not None:
            trial.report(val_metric, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    return best_metric, best_epoch, best_state, history


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RankUp classification requires CUDA; run through Modal T4")
    device = torch.device("cuda")
    data_dir = args.data_root / args.dataset
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_type") != "classification":
        raise ValueError("This runner only accepts nominal classification datasets")

    def load_split(name):
        with np.load(data_dir / f"{name}.npz", allow_pickle=False) as split:
            return split["X"].astype(np.float32), split["y"] if "y" in split else None

    x_lab, y_lab_raw = load_split("train_labeled")
    x_unlab, _ = load_split("train_unlabeled")
    x_val, y_val_raw = load_split("validation")
    classes = [str(value) for value in manifest["classes"]]
    class_to_id = {label: index for index, label in enumerate(classes)}

    def encode(values):
        labels = np.asarray(values).astype(str)
        unknown = sorted(set(labels) - set(class_to_id))
        if unknown:
            raise ValueError(f"Unknown class labels in split: {unknown}")
        return np.asarray([class_to_id[label] for label in labels], dtype=np.int64)

    y_lab, y_val = encode(y_lab_raw), encode(y_val_raw)
    n_classes = len(classes)
    class_ratio = float(manifest["labeled_class_ratio"])
    primary_metric = "accuracy" if class_ratio <= args.balance_ratio_threshold else "macro_f1"
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )

    def objective(trial):
        weak, strong = suggest_strengths(trial, x_lab.shape[1])
        trial.set_user_attr("weak_strength", weak)
        trial.set_user_attr("strong_strength", strong)
        best_value, best_epoch, _, _ = fit_model(
            args, x_lab, y_lab, x_unlab, x_val, y_val,
            n_classes, primary_metric, weak, strong, args.optuna_epochs, device, trial,
        )
        trial.set_user_attr("best_epoch", best_epoch)
        return best_value

    def save_progress(current_study, _):
        rows = [{
            "trial": trial.number, "state": trial.state.name,
            f"validation_{primary_metric}": trial.value, **trial.params,
            "weak_strength": trial.user_attrs.get("weak_strength"),
            "strong_strength": trial.user_attrs.get("strong_strength"),
            "best_epoch": trial.user_attrs.get("best_epoch"),
        } for trial in current_study.trials]
        pd.DataFrame(rows).to_csv(
            output_dir / f"rankup_cls_{METHOD}_optuna_trials.csv", index=False
        )

    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   callbacks=[save_progress], gc_after_trial=True)
    best_trial = study.best_trial
    best_weak = float(best_trial.user_attrs["weak_strength"])
    best_strong = float(best_trial.user_attrs["strong_strength"])
    (output_dir / f"rankup_cls_{METHOD}_best_params.json").write_text(
        json.dumps({
            "dataset": args.dataset, "method": f"RankUp-CLS + {METHOD}",
            "adaptation": "multiclass pseudo-label consistency; ordinal ARC omitted",
            "primary_metric": primary_metric, "best_trial": best_trial.number,
            "best_validation_metric": best_trial.value, "n_trials": len(study.trials),
            "weak_strength": best_weak, "strong_strength": best_strong,
            "parameters": best_trial.params,
        }, indent=2), encoding="utf-8",
    )
    _, best_epoch, best_state, history = fit_model(
        args, x_lab, y_lab, x_unlab, x_val, y_val,
        n_classes, primary_metric, best_weak, best_strong, args.epochs, device,
    )
    x_test, y_test_raw = load_split("test")
    y_test = encode(y_test_raw)
    model = TabMClassifier(x_lab.shape[1], n_classes).to(device)
    model.load_state_dict(best_state)
    predictions = predict(model, x_test, device)
    metrics = {
        "dataset": args.dataset, "task_type": "classification",
        "method": f"RankUp-CLS + {METHOD}",
        "adaptation": "weak-to-strong multiclass pseudo-labeling; ordinal ARC omitted",
        "primary_metric": primary_metric,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "best_trial": int(best_trial.number), "best_validation_metric": float(best_trial.value),
        "best_epoch": int(best_epoch), "weak_strength": best_weak,
        "strong_strength": best_strong, "n_labeled": int(len(y_lab)),
        "n_unlabeled": int(len(x_unlab)), "classes": classes,
    }
    (output_dir / f"rankup_cls_{METHOD}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / f"rankup_cls_{METHOD}_predictions.npz",
        y_true=y_test, y_pred=predictions,
    )
    pd.DataFrame(history).assign(dataset=args.dataset, augmentation=METHOD).to_csv(
        output_dir / f"rankup_cls_{METHOD}_history.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Standalone RankUp-CLS + {METHOD} classification experiment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/task3-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/task3-results"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--optuna-epochs", type=int, default=40)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--unlabeled-weight", type=float, default=0.2)
    parser.add_argument("--confidence-cutoff", type=float, default=0.95)
    parser.add_argument("--donor-confidence", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--balance-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
