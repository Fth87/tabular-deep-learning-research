"""Standalone RankUp continuous regression experiment using Zero Out augmentation."""

from __future__ import annotations

import argparse
import json
import random
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
from experiments.recording import save_run_config, save_checkpoint
from experiments.resume import load_study, save_study, remaining_trials
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tabm import TabM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

METHOD = "zero_out"


class TabMRankUp(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.model = TabM.make(n_num_features=n_features, d_out=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        return output[..., :1], output[..., 1:]


def pairwise_logits(rank_logits: torch.Tensor) -> torch.Tensor:
    if rank_logits.ndim == 2:
        return rank_logits[:, None, :] - rank_logits[None, :, :]
    rank_logits = rank_logits.permute(1, 0, 2)
    return rank_logits[:, :, None, :] - rank_logits[:, None, :, :]


def supervised_rank_loss(rank_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = pairwise_logits(rank_logits)
    difference = y[:, None, 0] - y[None, :, 0]
    greater = (difference > 0).float() + 0.5 * (difference == 0).float()
    targets = torch.stack([1 - greater, greater], dim=-1)
    if logits.ndim == 4:
        targets = targets.unsqueeze(0).expand(logits.shape[0], -1, -1, -1)
    return F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1, 2))


def pseudo_rank_loss(
    weak: torch.Tensor, strong: torch.Tensor, cutoff: float,
) -> tuple[torch.Tensor, float]:
    probabilities = F.softmax(pairwise_logits(weak), dim=-1)
    if probabilities.ndim == 4:
        probabilities = probabilities.mean(dim=0)
    confidence, targets = probabilities.max(dim=-1)
    logits = pairwise_logits(strong)
    if logits.ndim == 4:
        targets = targets.unsqueeze(0).expand(logits.shape[0], -1, -1)
        confidence = confidence.unsqueeze(0).expand(logits.shape[0], -1, -1)
    losses = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), reduction="none")
    accepted = confidence.reshape(-1) >= cutoff
    loss = (losses * accepted).mean()
    return loss, float(accepted.float().mean().detach().cpu())


def predict(model: nn.Module, x: np.ndarray, scaler: StandardScaler, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.from_numpy(x[start:start + 1024]).to(device)
            regression, _ = model(batch)
            predictions.extend(regression.mean(dim=1).squeeze(-1).cpu().numpy().tolist())
    return scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).reshape(-1)


def feature_mask(x: torch.Tensor, rate: float) -> torch.Tensor:
    width = x.shape[1]
    count = min(width, max(1, int(round(rate * width))))
    order = torch.rand((len(x), width), device=x.device).argsort(dim=1)
    return torch.zeros_like(x, dtype=torch.bool).scatter_(1, order[:, :count], True)


def augment_method(x, strength, train_features, pseudo_labels, confidence,
                   donor_confidence, quantile_edges, feature_groups):
    return torch.where(feature_mask(x, strength), torch.zeros_like(x), x)


def augment(
    x: torch.Tensor, strength: float, train_features: torch.Tensor,
    pseudo_labels: torch.Tensor, confidence: torch.Tensor,
    donor_confidence: float, quantile_edges: torch.Tensor | None,
    feature_groups: list[torch.Tensor] | None,
) -> torch.Tensor:
    return augment_method(
        x, strength, train_features, pseudo_labels, confidence,
        donor_confidence, quantile_edges, feature_groups,
    )


def suggest_strengths(trial: optuna.Trial, n_features: int) -> tuple[float, float]:
    if METHOD == "gaussian":
        weak = trial.suggest_float("weak_sigma", 0.005, 1.40, log=True)
        relative_gap = trial.suggest_float("relative_gap", 0.001, 1.0)
        strong = weak + (1.50 - weak) * relative_gap
    else:
        weak = trial.suggest_float("weak_strength", 0.01, 0.95)
        relative_gap = trial.suggest_float("relative_gap", 0.001, 1.0)
        strong = weak + (1.0 - weak) * relative_gap
    return weak, strong


def repeat_loader(loader):
    while True:
        yield from loader


def fit_model(
    args: argparse.Namespace,
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    x_unlab: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weak_strength: float,
    strong_strength: float,
    epochs: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, int, dict, list[dict]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_scaler = StandardScaler().fit(y_lab.reshape(-1, 1))
    y_lab_scaled = target_scaler.transform(y_lab.reshape(-1, 1)).astype(np.float32)
    x_lab_gpu = torch.from_numpy(x_lab).to(device)
    x_unlab_gpu = torch.from_numpy(x_unlab).to(device)
    train_features = torch.cat([x_lab_gpu, x_unlab_gpu], dim=0)
    quantile_edges = make_quantile_edges(train_features) if METHOD == "range_limited" else None
    feature_groups = make_feature_groups(train_features) if METHOD == "tabcutmix_plus" else None

    model = TabMRankUp(x_lab.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
    labeled_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_lab), torch.from_numpy(y_lab_scaled),
                      torch.from_numpy(y_lab.astype(np.float32).reshape(-1, 1))),
        batch_size=args.batch_size, shuffle=True,
    )
    unlabeled_loader = DataLoader(TensorDataset(torch.from_numpy(x_unlab)),
                                  batch_size=args.batch_size, shuffle=True)
    labeled_batches = repeat_loader(labeled_loader)
    best_mse, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals, labeled_losses, unlabeled_losses, accept_rates = [], [], [], []
        for (x_u_cpu,) in islice(unlabeled_loader, args.max_steps):
            x_l_cpu, y_l_scaled_cpu, y_l_ord_cpu = next(labeled_batches)
            x_l, y_l_scaled, y_l_ord = (value.to(device) for value in
                                         (x_l_cpu, y_l_scaled_cpu, y_l_ord_cpu))
            x_u = x_u_cpu.to(device)
            pseudo_labels, confidence = None, None
            weak = augment(x_u, weak_strength, train_features, pseudo_labels, confidence,
                           args.donor_confidence, quantile_edges, feature_groups)
            strong = augment(x_u, strong_strength, train_features, pseudo_labels, confidence,
                             args.donor_confidence, quantile_edges, feature_groups)
            pred_l, rank_l = model(x_l)
            prediction_loss = F.mse_loss(pred_l, y_l_scaled[:, None, :].expand_as(pred_l))
            rank_lab_loss = supervised_rank_loss(rank_l, y_l_ord)
            with torch.no_grad():
                _, rank_weak = model(weak)
            _, rank_strong = model(strong)
            rank_unlab_loss, accept_rate = pseudo_rank_loss(
                rank_weak, rank_strong, args.confidence_cutoff
            )
            loss = prediction_loss + args.rank_weight * rank_lab_loss + args.unlabeled_weight * rank_unlab_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
            labeled_losses.append(float(rank_lab_loss.detach().cpu()))
            unlabeled_losses.append(float(rank_unlab_loss.detach().cpu()))
            accept_rates.append(accept_rate)

        val_scores = predict(model, x_val, target_scaler, device)
        val_mse = float(mean_squared_error(y_val, val_scores))
        history.append({"epoch": epoch, "loss": float(np.mean(totals)),
                        "rank_labeled": float(np.mean(labeled_losses)),
                        "rank_unlabeled": float(np.mean(unlabeled_losses)),
                        "pseudo_accept_rate": float(np.mean(accept_rates)),
                        "validation_mse": val_mse})
        if trial is not None:
            trial.report(val_mse, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    return best_mse, best_epoch, best_state, history


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RankUp requires a CUDA GPU; run through the Modal T4 function")
    device = torch.device("cuda")
    print(f"RankUp {METHOD} PyTorch GPU: {torch.cuda.get_device_name(device)}")
    data_dir = args.data_root / args.dataset
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_type") != "regression":
        raise ValueError("RankUp regression experiment only accepts regression datasets")

    def load_split(name: str) -> tuple[np.ndarray, np.ndarray | None]:
        with np.load(data_dir / f"{name}.npz", allow_pickle=False) as split:
            target = split["y"] if "y" in split else None
            return split["X"].astype(np.float32), target

    x_lab, y_lab = load_split("train_labeled")
    x_unlab, _ = load_split("train_unlabeled")
    x_val, y_val = load_split("validation")
    assert y_lab is not None and y_val is not None
    y_lab = np.asarray(y_lab, dtype=np.float32).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(output_dir, "rankup", METHOD, args, manifest)
    if args.params_file is None:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        )

        def objective(trial: optuna.Trial) -> float:
            weak, strong = suggest_strengths(trial, x_lab.shape[1])
            trial.set_user_attr("weak_strength", weak)
            trial.set_user_attr("strong_strength", strong)
            val_mse, best_epoch, _, _ = fit_model(
                args, x_lab, y_lab, x_unlab, x_val, y_val,
                weak, strong, args.optuna_epochs, trial,
            )
            trial.set_user_attr("best_epoch", best_epoch)
            return val_mse

        def save_progress(current_study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
            rows = [{
                "trial": item.number, "state": item.state.name,
                "validation_mse": item.value, **item.params,
                "weak_strength": item.user_attrs.get("weak_strength"),
                "strong_strength": item.user_attrs.get("strong_strength"),
                "best_epoch": item.user_attrs.get("best_epoch"),
            } for item in current_study.trials]
            pd.DataFrame(rows).to_csv(
                output_dir / f"rankup_{METHOD}_optuna_trials.csv", index=False
            )

        if args.n_trials < 1:
            raise ValueError("--n-trials must be at least 1")
        study_path = output_dir / f"rankup_{METHOD}_study.pkl"
        study = load_study(study, study_path)
        study.optimize(objective, n_trials=remaining_trials(study, args.n_trials), n_jobs=1,
                       callbacks=[save_progress, lambda current, _: save_study(current, study_path)],
                       gc_after_trial=True)
        best_trial = study.best_trial
        trial_counts = {state.name.lower(): sum(t.state == state for t in study.trials)
                        for state in optuna.trial.TrialState}
        search_space = {name: json.loads(optuna.distributions.distribution_to_json(distribution))
                        for name, distribution in best_trial.distributions.items()}
        best_weak = float(best_trial.user_attrs["weak_strength"])
        best_strong = float(best_trial.user_attrs["strong_strength"])
        (output_dir / f"rankup_{METHOD}_best_params.json").write_text(json.dumps({
            "dataset": args.dataset, "method": METHOD,
            "labeled_fraction": manifest["labeled_fraction_of_train"], "split_seed": manifest["seed"], "tuning_seed": args.seed,
            "best_trial": best_trial.number, "best_validation_mse": best_trial.value,
            "n_trials": len(study.trials), "optuna_epochs": args.optuna_epochs,
            "parameters": best_trial.params, "trial_counts": trial_counts,
            "search_space": search_space,
            "weak_strength": best_weak, "strong_strength": best_strong,
        }, indent=2), encoding="utf-8")
        print(f"Best Optuna trial={best_trial.number}, validation MSE={best_trial.value:.5f}")

        if args.tune_only:
            return
    else:
        if args.tune_only:
            raise ValueError("--tune-only cannot be combined with --params-file")
        params = json.loads(args.params_file.read_text(encoding="utf-8"))
        if (params["dataset"] != args.dataset or params["method"] != METHOD
                or params.get("labeled_fraction") != manifest["labeled_fraction_of_train"]
                or params.get("split_seed") != manifest["seed"]):
            raise ValueError("Parameter file does not match dataset, method, fraction or split seed")
        best_weak = float(params["weak_strength"])
        best_strong = float(params["strong_strength"])
        best_trial = SimpleNamespace(number=int(params["best_trial"]),
                                     value=float(params["best_validation_mse"]))

    # Refit using the selected augmentation strengths; the test split is opened only now.
    _, best_epoch, best_state, history = fit_model(
        args, x_lab, y_lab, x_unlab, x_val, y_val,
        best_weak, best_strong, args.epochs,
    )
    x_test, y_test = load_split("test")
    assert y_test is not None
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1)
    target_scaler = StandardScaler().fit(y_lab.reshape(-1, 1))
    model = TabMRankUp(x_lab.shape[1]).to(device)
    model.load_state_dict(best_state)
    scores = predict(model, x_test, target_scaler, device)
    train_score = float(mean_squared_error(y_lab, predict(model, x_lab, target_scaler, device)))
    validation_score = float(mean_squared_error(y_val, predict(model, x_val, target_scaler, device)))
    save_checkpoint(output_dir, "rankup", METHOD, model, x_lab.shape[1], args.seed,
                    best_weak, best_strong, target_scaler=target_scaler)
    metrics = {
        "dataset": args.dataset, "task_type": "regression",
        "method": f"RankUp + {METHOD}", "primary_metric": "mse",
        "mse": float(mean_squared_error(y_test, scores)),
        "train_mse": train_score, "validation_mse": validation_score,
        "mae": float(mean_absolute_error(y_test, scores)),
        "r2": float(r2_score(y_test, scores)),
        "best_trial": int(best_trial.number),
        "best_validation_mse": float(best_trial.value), "best_epoch": int(best_epoch),
        "weak_strength": best_weak, "strong_strength": best_strong,
        "n_labeled": int(len(y_lab)), "n_unlabeled": int(len(x_unlab)), "seed": int(args.seed),
    }
    (output_dir / f"rankup_{METHOD}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / f"rankup_{METHOD}_predictions.npz",
                        y_true=y_test, y_score=scores, residual=scores - y_test)
    pd.DataFrame(history).assign(dataset=args.dataset, augmentation=METHOD).to_csv(
        output_dir / f"rankup_{METHOD}_history.csv", index=False)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Standalone RankUp {METHOD} Optuna experiment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/task3-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/task3-results"))
    parser.add_argument("--epochs", type=int, default=120, help="Final training epochs")
    parser.add_argument("--optuna-epochs", type=int, default=40)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--unlabeled-weight", type=float, default=0.2)
    parser.add_argument("--confidence-cutoff", type=float, default=0.95)
    parser.add_argument("--donor-confidence", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--params-file", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
