"""Task 3 adapter for STUNT's self-generated few-shot task objective."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn


def load_split(folder: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(folder / f"{name}.npz", allow_pickle=False) as values:
        if "y" not in values:
            return values["X"].astype(np.float32), np.empty(0, dtype=str)
        return values["X"].astype(np.float32), values["y"].astype(str)


def encode_targets(train, validation, test, manifest):
    if manifest["task_type"] == "ordinal_regression":
        levels = np.asarray(manifest["target_values"], dtype=np.float64)
        def encode(values):
            values = np.asarray(values, dtype=np.float64)
            indices = np.searchsorted(levels, values)
            if np.any(indices >= len(levels)) or not np.allclose(levels[indices], values, rtol=1e-6, atol=1e-7):
                raise ValueError("Ordinal target contains a level missing from the manifest")
            return indices
        return encode(train), encode(validation), encode(test), levels
    encoder = LabelEncoder().fit(train)
    return encoder.transform(train), encoder.transform(validation), encoder.transform(test), None


def make_episode(
    unlabeled: np.ndarray,
    ways: int,
    shots: int,
    queries: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a STUNT task by clustering selected columns, then corrupting them."""
    row_count, feature_count = unlabeled.shape
    min_rows = ways * (shots + queries)
    sample_size = min(row_count, max(min_rows * 4, 2048))
    rows = rng.choice(row_count, size=sample_size, replace=False)
    x = unlabeled[rows].copy()

    low = max(1, int(feature_count * 0.2))
    high = max(low, int(feature_count * 0.5))
    for _ in range(8):
        selected_count = int(rng.integers(low, high + 1))
        selected = rng.choice(feature_count, size=selected_count, replace=False)
        pseudo_y = MiniBatchKMeans(
            n_clusters=ways, random_state=int(rng.integers(0, 2**31 - 1)),
            n_init=3, max_iter=50, batch_size=min(1024, sample_size),
        ).fit_predict(x[:, selected])
        classes, counts = np.unique(pseudo_y, return_counts=True)
        valid = classes[counts >= shots + queries]
        if len(valid) == ways:
            break
    else:
        raise RuntimeError("Could not form a balanced K-Means pseudo-task after 8 attempts")

    corrupted = x.copy()
    for column in selected:
        corrupted[:, column] = corrupted[rng.permutation(sample_size), column]

    support_rows: list[int] = []
    query_rows: list[int] = []
    support_labels: list[int] = []
    query_labels: list[int] = []
    chosen_classes = rng.choice(valid, size=ways, replace=False)
    for task_label, pseudo_class in enumerate(chosen_classes):
        members = rng.permutation(np.flatnonzero(pseudo_y == pseudo_class))
        support_rows.extend(members[:shots].tolist())
        query_rows.extend(members[shots:shots + queries].tolist())
        support_labels.extend([task_label] * shots)
        query_labels.extend([task_label] * queries)

    return (
        corrupted[support_rows], np.asarray(support_labels, dtype=np.int64),
        corrupted[query_rows], np.asarray(query_labels, dtype=np.int64),
    )


def make_support(
    x: np.ndarray, y: np.ndarray, shots: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    classes = np.unique(y)
    available = min(int(np.sum(y == cls)) for cls in classes)
    actual_shots = min(shots, available)
    chosen = np.concatenate([
        rng.permutation(np.flatnonzero(y == cls))[:actual_shots] for cls in classes
    ])
    return x[chosen], y[chosen], actual_shots


def evaluate(
    model: nn.Module,
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    device: torch.device,
    ordinal_levels: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        support = model(torch.from_numpy(support_x[None]).to(device))[0]
        query = model(torch.from_numpy(query_x[None]).to(device))[0]
        classes = np.unique(support_y)
        prototypes = torch.stack([
            support[torch.as_tensor(support_y == cls, device=device)].mean(dim=0)
            for cls in classes
        ])
        distances = torch.cdist(query, prototypes).square()
        predicted = classes[distances.argmin(dim=1).cpu().numpy()]
    if ordinal_levels is None:
        score = float(accuracy_score(query_y, predicted))
    else:
        score = float(mean_absolute_error(ordinal_levels[query_y], ordinal_levels[predicted]))
    return score, predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/task3-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/task3-results"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--episodes-per-step", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("STUNT requires a CUDA GPU; run it through the Modal T4 function")
    device = torch.device("cuda")
    print(f"STUNT PyTorch GPU: {torch.cuda.get_device_name(device)}")

    data_dir = args.data_root / args.dataset
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    task_type = manifest.get("task_type", "classification")
    x_labeled, y_labeled_text = load_split(data_dir, "train_labeled")
    x_unlabeled, _ = load_split(data_dir, "train_unlabeled")
    x_val, y_val_text = load_split(data_dir, "validation")
    x_test, y_test_text = load_split(data_dir, "test")

    try:
        y_labeled, y_val, y_test, ordinal_levels = encode_targets(
            y_labeled_text, y_val_text, y_test_text, manifest
        )
        y_labeled, y_val, y_test = [
            np.asarray(values, dtype=np.int64) for values in (y_labeled, y_val, y_test)
        ]
    except ValueError as exc:
        raise ValueError("A split contains a target level missing from labeled train") from exc

    actual_ways = min(args.ways, len(ordinal_levels) if ordinal_levels is not None else len(np.unique(y_labeled)))
    support_x, support_y, actual_shots = make_support(
        x_labeled, y_labeled, args.shots, rng
    )
    model_source = Path(__file__).resolve().parent / "paper_repo"
    sys.path.insert(0, str(model_source))
    from models.protonet_model.mlp import MLPProto

    model = MLPProto(x_labeled.shape[1], 256, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_score = float("inf") if ordinal_levels is not None else -1.0
    best_state = None
    best_step = 0
    stale = 0
    start = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        support_batches, support_targets = [], []
        query_batches, query_targets = [], []
        for _ in range(args.episodes_per_step):
            sx, sy, qx, qy = make_episode(
                x_unlabeled, actual_ways, 1, args.queries, rng
            )
            support_batches.append(sx)
            support_targets.append(sy)
            query_batches.append(qx)
            query_targets.append(qy)

        sx_t = torch.as_tensor(np.stack(support_batches), device=device)
        qx_t = torch.as_tensor(np.stack(query_batches), device=device)
        sy_t = torch.as_tensor(np.stack(support_targets), device=device)
        qy_t = torch.as_tensor(np.stack(query_targets), device=device)
        support_embeddings = model(sx_t)
        query_embeddings = model(qx_t)
        prototypes = torch.stack([
            support_embeddings[:, sy_t[0] == cls].mean(dim=1)
            for cls in range(actual_ways)
        ], dim=1)
        logits = -torch.cdist(query_embeddings, prototypes).square()
        loss = criterion(logits.reshape(-1, actual_ways), qy_t.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            val_score, _ = evaluate(
                model, support_x, support_y, x_val, y_val, device, ordinal_levels
            )
            improved = val_score < best_score if ordinal_levels is not None else val_score > best_score
            if improved:
                best_score = val_score
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                best_step = step
                stale = 0
            else:
                stale += 25
            metric_name = "val_mae" if ordinal_levels is not None else "val_accuracy"
            print(f"step={step} loss={loss.item():.5f} {metric_name}={val_score:.5f}")
            if stale >= args.patience * 25:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, predictions = evaluate(model, support_x, support_y, x_test, y_test, device, ordinal_levels)
    y_true_values = ordinal_levels[y_test] if ordinal_levels is not None else y_test
    y_pred_values = ordinal_levels[predictions] if ordinal_levels is not None else predictions
    metrics = {
        "dataset": args.dataset,
        "task_type": task_type,
        "method": "STUNT (self-generated tasks, prototypical few-shot evaluation; ordered classes)"
        if ordinal_levels is not None else "STUNT (self-generated tasks, prototypical few-shot evaluation)",
        "primary_metric": "mae" if ordinal_levels is not None else manifest["primary_metric"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "fit_seconds": float(time.time() - start), "best_step": int(best_step),
        "best_validation_mae" if ordinal_levels is not None else "best_validation_accuracy": float(best_score),
        "ways_per_synthetic_task": actual_ways,
        "shots_per_real_class": actual_shots, "n_labeled": int(len(y_labeled)),
        "n_unlabeled": int(len(x_unlabeled)), "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)), "steps_requested": args.steps,
    }
    if ordinal_levels is not None:
        rho = spearmanr(y_true_values, y_pred_values).statistic
        metrics.update({
            "mae": float(mean_absolute_error(y_true_values, y_pred_values)),
            "rmse": float(mean_squared_error(y_true_values, y_pred_values) ** 0.5),
            "r2": float(r2_score(y_true_values, y_pred_values)),
            "spearman": float(0.0 if np.isnan(rho) else rho),
            "qwk": float(cohen_kappa_score(y_true_values, y_pred_values, weights="quadratic")),
        })
    target_dir = args.output_root / args.dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "stunt_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(target_dir / "stunt_predictions.npz",
                        y_true=y_true_values, y_pred=y_pred_values, y_pred_class=predictions)
    torch.save(model.state_dict(), target_dir / "stunt_model.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
