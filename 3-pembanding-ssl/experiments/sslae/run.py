"""Run the authors' SSLAE classifier on a prepared Task 3 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder


def load_split(folder: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(folder / f"{name}.npz", allow_pickle=False) as data:
        if "y" not in data:
            return data["X"], np.empty(0, dtype=str)
        return data["X"], data["y"].astype(str)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/task3-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/task3-results"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(__file__).resolve().parent / "paper_repo"
    sys.path.insert(0, str(source))
    from sslae.sklearn_classifier import SSLAEClassifier

    data_dir = args.data_root / args.dataset
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    task_type = manifest.get("task_type", "classification")
    X_labeled, y_labeled_text = load_split(data_dir, "train_labeled")
    X_unlabeled, _ = load_split(data_dir, "train_unlabeled")
    X_val, y_val_text = load_split(data_dir, "validation")
    X_test, y_test_text = load_split(data_dir, "test")

    try:
        y_labeled, y_val, y_test, ordinal_levels = encode_targets(
            y_labeled_text, y_val_text, y_test_text, manifest
        )
    except ValueError as exc:
        raise ValueError("Validation/test contains a class absent from the labeled training split") from exc

    model = SSLAEClassifier(
        ssl_weight=0.5, batch_size=min(32, max(2, len(y_labeled))),
        num_epochs=args.epochs, stopping_crit=10,
        random_state=args.seed, verbose=False,
    )
    model.fit(X_labeled, X_unlabeled, y_labeled, X_val=X_val, y_val=y_val)
    predictions = np.asarray(model.predict(X_test), dtype=int).reshape(-1)
    y_true_values = ordinal_levels[y_test] if ordinal_levels is not None else y_test
    y_pred_values = ordinal_levels[predictions] if ordinal_levels is not None else predictions
    metrics = {
        "dataset": args.dataset,
        "task_type": task_type,
        "method": "SSLAE (authors' implementation; ordinal labels treated as ordered classes)"
        if ordinal_levels is not None else "SSLAE (authors' implementation)",
        "primary_metric": "mae" if ordinal_levels is not None else manifest["primary_metric"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "fit_seconds": float(model.execution_time_),
        "n_labeled": int(len(y_labeled)), "n_unlabeled": int(len(X_unlabeled)),
        "n_validation": int(len(X_val)), "n_test": int(len(X_test)),
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
    (target_dir / "sslae_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(target_dir / "sslae_predictions.npz",
                        y_true=y_true_values, y_pred=y_pred_values, y_pred_class=predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
