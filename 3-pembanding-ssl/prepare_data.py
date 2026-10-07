"""Create shared, leakage-safe splits for Task 3 SSL comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_one(spec: dict, config: dict, output_root: Path) -> dict:
    name = spec["name"]
    csv_path = Path(spec["csv"]).expanduser()
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV for {name!r} is missing: {csv_path}")
    frame = pd.read_csv(csv_path)
    target = spec["target"]
    if not target or target not in frame.columns:
        raise ValueError(f"Target column {target!r} is missing in {csv_path}")
    if target in spec.get("drop_columns", []):
        raise ValueError(f"{name}: target column {target!r} cannot also be in drop_columns")
    frame = frame.drop(columns=[c for c in spec.get("drop_columns", []) if c in frame.columns])
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    task_type = spec.get("task_type", "classification")
    raw_y = frame.pop(target)
    if task_type == "ordinal_regression":
        y = pd.to_numeric(raw_y, errors="raise")
        if not np.isfinite(y.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name}: ordinal target must contain finite numeric values")
    elif task_type == "classification":
        y = raw_y.astype(str)
    else:
        raise ValueError(f"{name}: unsupported task_type {task_type!r}")
    X = frame
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError(f"{name}: each target value needs at least 3 rows for stratified splits; found {counts.to_dict()}")

    seed = int(config.get("seed", 42))
    train_fraction = float(config.get("train_fraction", 0.7))
    val_fraction = float(config.get("validation_fraction", 0.15))
    test_fraction = float(config.get("test_fraction", 0.15))
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1")
    labeled_fraction = float(config.get("labeled_fraction", 0.1))
    all_idx = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=test_fraction, random_state=seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_fraction / (train_fraction + val_fraction),
        random_state=seed, stratify=y.iloc[trainval_idx]
    )
    labeled_idx, unlabeled_idx = train_test_split(
        train_idx, test_size=1 - labeled_fraction, random_state=seed,
        stratify=y.iloc[train_idx]
    )
    missing_labeled_classes = set(y.iloc[train_idx]) - set(y.iloc[labeled_idx])
    if task_type == "classification" and missing_labeled_classes:
        raise ValueError(
            f"{name}: labeled fraction leaves out classes {sorted(missing_labeled_classes)}; "
            "use more labeled data or a larger dataset"
        )

    if "categorical_columns" in spec:
        categorical_cols = spec["categorical_columns"]
        missing_categories = sorted(set(categorical_cols) - set(X.columns))
        if missing_categories:
            raise ValueError(f"{name}: categorical columns not found: {missing_categories}")
        numeric_cols = [c for c in X.columns if c not in categorical_cols]
        nonnumeric_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(X[c])]
        if nonnumeric_numeric:
            raise ValueError(f"{name}: mark non-numeric feature columns categorical: {nonnumeric_numeric}")
        X[categorical_cols] = X[categorical_cols].astype(object)
    else:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    if not transformers:
        raise ValueError(f"{name}: no usable feature columns")

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    # Fit only on the training pool; unlabeled training rows are valid SSL input.
    preprocessor.fit(X.iloc[train_idx])
    Xz = np.asarray(preprocessor.transform(X), dtype=np.float32)
    y_values = y.to_numpy(dtype=np.float32 if task_type == "ordinal_regression" else str)
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)

    split_indices = {
        "train_labeled": labeled_idx, "train_unlabeled": unlabeled_idx,
        "validation": val_idx, "test": test_idx,
    }
    for split, indices in split_indices.items():
        if split == "train_unlabeled":
            np.savez_compressed(out / f"{split}.npz", X=Xz[indices])
        else:
            np.savez_compressed(out / f"{split}.npz", X=Xz[indices], y=y_values[indices])
        part = X.iloc[indices].copy()
        if split != "train_unlabeled":
            part[target] = y.iloc[indices].to_numpy()
        part.to_csv(out / f"{split}.csv", index=False)
    joblib.dump(preprocessor, out / "preprocessor.joblib")

    labeled_counts = pd.Series(y_values[labeled_idx]).value_counts()
    ratio = float(labeled_counts.max() / labeled_counts.min())
    threshold = float(config.get("class_balance_ratio_threshold", 1.5))
    manifest = {
        "name": name, "source_csv": str(csv_path.resolve()), "target": target,
        "task_type": task_type,
        "rows": len(y), "raw_features": X.shape[1], "encoded_features": Xz.shape[1],
        "classes": sorted(map(str, np.unique(y_values))) if task_type == "classification" else [],
        "target_values": sorted(map(float, np.unique(y_values))) if task_type == "ordinal_regression" else [],
        "class_counts_labeled": {str(k): int(v) for k, v in labeled_counts.items()},
        "labeled_class_ratio": ratio if task_type == "classification" else None,
        "primary_metric": (
            "mae" if task_type == "ordinal_regression"
            else "accuracy" if ratio <= threshold else "macro_f1"
        ),
        "split_sizes": {k: int(len(v)) for k, v in split_indices.items()},
        "seed": seed, "train_fraction": train_fraction,
        "validation_fraction": val_fraction, "test_fraction": test_fraction,
        "labeled_fraction_of_train": labeled_fraction,
        "feature_columns": X.columns.tolist(), "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/datasets.json")
    parser.add_argument("--output", type=Path, default=Path("/mnt/task3-data"))
    parser.add_argument("--dataset", help="Run one configured dataset by name")
    args = parser.parse_args()
    config = read_config(args.config)
    specs = [d for d in config["datasets"] if args.dataset is None or d["name"] == args.dataset]
    if not specs:
        raise SystemExit(f"No dataset named {args.dataset!r} in {args.config}")
    for spec in specs:
        if not spec.get("csv") or not spec.get("target"):
            print(f"SKIP {spec['name']}: fill csv and target in {args.config}")
            continue
        print(json.dumps(prepare_one(spec, config, args.output), indent=2))


if __name__ == "__main__":
    main()
