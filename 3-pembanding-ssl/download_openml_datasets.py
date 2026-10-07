"""Download the five requested classification datasets from OpenML."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


DATASETS = [
    {"id": 37, "slug": "diabetes_pima", "file": "diabetes_pima_openml37.csv"},
    {"id": 23, "slug": "cmc", "file": "cmc_openml23.csv"},
    {"id": 182, "slug": "satimage", "file": "satimage_openml182.csv"},
    {"id": 1461, "slug": "bank_marketing", "file": "bank_marketing_openml1461.csv"},
    {"id": 1590, "slug": "adult_income", "file": "adult_income_openml1590.csv"},
]


def main() -> None:
    dataset_root = Path(__file__).parent / "dataset"
    output_dir = dataset_root / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in DATASETS:
        dataset = fetch_openml(data_id=spec["id"], as_frame=True, parser="auto", cache=False)
        X = dataset.data.copy()
        y = dataset.target.copy()
        categorical = X.select_dtypes(include=["category", "object", "string", "bool"]).columns.tolist()
        numeric = [c for c in X.columns if c not in categorical]
        frame = X.copy()
        frame[y.name] = y
        csv_path = output_dir / spec["file"]
        frame.to_csv(csv_path, index=False)

        manifest.append({
            **spec,
            "openml_name": dataset.details.get("name"),
            "openml_version": int(dataset.details.get("version", 1)),
            "target": y.name,
            "rows": int(len(frame)),
            "features": int(X.shape[1]),
            "classes": int(y.nunique(dropna=True)),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "target_counts": {str(k): int(v) for k, v in y.value_counts(dropna=False).items()},
            "missing_feature_cells": int(X.isna().sum().sum()),
            "missing_target_rows": int(y.isna().sum()),
            "csv": str(csv_path.relative_to(Path(__file__).parent)),
            "source": f"https://www.openml.org/d/{spec['id']}",
        })
        print(
            f"{spec['slug']}: rows={len(frame)}, features={X.shape[1]}, "
            f"classes={y.nunique(dropna=True)}, numeric={len(numeric)}, "
            f"categorical={len(categorical)}, missing_features={int(X.isna().sum().sum())}"
        )

    (dataset_root / "openml_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
