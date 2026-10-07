"""Run the GFTab model on Task 3's shared splits and raw mixed-type columns."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def read_table(data_dir: Path, split: str, features: list[str], target: str | None) -> tuple[pd.DataFrame, np.ndarray | None]:
    frame = pd.read_csv(data_dir / f"{split}.csv")
    x = frame[features]
    y = frame[target].astype(str).to_numpy() if target and target in frame else None
    return x, y


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


def encode_features(
    x_train: pd.DataFrame,
    matrices: list[pd.DataFrame],
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    numeric_outputs: list[np.ndarray] = []
    cardinalities: list[int] = []

    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        train_num = numeric_imputer.fit_transform(x_train[numeric_columns])
        quantiles = QuantileTransformer(
            n_quantiles=min(1000, len(train_num)),
            output_distribution="normal", random_state=seed,
        )
        quantiles.fit(train_num)
        numeric_outputs = [
            quantiles.transform(numeric_imputer.transform(frame[numeric_columns])).astype(np.float32)
            for frame in matrices
        ]

    cat_matrices = [np.empty((len(frame), len(categorical_columns)), dtype=np.int64) for frame in matrices]
    for col_index, column in enumerate(categorical_columns):
        train_values = x_train[column].fillna("__missing__").astype(str)
        categories = sorted(train_values.unique().tolist())
        lookup = {value: index for index, value in enumerate(categories)}
        cardinalities.append(len(categories) + 1)
        for split_index, frame in enumerate(matrices):
            values = frame[column].fillna("__missing__").astype(str)
            cat_matrices[split_index][:, col_index] = [lookup.get(value, len(lookup)) for value in values]

    return numeric_outputs, cat_matrices, cardinalities


def encode_leaf_ids(leaf_arrays: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    widths = {array.shape[1] for array in leaf_arrays}
    if len(widths) != 1:
        raise ValueError(f"XGBoost returned inconsistent leaf widths: {sorted(widths)}")
    combined = np.concatenate(leaf_arrays, axis=0).astype(np.int64)
    encoded = combined.copy()
    next_id = 0
    for tree_index in range(combined.shape[1]):
        values = np.unique(combined[:, tree_index])
        lookup = {value: next_id + offset for offset, value in enumerate(values)}
        encoded[:, tree_index] = [lookup[value] for value in combined[:, tree_index]]
        next_id += len(values)
    sizes = [len(array) for array in leaf_arrays]
    outputs: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        outputs.append(encoded[offset:offset + size])
        offset += size
    return outputs, next_id, combined.shape[1]


def batch_tensors(*arrays: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(array, device=device) for array in arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/task3-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/task3-results"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches-per-epoch", type=int, default=100)
    parser.add_argument("--max-validation-batches", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GFTab's upstream GFK and corruption code requires a CUDA GPU; run via Modal T4")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    data_dir = args.data_root / args.dataset
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    task_type = manifest.get("task_type", "classification")
    features = manifest["feature_columns"]
    target = manifest["target"]
    numeric_columns = manifest["numeric_columns"]
    categorical_columns = manifest["categorical_columns"]

    x_lab_raw, y_lab_text = read_table(data_dir, "train_labeled", features, target)
    x_unlab_raw, _ = read_table(data_dir, "train_unlabeled", features, target)
    x_val_raw, y_val_text = read_table(data_dir, "validation", features, target)
    x_test_raw, y_test_text = read_table(data_dir, "test", features, target)
    frames = [x_lab_raw, x_unlab_raw, x_val_raw, x_test_raw]
    x_train_pool = pd.concat([x_lab_raw, x_unlab_raw], ignore_index=True)
    x_num, x_cat, cat_cardinalities = encode_features(
        x_train_pool, frames, numeric_columns, categorical_columns, args.seed
    )
    try:
        y_lab, y_val, y_test, ordinal_levels = encode_targets(
            y_lab_text, y_val_text, y_test_text, manifest
        )
        y_lab, y_val, y_test = [
            np.asarray(values, dtype=np.int64) for values in (y_lab, y_val, y_test)
        ]
    except ValueError as exc:
        raise ValueError("Validation/test contains a target level absent from the training data") from exc
    n_classes = len(ordinal_levels) if ordinal_levels is not None else len(np.unique(y_lab))

    # Fit the paper's tree-based target embedding only on labeled training data.
    tree_x_lab = np.concatenate([x_num[0], x_cat[0].astype(np.float32)], axis=1)
    tree_x_val = np.concatenate([x_num[2], x_cat[2].astype(np.float32)], axis=1)
    tree_x_test = np.concatenate([x_num[3], x_cat[3].astype(np.float32)], axis=1)
    tree = XGBClassifier(
        n_estimators=max(10, len(features)), max_depth=4, n_jobs=-1,
        random_state=args.seed, eval_metric="mlogloss",
        tree_method="hist", device="cuda",
    )
    tree.fit(tree_x_lab, y_lab)

    def leaf_matrix(values: np.ndarray) -> np.ndarray:
        leaves = np.asarray(tree.apply(values))
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        elif leaves.ndim > 2:
            leaves = leaves.reshape(len(values), -1)
        return leaves.astype(np.int64)

    leaf_splits, leaf_count, tree_count = encode_leaf_ids([
        leaf_matrix(tree_x_lab), leaf_matrix(tree_x_val), leaf_matrix(tree_x_test),
    ])
    leaves_lab, leaves_val, leaves_test = leaf_splits

    repo = Path(__file__).resolve().parent / "paper_repo"
    sys.path.insert(0, str(repo))
    from algorithms.GKF_solver import GFK
    from models.GKSMT import Model

    class Task3GFTabModel(Model):
        def tab_corruption(self, x1, x2, new_unique_value, unique_value, lambda_value=1.0):
            if x2.shape[1] == 0:
                batch_size, feature_dim = x1.shape
                index1 = torch.randperm(feature_dim, device=x1.device)
                index2 = torch.randperm(feature_dim, device=x1.device)
                index_row = torch.randperm(batch_size, device=x1.device)
                shuffled = x1[index_row]
                weak = lambda_value * x1 + (1 - lambda_value) * shuffled[:, index1]
                strong = (1 - lambda_value) * x1 + lambda_value * shuffled[:, index2]
                return weak, strong, x2.clone(), x2.clone()
            return super().tab_corruption(x1, x2, new_unique_value, unique_value, lambda_value)

    n_cont = len(numeric_columns)
    n_cat = len(categorical_columns)
    model_args = SimpleNamespace(
        num_cont=n_cont, num_cat=n_cat, lambda_value=0.9, cat_mask_ratio=0.2,
        fcn_hidden=12, fcn_n_sin=3, dim_head=64, embed_dim=24, cont_emb=16,
        depth=2, heads=1, cat_unique_list=cat_cardinalities, out_class=n_classes,
    )
    model = Task3GFTabModel(model_args, leaf_count, tree_count, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    classification_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
    gfk_loss = GFK()

    def loader(
        indices: np.ndarray, *values: np.ndarray, shuffle: bool, drop_last: bool | None = None,
    ) -> DataLoader:
        tensors = [torch.as_tensor(value[indices]) for value in values]
        return DataLoader(
            TensorDataset(*tensors), batch_size=args.batch_size, shuffle=shuffle,
            drop_last=shuffle if drop_last is None else drop_last,
        )

    idx_unlab = np.arange(len(x_unlab_raw))
    idx_lab = np.arange(len(y_lab))
    idx_val = np.arange(len(y_val))
    unlab_loader = loader(idx_unlab, x_num[1], x_cat[1], shuffle=True)
    lab_loader = loader(idx_lab, leaves_lab, y_lab, shuffle=True, drop_last=False)
    val_self_loader = loader(idx_val, x_num[2], x_cat[2], shuffle=False, drop_last=True)
    val_sup_loader = loader(idx_val, leaves_val, y_val, shuffle=False)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        self_losses: list[float] = []
        cls_losses: list[float] = []
        for batch_index, (batch_num, batch_cat) in enumerate(unlab_loader):
            if batch_index >= args.max_batches_per_epoch:
                break
            batch_num, batch_cat = batch_tensors(batch_num, batch_cat, device=device)
            negative, positive, _, _ = model(batch_num.float(), batch_cat.long(), None, mode="self")
            loss = gfk_loss.fit(negative.detach(), positive)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self_losses.append(float(loss.detach().cpu()))

        model.train()
        for batch_index, (batch_leaves, batch_y) in enumerate(lab_loader):
            if batch_index >= args.max_batches_per_epoch:
                break
            batch_leaves, batch_y = batch_tensors(batch_leaves, batch_y, device=device)
            logits, _ = model(None, None, batch_leaves.long(), mode="semi")
            loss = classification_loss(logits, batch_y.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cls_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_self_losses: list[float] = []
        val_cls_losses: list[float] = []
        val_true_ids: list[np.ndarray] = []
        val_pred_ids: list[np.ndarray] = []
        with torch.no_grad():
            for batch_index, (batch_num, batch_cat) in enumerate(val_self_loader):
                if batch_index >= args.max_validation_batches:
                    break
                batch_num, batch_cat = batch_tensors(batch_num, batch_cat, device=device)
                negative, positive, _, _ = model(batch_num.float(), batch_cat.long(), None, mode="self")
                val_self_losses.append(float(gfk_loss.fit(negative, positive).cpu()))
            for batch_index, (batch_leaves, batch_y) in enumerate(val_sup_loader):
                if batch_index >= args.max_validation_batches:
                    break
                batch_leaves, batch_y = batch_tensors(batch_leaves, batch_y, device=device)
                logits, _ = model(None, None, batch_leaves.long(), mode="semi")
                val_cls_losses.append(float(classification_loss(logits, batch_y.long()).cpu()))
                val_true_ids.append(batch_y.cpu().numpy().reshape(-1))
                val_pred_ids.append(logits.argmax(dim=1).cpu().numpy())

        if ordinal_levels is None:
            val_loss = float(np.mean(val_self_losses) + np.mean(val_cls_losses))
        else:
            val_loss = float(mean_absolute_error(
                ordinal_levels[np.concatenate(val_true_ids)],
                ordinal_levels[np.concatenate(val_pred_ids)],
            ))
        print(
            f"epoch={epoch} self_loss={np.mean(self_losses):.5f} "
            f"cls_loss={np.mean(cls_losses):.5f} "
            f"{'val_loss' if ordinal_levels is None else 'val_mae'}={val_loss:.5f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = loader(np.arange(len(y_test)), leaves_test, y_test, shuffle=False)
    true_values: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_leaves, batch_y in test_loader:
            batch_leaves, batch_y = batch_tensors(batch_leaves, batch_y, device=device)
            logits, _ = model(None, None, batch_leaves.long(), mode="semi")
            true_values.append(batch_y.cpu().numpy().reshape(-1))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    y_true_ids = np.concatenate(true_values)
    y_pred_ids = np.concatenate(predictions)
    y_true = ordinal_levels[y_true_ids] if ordinal_levels is not None else y_true_ids
    y_pred = ordinal_levels[y_pred_ids] if ordinal_levels is not None else y_pred_ids
    metrics = {
        "dataset": args.dataset,
        "task_type": task_type,
        "method": "GFTab (authors' GKSMT, GFK loss and tree embedding; ordered classes)"
        if ordinal_levels is not None else "GFTab (authors' GKSMT, GFK loss and tree embedding)",
        "primary_metric": "mae" if ordinal_levels is not None else manifest["primary_metric"],
        "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
        "macro_precision": float(precision_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)),
        "fit_seconds": float(time.time() - start), "best_epoch": best_epoch,
        "best_validation_mae" if ordinal_levels is not None else "best_validation_loss": best_loss,
        "n_labeled": int(len(y_lab)),
        "n_unlabeled": int(len(x_unlab_raw)), "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)), "numeric_features": n_cont,
        "categorical_features": n_cat, "tree_embeddings": tree_count,
        "epochs_requested": args.epochs,
    }
    if ordinal_levels is not None:
        rho = spearmanr(y_true, y_pred).statistic
        metrics.update({
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(0.0 if np.isnan(rho) else rho),
            "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        })
    target_dir = args.output_root / args.dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "gftab_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(target_dir / "gftab_predictions.npz",
                        y_true=y_true, y_pred=y_pred, y_true_class=y_true_ids, y_pred_class=y_pred_ids)
    torch.save(model.state_dict(), target_dir / "gftab_model.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
