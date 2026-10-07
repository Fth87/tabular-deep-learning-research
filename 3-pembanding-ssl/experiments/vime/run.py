"""VIME runner adapted to Task 3's shared CSV splits.

The upstream VIME loader is MNIST-specific and its semi-supervised module uses
TensorFlow 1.x contrib. This runner keeps the source corruption operators and
objectives while implementing them with TensorFlow 2/Keras.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_split(folder: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(folder / f"{name}.npz", allow_pickle=False) as data:
        if "y" not in data:
            return data["X"].astype(np.float32), np.empty(0, dtype=str)
        return data["X"].astype(np.float32), data["y"].astype(str)


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
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--mask-probability", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reuse the authors' column-wise swap corruption implementation.
    vime_repo = Path(__file__).resolve().parent / "paper_repo"
    sys.path.insert(0, str(vime_repo))
    from vime_utils import mask_generator, pretext_generator

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        raise RuntimeError("VIME requires a CUDA GPU; run it through the Modal T4 function")
    for gpu_device in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_device, True)
    print(f"VIME TensorFlow GPU: {gpu_devices[0].name}")
    data_dir = args.data_root / args.dataset
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    task_type = manifest.get("task_type", "classification")
    X_lab, y_lab_text = load_split(data_dir, "train_labeled")
    X_unlab, _ = load_split(data_dir, "train_unlabeled")
    X_val, y_val_text = load_split(data_dir, "validation")
    X_test, y_test_text = load_split(data_dir, "test")

    y_lab, y_val, y_test, ordinal_levels = encode_targets(
        y_lab_text, y_val_text, y_test_text, manifest
    )
    y_lab, y_val, y_test = [np.asarray(values, dtype=np.int32) for values in (y_lab, y_val, y_test)]
    n_classes = len(ordinal_levels) if ordinal_levels is not None else len(np.unique(y_lab))

    # VIME's feature reconstruction head uses sigmoid, so its input must be [0, 1].
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate([X_lab, X_unlab], axis=0))
    scale_features = lambda values: np.clip(scaler.transform(values), 0.0, 1.0).astype(np.float32)
    X_lab = scale_features(X_lab)
    X_unlab = scale_features(X_unlab)
    X_val = scale_features(X_val)
    X_test = scale_features(X_test)

    n_features = X_lab.shape[1]
    inputs = tf.keras.Input(shape=(n_features,))
    latent = tf.keras.layers.Dense(n_features, activation="relu", name="encoder")(inputs)
    mask_head = tf.keras.layers.Dense(n_features, activation="sigmoid", name="mask")(latent)
    value_head = tf.keras.layers.Dense(n_features, activation="sigmoid", name="feature")(latent)
    pretrainer = tf.keras.Model(inputs, [mask_head, value_head])
    encoder = tf.keras.Model(inputs, latent)
    pretrainer.compile(
        optimizer="rmsprop",
        loss={"mask": "binary_crossentropy", "feature": "mean_squared_error"},
        loss_weights={"mask": 1.0, "feature": args.alpha},
    )
    start = time.time()
    mask = mask_generator(args.mask_probability, X_unlab)
    mask_target, corrupted = pretext_generator(mask, X_unlab)
    pretrainer.fit(
        corrupted,
        {"mask": mask_target.astype(np.float32), "feature": X_unlab},
        epochs=args.pretrain_epochs,
        batch_size=min(args.batch_size, len(X_unlab)),
        verbose=2,
    )

    X_lab_z = encoder.predict(X_lab, verbose=0)
    X_unlab_z = encoder.predict(X_unlab, verbose=0)
    X_val_z = encoder.predict(X_val, verbose=0)
    X_test_z = encoder.predict(X_test, verbose=0)
    classifier = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(n_classes),
    ])
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    best_val = float("inf")
    best_weights = None
    best_iteration = 0
    stale = 0
    for iteration in range(1, args.iterations + 1):
        lab_idx = np.random.choice(len(X_lab_z), args.batch_size, replace=len(X_lab_z) < args.batch_size)
        unl_idx = np.random.choice(len(X_unlab_z), args.batch_size, replace=len(X_unlab_z) < args.batch_size)
        x_batch = tf.convert_to_tensor(X_lab_z[lab_idx], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_lab[lab_idx], dtype=tf.int32)
        xu_original = X_unlab[unl_idx]
        xu_views = []
        for _ in range(args.views):
            view_mask = mask_generator(args.mask_probability, xu_original)
            _, view = pretext_generator(view_mask, xu_original)
            xu_views.append(encoder.predict(view.astype(np.float32), verbose=0))
        xu_batch = tf.convert_to_tensor(np.stack(xu_views, axis=0), dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = classifier(x_batch, training=True)
            supervised_loss = loss_fn(y_batch, logits)
            view_logits = tf.stack(
                [classifier(xu_batch[i], training=True) for i in range(args.views)], axis=0
            )
            consistency_loss = tf.reduce_mean(tf.math.reduce_variance(view_logits, axis=0))
            total_loss = supervised_loss + args.beta * consistency_loss
        gradients = tape.gradient(total_loss, classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))

        if iteration % 10 == 0 or iteration == 1:
            val_logits = classifier(X_val_z, training=False)
            val_loss = float(loss_fn(y_val, val_logits).numpy())
            val_predictions = np.argmax(val_logits.numpy(), axis=1)
            if ordinal_levels is None:
                val_metric = val_loss
            else:
                val_metric = float(mean_absolute_error(
                    ordinal_levels[y_val], ordinal_levels[val_predictions]
                ))
            if val_metric < best_val:
                best_val = val_metric
                best_weights = classifier.get_weights()
                best_iteration = iteration
                stale = 0
            else:
                stale += 10
            if iteration % 100 == 0:
                metric_name = "val_loss" if ordinal_levels is None else "val_mae"
                print(f"iteration={iteration} {metric_name}={val_metric:.5f} consistency={float(consistency_loss.numpy()):.5f}")
            if stale >= 100:
                break

    if best_weights is not None:
        classifier.set_weights(best_weights)
    predictions = np.argmax(classifier(X_test_z, training=False).numpy(), axis=1)
    y_true_values = ordinal_levels[y_test] if ordinal_levels is not None else y_test
    y_pred_values = ordinal_levels[predictions] if ordinal_levels is not None else predictions
    metrics = {
        "dataset": args.dataset,
        "task_type": task_type,
        "method": "VIME (author corruption/loss, TensorFlow 2 adapter; ordered classes)"
        if ordinal_levels is not None else "VIME (author corruption/loss, TensorFlow 2 adapter)",
        "primary_metric": "mae" if ordinal_levels is not None else manifest["primary_metric"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "best_iteration": int(best_iteration),
        "best_validation_mae" if ordinal_levels is not None else "best_validation_loss": float(best_val),
        "fit_seconds": float(time.time() - start), "n_labeled": int(len(y_lab)),
        "n_unlabeled": int(len(X_unlab)), "n_validation": int(len(y_val)), "n_test": int(len(y_test)),
        "pretrain_epochs": args.pretrain_epochs, "iterations_requested": args.iterations,
        "views": args.views, "mask_probability": args.mask_probability,
        "alpha": args.alpha, "beta": args.beta,
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
    (target_dir / "vime_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(target_dir / "vime_predictions.npz",
                        y_true=y_true_values, y_pred=y_pred_values, y_pred_class=predictions)
    pretrainer.save(target_dir / "vime_pretrainer.keras")
    classifier.save(target_dir / "vime_classifier.keras")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
