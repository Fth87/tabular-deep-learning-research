"""Latih head SSL dengan in-context subsetting di atas TabPFN yang dibekukan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from stage8_common import load, perturb, rotate, scores

HERE = Path(__file__).resolve().parent
DATASETS = ("nhanes", "pendigits", "superconductivity")
PERCENTS = (1, 10, 30)
SEEDS = (42, 123, 456)
METRIC = {"nhanes": ("qwk_calibrated", 1), "pendigits": ("macro_f1", 1), "superconductivity": ("mse", -1)}
SUPERVISED = ("tabpfn", "head_sup", "ics_sup")
SSL_IDENTITY = ("head_distill", "head_match")
LATENT_DISTILL = {"head_gaussian": ("gaussian", 0.5), "head_dropout": ("dropout", 0.5),
                  "head_fnoise": ("fnoise", 0.3), "head_tabcutmix": ("tabcutmix", 0.5),
                  "head_angular": ("angular", (0.3, 0.5)), "head_scarf": ("scarf", 0.5)}
SSL_AUGMENTED = (*LATENT_DISTILL, "head_match_gaussian", "ics_distill_70", "ics_match_70", "ics_both_70")
CORE = (*SUPERVISED, *SSL_IDENTITY, *SSL_AUGMENTED)
ABLATIONS = ("ics_distill_50", "ics_distill_90", "ics_distill_ta")
VARIANTS = (*CORE, *ABLATIONS)
MATCH_VARIANTS = ("head_match", "head_match_gaussian", "ics_match_70")
EMA_DECAY, CONFIDENCE = 0.99, 0.95
WEIGHT_REPO = "Prior-Labs/tabpfn_3_5"
WEIGHT_NAME = "tabpfn-v3.5-20260909.safetensors"
TABPFN_VERSION = "10.2.0"
SPLITS = ("train_labeled", "train_unlabeled", "validation", "test")
SCHEMA = 2


def eligible(names, percent):
    return [name for name in names if percent < 30 or name not in ("ics_distill_50", "ics_distill_90")]


def experiment_names(choice):
    if choice == "core":
        return CORE
    if choice == "all":
        return VARIANTS
    return parse_list(choice, VARIANTS)


def find_data(explicit):
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [HERE.parent / "5-augmentasi-latent/data"]
        candidates += [p.parent for p in sorted(Path("/kaggle/input").rglob("label_01"))]
    for path in candidates:
        if all(part.is_file() for part in split_files(path)):
            return path.resolve()
    raise SystemExit("Split lengkap tidak ditemukan. Tambahkan dataset Kaggle tahap 7 atau set DATA_DIR.")


def read_data(data_root, dataset, percent):
    root = data_root / f"label_{percent:02d}" / dataset
    manifest = json.loads((root / "manifest.json").read_text())
    task = manifest["task_type"]
    class_ids = {str(c): i for i, c in enumerate(manifest.get("classes", []))}
    levels = np.asarray(manifest["target_values"]) if task == "ordinal_regression" else None
    splits = {}
    for name in SPLITS:
        x, y = load(root, name)
        if name == "train_unlabeled":
            splits[name] = (x, None)
        elif task == "classification":
            splits[name] = (x, np.array([class_ids[c] for c in y.astype(str)], dtype=np.int64))
        else:
            splits[name] = (x, np.asarray(y, dtype=np.float32).reshape(-1))
    return task, splits, levels


def choose_unlabeled(n, limit, seed):
    if not limit or n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))


def context_strata(y, task):
    if task != "regression":
        return y
    edges = np.unique(np.quantile(y, np.linspace(0.1, 0.9, 9)))
    return np.searchsorted(edges, y, side="right")


def stratified_subset(y, task, rho, rng):
    """Proportional allocation with at least one row per stratum present in the pool."""
    _, inverse, counts = np.unique(context_strata(y, task), return_inverse=True, return_counts=True)
    total = max(len(counts), round(len(y) * rho))
    target = counts * (total / len(y))
    quota = np.maximum(1, np.floor(target).astype(int))
    while quota.sum() > total:
        excess = np.where(quota > 1, quota - target, -np.inf)
        quota[excess.argmax()] -= 1
    while quota.sum() < total:
        deficit = np.where(quota < counts, target - quota, -np.inf)
        quota[deficit.argmax()] += 1
    rows = [rng.choice(np.flatnonzero(inverse == k), size=n, replace=False) for k, n in enumerate(quota)]
    # Keep relative input order so rho=1 is the original context.
    return np.sort(np.concatenate(rows))


def folds(x, y, task, seed):
    # Same splitter as TabPFNEmbedding.fit_transform in stage 7.
    splitter = StratifiedKFold if task == "classification" else KFold
    return list(splitter(n_splits=5, shuffle=True, random_state=seed).split(x, y))


def weights(out, override):
    if override:
        path = Path(override).expanduser().resolve(strict=True)
        if path.name != WEIGHT_NAME:
            raise SystemExit(f"File bobot harus bernama {WEIGHT_NAME}")
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(WEIGHT_REPO, WEIGHT_NAME, local_dir=out / "weights"))


def make_model(task, seed, model_path, device):
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return cls(model_path=model_path, device=device, random_state=seed)


def chunked(fn, x, size):
    return np.concatenate([fn(x[s:s + size]) for s in range(0, len(x), size)], axis=0)


def embedding(model, x, chunk):
    def run(rows):
        members = np.asarray(model.get_embeddings(rows, data_source="test"))
        if members.ndim != 3 or members.shape[1] != len(rows):
            raise ValueError(f"Shape embedding TabPFN tidak dikenal: {members.shape}")
        return members.mean(axis=0, dtype=np.float32)

    return chunked(run, x, chunk)


def teacher(model, x, task, chunk):
    infer = model.predict_proba if task == "classification" else model.predict
    return chunked(infer, x, chunk).astype(np.float32)


def build_base(new_model, task, splits, unlabeled_rows, seed, chunk):
    x, y = splits["train_labeled"]
    model = new_model()
    labeled = []
    for context, query in folds(x, y, task, seed):
        model.fit(x[context], y[context])
        labeled.append((query, embedding(model, x[query], chunk)))
    order = np.argsort(np.concatenate([query for query, _ in labeled]))
    base = {"e_lab": np.concatenate([emb for _, emb in labeled])[order], "unlabeled_rows": unlabeled_rows}
    model.fit(x, y)
    for name, query in (
        ("unl", splits["train_unlabeled"][0][unlabeled_rows]),
        ("val", splits["validation"][0]), ("test", splits["test"][0]),
    ):
        base[f"e_{name}"] = embedding(model, query, chunk)
        base[f"t_{name}"] = teacher(model, query, task, chunk)
    return base


def build_views(new_model, task, splits, unlabeled_rows, seed, chunk, views, setting):
    x, y = splits["train_labeled"]
    model = new_model()
    labeled = setting == "lab70"
    fold_pairs = folds(x, y, task, seed) if labeled else [(np.arange(len(x)), unlabeled_rows)]
    all_embeddings, all_predictions, all_rhos, all_sizes = [], [], [], []
    # Separate deterministic streams for view settings and folds.
    setting_id = {"50": 50, "70": 70, "90": 90, "ta": 100, "lab70": 170}[setting]
    rng = np.random.default_rng(np.random.SeedSequence([seed, setting_id]))
    for _ in range(views):
        rho = float(rng.uniform(0.5, 0.99)) if setting == "ta" else int(setting[-2:]) / 100
        view, targets, sizes = [], [], []
        for context, query in fold_pairs:
            used = context[stratified_subset(y[context], task, rho, rng)]
            model.fit(x[used], y[used])
            rows = x[query] if labeled else splits["train_unlabeled"][0][query]
            view.append(embedding(model, rows, chunk))
            if not labeled:
                targets.append(teacher(model, rows, task, chunk))
            sizes.append(len(used))
        merged = np.concatenate(view)
        if labeled:
            merged = merged[np.argsort(np.concatenate([q for _, q in fold_pairs]))]
        all_embeddings.append(merged)
        if targets:
            all_predictions.append(np.concatenate(targets))
        all_rhos.append(rho)
        all_sizes.append(sizes)
    result = {"embedding": np.stack(all_embeddings), "rho": np.array(all_rhos), "context_sizes": np.array(all_sizes)}
    if all_predictions:
        result["teacher"] = np.stack(all_predictions)
    return result


def atomic_npz(path, arrays, compress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as sink:
        save = np.savez_compressed if compress else np.savez
        save(sink, **arrays)
        sink.flush()
        os.fsync(sink.fileno())
    tmp.replace(path)


def cached(path, build):
    if path.exists():
        with np.load(path, allow_pickle=False) as source:
            return dict(source)
    start = time.monotonic()
    result = build()
    result["build_seconds"] = np.asarray(time.monotonic() - start)
    atomic_npz(path, result)
    print(f"{path.parent.name}/{path.stem}: {float(result['build_seconds']):.1f} s", flush=True)
    return result


def calibrate(y_val, pred_val, levels):
    proportions = np.cumsum([(y_val == level).mean() for level in levels])[:-1]
    return np.quantile(pred_val, proportions)


def evaluate(task, levels, y_val, y_test, pred_val, pred_test):
    classification = task == "classification"
    raw_val = pred_val.argmax(axis=1) if classification else pred_val
    raw_test = pred_test.argmax(axis=1) if classification else pred_test
    arrays = {"y_validation": y_val, "y_test": y_test, "pred_validation": raw_val, "pred_test": raw_test}
    if classification:
        arrays.update(proba_validation=pred_val, proba_test=pred_test)
    results = {split: scores(task, arrays[f"y_{split}"], arrays[f"pred_{split}"], levels)
               for split in ("validation", "test")}
    if levels is not None:
        cutoffs = calibrate(y_val, raw_val, levels)
        arrays.update(levels=levels, cutoffs=cutoffs)
        for split in ("validation", "test"):
            calibrated = levels[np.searchsorted(cutoffs, arrays[f"pred_{split}"], side="right")]
            arrays[f"pred_{split}_calibrated"] = calibrated
            results[split]["qwk_calibrated"] = float(cohen_kappa_score(
                arrays[f"y_{split}"], calibrated, labels=levels, weights="quadratic"
            ))
    for values in results.values():
        for key, value in values.items():
            if not np.isfinite(value):
                values[key] = None
    return results, arrays


def diagnostic(base, view, y_lab, classification, seed):
    """Read weak, ICS and equal-cosine rotations through one supervised probe."""
    estimator = LogisticRegression(max_iter=1000) if classification else Ridge()
    probe = make_pipeline(StandardScaler(), estimator).fit(base["e_lab"], y_lab)
    weak = torch.from_numpy(base["e_unl"])
    generator = torch.Generator().manual_seed(seed)
    weak_readout = probe.predict(weak.numpy())
    weak_teacher = base["t_unl"]
    if classification:
        weak_teacher = weak_teacher.argmax(1)
    values = {"cosine_ics": [], "cosine_angular": [], "drift_ics": [], "drift_angular": [], "drift_teacher": []}
    for strong_np, strong_teacher in zip(view["embedding"], view["teacher"], strict=True):
        strong = torch.from_numpy(strong_np)
        c = F.cosine_similarity(weak, strong, dim=1).clamp(-1, 1)
        valid = (weak.norm(dim=1) > 1e-6) & (strong.norm(dim=1) > 1e-6)
        rotated = rotate(weak, torch.randn(weak.shape, generator=generator), c[:, None])
        measured = F.cosine_similarity(weak, rotated, dim=1)
        if not torch.allclose(c[valid], measured[valid], atol=1e-4, rtol=0):
            raise RuntimeError("Cosine rotasi tidak cocok dengan ICS")
        strong_readout = probe.predict(strong_np)
        angular_readout = probe.predict(rotated.numpy())
        if classification:
            strong_teacher = strong_teacher.argmax(1)
        values["cosine_ics"].append(np.where(valid, c.numpy(), np.nan))
        values["cosine_angular"].append(np.where(valid, measured.numpy(), np.nan))
        values["drift_ics"].append(np.abs(strong_readout - weak_readout) if not classification else strong_readout != weak_readout)
        values["drift_angular"].append(np.abs(angular_readout - weak_readout) if not classification else angular_readout != weak_readout)
        values["drift_teacher"].append(np.abs(strong_teacher - weak_teacher) if not classification else strong_teacher != weak_teacher)
    return {name: np.stack(rows) for name, rows in values.items()}


def train_head(name, task, dataset, levels, y_lab, y_val, base, labeled_views, unlabeled_views, seed, steps, device):
    torch.manual_seed(seed)
    lab_rng = torch.Generator(device=device).manual_seed(seed)
    unl_rng = torch.Generator(device=device).manual_seed(seed + 1)
    classification = task == "classification"
    tensor = lambda x: torch.as_tensor(x, device=device)
    xl, xu, xv, xt = (tensor(base[f"e_{split}"]) for split in ("lab", "unl", "val", "test"))
    vl = tensor(labeled_views) if labeled_views is not None else None
    vu = tensor(unlabeled_views) if unlabeled_views is not None else None
    tu = tensor(base["t_unl"])
    pseudo_donor, donor_conf = None, None
    if name == "head_tabcutmix":
        if classification:
            donor_conf, pseudo_donor = tu.max(1)
        else:
            raw = base["t_unl"]
            bins = (np.abs(raw[:, None] - levels).argmin(1) if levels is not None
                    else np.searchsorted(np.quantile(raw, np.linspace(0.1, 0.9, 9)), raw))
            pseudo_donor, donor_conf = tensor(bins), torch.ones(len(tu), device=device)
    n_out = tu.shape[1] if classification else 1
    target_mean, target_sd = 0.0, 1.0
    if not classification:
        target_mean, target_sd = float(y_lab.mean()), float(y_lab.std() + 1e-6)
        y_lab = (y_lab - target_mean) / target_sd
        tu = (tu - target_mean) / target_sd
    yl = tensor(y_lab)
    body = nn.Sequential(nn.Linear(xl.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU())
    model = nn.Sequential(body, nn.Linear(256, n_out)).to(device)
    ema = deepcopy(model).eval().requires_grad_(False) if name in MATCH_VARIANTS else None
    latent_op = ("gaussian", 0.5) if name == "head_match_gaussian" else LATENT_DISTILL.get(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(x):
        out = model(x)
        return out if classification else out[:, 0]

    @torch.no_grad()
    def predict(x):
        model.eval()
        out = forward(x)
        model.train()
        return (out.softmax(1) if classification else out * target_sd + target_mean).cpu().numpy()

    def supervised_loss(pred, target):
        return F.cross_entropy(pred, target) if classification else F.mse_loss(pred, target)

    def get_view(views, original, idx, include_original=False):
        if views is None:
            return original[idx]
        pick = int(aug_rng.integers(len(views) + include_original))
        return original[idx] if pick == len(views) else views[pick, idx]

    metric, sign = METRIC[dataset]
    best_score, best_state, best_step, bad_checks = -np.inf, None, 0, 0
    use_unlabeled = name not in ("head_sup", "ics_sup")
    aug_rng = np.random.default_rng(seed)
    aug_cos = {"labeled_embedding": [], "unlabeled_embedding": [], "unlabeled_hidden": []}
    match_count = torch.zeros((), device=device)
    best_accept_rate = None
    for step in range(1, steps + 1):
        il = torch.randint(len(xl), (min(32, len(xl)),), device=device, generator=lab_rng)
        labeled = get_view(vl, xl, il, include_original=True)
        loss = supervised_loss(forward(labeled), yl[il])
        if vl is not None:
            aug_cos["labeled_embedding"].append(F.cosine_similarity(xl[il], labeled).mean())
        if use_unlabeled:
            iu = torch.randint(len(xu), (224,), device=device, generator=unl_rng)
            strong = get_view(vu, xu, iu)
            if vu is not None:
                aug_cos["unlabeled_embedding"].append(F.cosine_similarity(xu[iu], strong).mean())
            hidden = body(strong)
            if latent_op:
                op, strength = latent_op
                donor = (pseudo_donor[iu], donor_conf[iu]) if pseudo_donor is not None else ()
                augmented = perturb(op, hidden[:, None], strength, *donor)[:, 0]
                with torch.no_grad():
                    aug_cos["unlabeled_hidden"].append(F.cosine_similarity(hidden, augmented).mean())
            else:
                augmented = hidden
            out = model[1](augmented)
            if ema is not None:
                with torch.no_grad():
                    weak = ema(xu[iu])
                if classification:
                    confidence, pseudo = weak.softmax(1).max(1)
                    mask = confidence >= CONFIDENCE
                    match_count += mask.sum()
                    unsup = (F.cross_entropy(out, pseudo, reduction="none") * mask).mean()
                else:
                    unsup = F.mse_loss(out[:, 0], weak[:, 0])
            elif classification:
                unsup = -(tu[iu] * F.log_softmax(out, dim=1)).sum(1).mean()
            else:
                unsup = F.mse_loss(out[:, 0], tu[iu])
            loss = loss + unsup
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters(), strict=True):
                    average.lerp_(current, 1 - EMA_DECAY)

        if step % 50 == 0 or step == steps:
            pred = predict(xv)
            if levels is not None:
                cutoffs = calibrate(y_val, pred, levels)
                score = cohen_kappa_score(y_val, levels[np.searchsorted(cutoffs, pred, side="right")],
                                          labels=levels, weights="quadratic")
            else:
                score = scores(task, y_val, pred.argmax(1) if classification else pred)[metric]
            if np.isfinite(score) and sign * score > best_score:
                best_score, best_step, bad_checks = sign * score, step, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                if ema is not None and classification:
                    best_accept_rate = float(match_count / (224 * step))
            else:
                bad_checks += 1
                if bad_checks >= 20:
                    break
    if best_state is None:
        raise RuntimeError(f"Semua skor validation tidak finite untuk {dataset} {name}")
    model.load_state_dict(best_state)
    info = {"best_step": best_step, "steps_run": step,
            "cosine": {key: float(torch.stack(vals).mean()) for key, vals in aug_cos.items() if vals}}
    if ema is not None and classification:
        info["unlabeled_accept_rate"] = float(match_count / (224 * step))
        info["unlabeled_accept_rate_through_checkpoint"] = best_accept_rate
    return predict(xv), predict(xt), info


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def read_jsonl(path, repair_tail=False):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or line.endswith("\n"):
                raise ValueError(f"JSONL rusak: {path}:{i + 1}") from None
            if repair_tail:
                tmp = path.with_suffix(".tmp")
                tmp.write_text("".join(lines[:i]))
                tmp.replace(path)
        else:
            if repair_tail and not line.endswith("\n"):
                with path.open("a") as sink:
                    sink.write("\n")
    return rows


def group_key(dataset, percent, seed):
    return f"{dataset}_{percent:02d}_s{seed}"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def split_files(root):
    return [root / f"label_{p:02d}" / dataset / file
            for p in PERCENTS for dataset in DATASETS
            for file in ("manifest.json", *(f"{split}.npz" for split in SPLITS))]


def run_group(args, profile, dataset, percent, seed, result_path, done):
    names = eligible(args.names, percent)
    task, splits, levels = read_data(args.data, dataset, percent)
    fingerprint = {
        "settings": args.config,
        "group": [dataset, percent, seed],
    }
    key = group_key(dataset, percent, seed)
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    folder = profile / "cache" / f"{key}_{signature}"
    todo = [
        name for name in names
        if (key, signature, name) not in done or
        not (profile / "prediksi" / key / f"{name}.npz").is_file()
    ]
    if not todo:
        return
    indices = choose_unlabeled(len(splits["train_unlabeled"][0]), args.max_unlabeled, seed)
    new_model = lambda: make_model(task, seed, args.weights, args.device)
    base = cached(folder / "base.npz", lambda: build_base(new_model, task, splits, indices, seed, args.chunk))
    required = {"ics_distill_90": "90", "ics_distill_70": "70", "ics_distill_50": "50", "ics_distill_ta": "ta",
                "ics_match_70": "70", "ics_both_70": "70"}
    settings = {required[name] for name in names if name in required}
    if "ics_sup" in names or "ics_both_70" in names:
        settings.add("lab70")
    views, diagnostic_paths = {}, {}
    prep_seconds = float(base["build_seconds"])
    y_lab = splits["train_labeled"][1]
    for setting in sorted(settings):
        view = cached(folder / f"ics_{setting}.npz", lambda: build_views(
            new_model, task, splits, indices, seed, args.chunk, args.views, setting
        ))
        prep_seconds += float(view["build_seconds"])
        views[setting] = view["embedding"]
        if setting != "lab70":
            path = profile / "diagnostik" / key / f"ics_{setting}.npz"
            if not path.exists():
                diagnostic_arrays = diagnostic(base, view, y_lab, task == "classification", seed)
                atomic_npz(path, {
                    **diagnostic_arrays, "context_sizes": view["context_sizes"], "rho": view["rho"],
                    "unlabeled_rows": indices,
                }, compress=True)
            diagnostic_paths[setting] = str(path.relative_to(profile))
    y_val, y_test = splits["validation"][1], splits["test"][1]
    for name in todo:
        start = time.monotonic()
        if name == "tabpfn":
            pv, pt, info = base["t_val"], base["t_test"], {"best_step": None}
        else:
            lv = views["lab70"] if name in ("ics_sup", "ics_both_70") else None
            uv = views.get(required.get(name))
            pv, pt, info = train_head(name, task, dataset, levels, y_lab, y_val, base, lv, uv, seed, args.steps, args.device)
        result, arrays = evaluate(task, levels, y_val, y_test, pv, pt)
        pred_path = profile / "prediksi" / key / f"{name}.npz"
        atomic_npz(pred_path, arrays, compress=True)
        row = {
            "name": name, "dataset": dataset, "label_percent": percent, "seed": seed, "task": task,
            "ssl_method": ("label_only" if name in SUPERVISED else
                           "ema_self_training" if name in MATCH_VARIANTS else "tabpfn_distillation"),
            **result, **info, "seconds": time.monotonic() - start, "tabpfn_seconds": prep_seconds,
            "tabpfn_version": version("tabpfn"), "weights": WEIGHT_NAME, "weights_sha256": args.weights_sha256,
            "data_and_config": signature, "n_unlabeled": len(indices), "predictions": str(pred_path.relative_to(profile)),
            "diagnostic": diagnostic_paths.get(required.get(name)),
        }
        with result_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        metric, _ = METRIC[dataset]
        print(f"{key} {name:15s} val {result['validation'][metric]:.4f} test {result['test'][metric]:.4f}", flush=True)


def partition(groups, count):
    # Stage 7's measured cost makes the three high-context Superconductivity groups dominate runtime.
    cost = {"nhanes": {1: 12, 10: 21, 30: 27}, "pendigits": {1: 8, 10: 13, 30: 18},
            "superconductivity": {1: 80, 10: 225, 30: 715}}
    batches, totals = [[] for _ in range(count)], [0] * count
    for group in sorted(groups, key=lambda group: -cost[group[0]][group[1]]):
        i = totals.index(min(totals))
        batches[i].append(group)
        totals[i] += cost[group[0]][group[1]]
    return [sorted(batch, key=groups.index) for batch in batches]


def parse_list(text, allowed):
    values = list(dict.fromkeys(text.split(",")))
    lookup = {str(value): value for value in allowed}
    if any(value not in lookup for value in values):
        raise SystemExit(f"Pilihan {text!r} tidak valid. Pilih dari {', '.join(lookup)}")
    return [lookup[value] for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("DATA_DIR"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/tahap8"))
    parser.add_argument("--weights", default=os.getenv("TABPFN_WEIGHTS"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--views", type=int)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--max-unlabeled", type=int, help="0 = semua baris")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--percents", default=",".join(map(str, PERCENTS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--names", default="core", help="core, all, atau nama varian dipisah koma")
    args = parser.parse_args()
    args.names = experiment_names(args.names)
    datasets = parse_list(args.datasets, DATASETS)
    percents = parse_list(args.percents, PERCENTS)
    seeds = parse_list(args.seeds, SEEDS)
    default_steps, default_views, default_limit = (200, 2, 512) if args.smoke else (3000, 4, 5000)
    args.steps = default_steps if args.steps is None else args.steps
    args.views = default_views if args.views is None else args.views
    args.max_unlabeled = default_limit if args.max_unlabeled is None else args.max_unlabeled
    if min(args.steps, args.views, args.chunk) < 1 or args.max_unlabeled < 0:
        parser.error("steps, views, chunk harus positif, max-unlabeled minimal 0")
    args.data = find_data(args.data)
    args.out = args.out.resolve()
    profile = args.out / ("smoke" if args.smoke else "full")
    profile.mkdir(parents=True, exist_ok=True)
    groups = [(d, p, s) for p in percents for d in datasets for s in seeds if eligible(args.names, p)]
    if args.smoke:
        smoke_groups = {("nhanes", 10, 42), ("pendigits", 1, 42), ("superconductivity", 30, 42)}
        groups = [group for group in groups if group in smoke_groups]
    if not groups:
        parser.error("Tidak ada grup yang cocok dengan pilihan ini")

    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if args.worker is None:
        available = torch.cuda.device_count() if args.device == "cuda" else 1
        count = args.gpus if args.gpus is not None else min(available, 1 if args.smoke else 2)
        if not 1 <= count <= available:
            parser.error(f"Diminta {count} worker, perangkat tersedia {available}. Aktifkan GPU Kaggle atau pakai --device cpu.")
    else:
        count = args.gpus
        if args.worker not in range(count):
            parser.error("worker di luar rentang")

    if version("tabpfn") != TABPFN_VERSION:
        parser.error(f"Pakai tabpfn=={TABPFN_VERSION} agar sebanding dengan tahap 7 v3.5")
    args.weights = weights(args.out, args.weights)
    args.weights_sha256 = digest(args.weights)
    data_sha256 = hashlib.sha256("".join(digest(path) for path in split_files(args.data)).encode()).hexdigest()
    args.config = {
        "schema": SCHEMA, "steps": args.steps, "views": args.views, "max_unlabeled": args.max_unlabeled,
        "chunk": args.chunk, "device": args.device, "tabpfn_version": TABPFN_VERSION,
        "weights_sha256": args.weights_sha256, "data_sha256": data_sha256,
        "ema_decay": EMA_DECAY, "pseudolabel_confidence": CONFIDENCE,
    }
    if args.worker is None:
        with (profile / ".run.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error(f"Training lain masih menulis ke {profile}")
            config_path = profile / "config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != args.config:
                parser.error("Konfigurasi pada folder hasil sudah berbeda. Gunakan --out baru.")
            save_json(config_path, args.config)
            for path in profile.glob("results_gpu*.jsonl"):
                read_jsonl(path, repair_tail=True)
            gpu_ids = visible.split(",") if visible else list(map(str, range(count)))
            if count == 1:
                run_worker(args, profile, groups, 0)
                return
            procs = [subprocess.Popen(
                [sys.executable, __file__, *sys.argv[1:], "--worker", str(i), "--gpus", str(count),
                 "--weights", str(args.weights)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids[i]},
            ) for i in range(count)]
            try:
                codes = [proc.wait() for proc in procs]
            finally:
                for proc in procs:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
            if any(codes):
                raise SystemExit(f"Worker gagal dengan exit code {codes}")
    else:
        run_worker(args, profile, partition(groups, count)[args.worker], args.worker)


def run_worker(args, profile, groups, worker):
    done = {(group_key(row["dataset"], row["label_percent"], row["seed"]), row["data_and_config"], row["name"])
            for path in profile.glob("results_gpu*.jsonl") for row in read_jsonl(path)}
    result_path = profile / f"results_gpu{worker}.jsonl"
    print(f"Worker {worker}: {len(groups)} grup, {sum(len(eligible(args.names, p)) for _, p, _ in groups)} run", flush=True)
    for dataset, percent, seed in groups:
        run_group(args, profile, dataset, percent, seed, result_path, done)


if __name__ == "__main__":
    main()
