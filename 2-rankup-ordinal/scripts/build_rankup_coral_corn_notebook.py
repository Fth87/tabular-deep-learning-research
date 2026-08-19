import json
from pathlib import Path
from textwrap import dedent


BASE = Path("kode/rankup_all_datasets.ipynb")
OUTPUT = Path("kode/rankup_all_datasets_coral_corn.ipynb")


def lines(text):
    return dedent(text).strip("\n").splitlines(keepends=True)


notebook = json.loads(BASE.read_text())


def replace(index, text, cell_type=None):
    cell = notebook["cells"][index]
    cell["source"] = lines(text)
    if cell_type:
        cell["cell_type"] = cell_type
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


replace(
    0,
    """
    # RankUp pada Semua Dataset

    Notebook Kaggle ini mempertahankan eksperimen **MLP**, **TabM**, dan **RealMLP** pada lima dataset ordinal, lalu membandingkan tiga objective:

    1. **Regression** — objective L1 dari notebook acuan.
    2. **CORAL** — cumulative ordinal regression dengan shared rank logit.
    3. **CORN** — conditional ordinal regression.

    Setiap objective menjalankan **Supervised**, **Fully-Supervised**, **RankUp + FESTA**, dan **RankUp + TabCutMix** dengan satu seed yang sama.

    Referensi implementasi: [paper CORAL](https://arxiv.org/abs/1901.07884), [paper CORN](https://arxiv.org/abs/2111.08851), [coral-pytorch](https://raschka-research-group.github.io/coral-pytorch/), dan [TabM](https://github.com/yandex-research/tabm).
    """,
)

replace(
    2,
    r'''
    %pip install -q tabm coral-pytorch

    import gc
    import random
    import time
    from itertools import cycle, islice
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from coral_pytorch.layers import CoralLayer
    from coral_pytorch.losses import coral_loss, corn_loss
    from scipy.stats import spearmanr
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tabm import TabM
    from torch.utils.data import DataLoader, TensorDataset

    sns.set_theme(style="whitegrid", context="notebook")

    SEED = 42
    N_LABEL = 200
    EPOCHS = 120
    BATCH_SIZE = 32
    SUPERVISED_BATCH_SIZE = 256
    MAX_RANKUP_STEPS = 100
    LOG_EVERY = 10
    W_RANK = 0.2
    W_UNLABELED = 0.2
    P_CUTOFF = 0.95
    FESTA_BINS = 10
    TSNE_MAX_SAMPLES = 500

    DATASET_FILES = {
        "Melbourne Airbnb": "oc10_melbourneAirbnb.csv",
        "Student Performance": "oc08_studentPerformance.csv",
        "NHANES": "oc05_nhanes.csv",
        "ESL": "oc06_esl.csv",
        "LEV": "oc05_lev.csv",
    }
    BACKBONES = ["MLP", "TabM", "RealMLP"]
    OBJECTIVES = ["Regression", "CORAL", "CORN"]
    EXPERIMENTS = ["Supervised", "Fully-Supervised", "RankUp + FESTA", "RankUp + TabCutMix"]

    WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    OUTPUT_DIR = WORK_DIR / "rankup_coral_corn_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    HARDWARE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | jumlah GPU tersedia: {torch.cuda.device_count()}")
    ''',
)

replace(
    6,
    '''
    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def prepare_data(df):
        X = df.drop(columns="y").to_numpy(dtype=np.float32)
        y_class = df["y"].to_numpy(dtype=np.int64)
        y_float = y_class.astype(np.float32)
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=SEED, stratify=y_class
        )
        if N_LABEL >= len(train_idx):
            raise ValueError(f"N_LABEL={N_LABEL} harus lebih kecil dari train={len(train_idx)}")
        labeled_idx, unlabeled_idx = train_test_split(
            train_idx, train_size=N_LABEL, random_state=SEED, stratify=y_class[train_idx]
        )

        x_scaler = StandardScaler().fit(X[train_idx])
        y_scaler = StandardScaler().fit(y_float[labeled_idx, None])
        X_scaled = x_scaler.transform(X).astype(np.float32)
        y_scaled = y_scaler.transform(y_float[:, None]).astype(np.float32)

        tensor = torch.from_numpy
        return {
            "x_labeled": tensor(X_scaled[labeled_idx]),
            "y_labeled_scaled": tensor(y_scaled[labeled_idx]),
            "y_labeled_class": tensor(y_class[labeled_idx]),
            "x_unlabeled": tensor(X_scaled[unlabeled_idx]),
            "x_train": tensor(X_scaled[train_idx]),
            "y_train_scaled": tensor(y_scaled[train_idx]),
            "y_train_class": tensor(y_class[train_idx]),
            "x_test": tensor(X_scaled[test_idx]),
            "y_test": y_class[test_idx],
            "y_scaler": y_scaler,
            "n_features": X.shape[1],
            "num_classes": len(np.unique(y_class)),
            "class_min": int(y_class.min()),
            "class_max": int(y_class.max()),
        }


    prepared = {name: prepare_data(df) for name, df in datasets.items()}
    for name, data in prepared.items():
        print(f"{name:20} | labeled={len(data['x_labeled'])} | unlabeled={len(data['x_unlabeled'])} | test={len(data['x_test'])}")
    ''',
)

replace(
    7,
    """
    ## 4. Model

    Setiap backbone mempertahankan encoder dan ranking head notebook acuan. Prediction head dipilih berdasarkan objective: satu output Regression, `K-1` threshold CORAL, atau `K-1` conditional logits CORN.
    """,
)

replace(
    8,
    '''
    class MLP(nn.Module):
        def __init__(self, n_features, num_classes, objective):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
            )
            self.predictor = (
                nn.Linear(32, 1) if objective == "Regression"
                else CoralLayer(32, num_classes) if objective == "CORAL"
                else nn.Linear(32, num_classes - 1)
            )
            self.ranker = nn.Linear(32, 2)

        def forward(self, x):
            z = self.encode(x)
            return self.predictor(z), self.ranker(z)

        def encode(self, x):
            return self.encoder(x)


    class NTPLinear(nn.Module):
        def __init__(self, in_features, out_features, zero_init=False):
            super().__init__()
            factor = 0.0 if zero_init else 1.0
            self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
            self.bias = nn.Parameter(factor * torch.randn(1, out_features))
            self.scale = in_features ** -0.5

        def forward(self, x):
            return self.scale * (x @ self.weight) + self.bias


    class RealMLP(nn.Module):
        def __init__(self, n_features, num_classes, objective):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(n_features))
            layers = []
            width = n_features
            for _ in range(3):
                layers += [NTPLinear(width, 256), nn.Mish()]
                width = 256
            self.encoder = nn.Sequential(*layers)
            self.predictor = (
                NTPLinear(width, 1, zero_init=True) if objective == "Regression"
                else CoralLayer(width, num_classes) if objective == "CORAL"
                else NTPLinear(width, num_classes - 1, zero_init=True)
            )
            self.ranker = NTPLinear(width, 2, zero_init=True)

        def forward(self, x):
            z = self.encode(x)
            return self.predictor(z), self.ranker(z)

        def encode(self, x):
            return self.encoder(x * self.scale)


    class TabMRankUp(nn.Module):
        def __init__(self, n_features, num_classes, objective):
            super().__init__()
            self.objective = objective
            self.num_classes = num_classes
            prediction_outputs = num_classes - 1 if objective == "CORN" else 1
            self.model = TabM.make(n_num_features=n_features, d_out=prediction_outputs + 2)
            self.coral = CoralLayer(1, num_classes) if objective == "CORAL" else None

        def forward(self, x):
            output = self.model(x)
            prediction_outputs = self.num_classes - 1 if self.objective == "CORN" else 1
            prediction = output[..., :prediction_outputs]
            if self.coral is not None:
                shape = prediction.shape[:2]
                prediction = self.coral(prediction.reshape(-1, 1)).reshape(*shape, -1)
            return prediction, output[..., prediction_outputs:]

        def encode(self, x):
            z = self.model.backbone(self.model.ensemble_view(x))
            return z.mean(dim=1)
    ''',
)

replace(
    11,
    """
    ## 6. Loss RankUp dan evaluasi

    Regression memakai L1 pada target terstandardisasi. CORAL dan CORN memakai loss resmi `coral-pytorch` pada kelas asli. Ranking loss tetap sama untuk ketiga objective.
    """,
)

replace(
    12,
    '''
    def regression_loss(prediction, target):
        if prediction.ndim == 3:
            target = target[:, None, :].expand_as(prediction)
        return F.l1_loss(prediction, target)


    def coral_levels(labels, num_classes):
        thresholds = torch.arange(num_classes - 1, device=labels.device)
        return (thresholds[None, :] < labels[:, None]).float()


    def prediction_loss(logits, y_scaled, y_class, objective, num_classes):
        if objective == "Regression":
            return regression_loss(logits, y_scaled)
        if logits.ndim == 3:
            k = logits.shape[1]
            logits = logits.flatten(0, 1)
            y_class = y_class.repeat_interleave(k)
        if objective == "CORAL":
            return coral_loss(logits, coral_levels(y_class, num_classes))
        if objective == "CORN":
            return corn_loss(logits, y_class, num_classes)
        raise ValueError(f"Objective tidak dikenal: {objective}")


    def ordinal_probabilities(logits, objective):
        probability = torch.sigmoid(logits)
        return probability if objective == "CORAL" else probability.cumprod(dim=-1)


    def ordinal_score(logits, objective):
        probability = ordinal_probabilities(logits, objective)
        if probability.ndim == 3:
            probability = probability.mean(dim=1)
        return probability.sum(dim=-1)


    def pairwise_logits(rank_logits):
        if rank_logits.ndim == 2:
            return rank_logits[:, None, :] - rank_logits[None, :, :]
        rank_logits = rank_logits.permute(1, 0, 2)
        return rank_logits[:, :, None, :] - rank_logits[:, None, :, :]


    def pairwise_targets(target):
        difference = target[:, None, 0] - target[None, :, 0]
        probability = (difference > 0).float() + 0.5 * (difference == 0).float()
        return torch.stack([1 - probability, probability], dim=-1)


    def supervised_rank_loss(rank_logits, target):
        logits = pairwise_logits(rank_logits)
        targets = pairwise_targets(target)
        if logits.ndim == 4:
            targets = targets.unsqueeze(0).expand(logits.shape[0], -1, -1, -1)
        return F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1, 2))


    def pseudo_rank_loss(weak_logits, strong_logits):
        weak_pairs = pairwise_logits(weak_logits)
        weak_probability = F.softmax(weak_pairs, dim=-1)
        if weak_probability.ndim == 4:
            weak_probability = weak_probability.mean(0)
        confidence, pseudo_target = weak_probability.max(dim=-1)

        strong_pairs = pairwise_logits(strong_logits)
        if strong_pairs.ndim == 4:
            pseudo_target = pseudo_target.unsqueeze(0).expand(strong_pairs.shape[0], -1, -1)
            confidence = confidence.unsqueeze(0).expand(strong_pairs.shape[0], -1, -1)
        loss = F.cross_entropy(
            strong_pairs.reshape(-1, 2), pseudo_target.reshape(-1), reduction="none"
        )
        mask = confidence.reshape(-1) >= P_CUTOFF
        return (loss[mask].mean() if mask.any() else loss.sum() * 0), mask.float().mean().item()


    def evaluate(y_true, score, class_min, class_max):
        prediction = np.clip(np.rint(score), class_min, class_max).astype(int)
        spearman = spearmanr(y_true, score).statistic
        return {
            "R2": r2_score(y_true, score),
            "Spearman": 0.0 if np.isnan(spearman) else spearman,
            "MAE": mean_absolute_error(y_true, score),
            "RMSE": mean_squared_error(y_true, score) ** 0.5,
            "QWK": cohen_kappa_score(y_true, prediction, weights="quadratic"),
        }


    demo_x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    demo_y = torch.tensor([[0.0], [1.0], [2.0]])
    demo_labels = torch.tensor([0, 2, 4])
    demo_logits = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
    corn_probability = ordinal_probabilities(demo_logits, "CORN")
    assert pairwise_targets(demo_y)[2, 0, 1] == 1
    assert coral_levels(demo_labels, 5).tolist() == [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    assert torch.all(corn_probability[:, 1:] <= corn_probability[:, :-1])
    assert 0 <= ordinal_score(demo_logits, "CORAL").item() <= 4
    assert festa_view(demo_x, festa_edges(demo_x, 2), 0.5).shape == demo_x.shape
    assert tabcutmix_view(demo_x, 0.5).shape == demo_x.shape
    print("Self-check Regression, CORAL, CORN, RankUp, dan augmentasi: OK")
    ''',
)

replace(
    13,
    """
    ## 7. DataLoader dan inference

    Regression dikembalikan melalui inverse target scaler. CORAL dan CORN menghasilkan expected ordinal score dari cumulative probabilities. Pada TabM, probabilitas anggota ensemble dirata-ratakan sebelum skor dihitung.
    """,
)

replace(
    14,
    '''
    def make_loader(*tensors, batch_size=BATCH_SIZE, shuffle=True):
        return DataLoader(
            TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(SEED), pin_memory=torch.cuda.is_available(),
        )


    def predict(model, data, objective):
        loader = make_loader(data["x_test"], batch_size=1024, shuffle=False)
        scores = []
        model.eval()
        with torch.inference_mode():
            for (x,) in loader:
                logits, _ = model(x.to(DEVICE, non_blocking=True))
                if objective == "Regression":
                    if logits.ndim == 3:
                        logits = logits.mean(dim=1)
                    scores.append(logits.cpu().numpy())
                else:
                    scores.append(ordinal_score(logits, objective).cpu().numpy())
        score = np.concatenate(scores)
        if objective == "Regression":
            return data["y_scaler"].inverse_transform(score.reshape(-1, 1)).ravel()
        return score.ravel()


    def extract_features(model, data):
        loader = make_loader(data["x_test"], batch_size=1024, shuffle=False)
        features = []
        model.eval()
        with torch.inference_mode():
            for (x,) in loader:
                features.append(model.encode(x.to(DEVICE, non_blocking=True)).cpu().numpy())
        return np.concatenate(features)
    ''',
)

replace(
    16,
    '''
    results, predictions, histories = [], [], []
    representations = {}
    results_df = predictions_df = history_df = pd.DataFrame()
    ''',
)

replace(
    17,
    """
    ## 9. Helper visualisasi

    Visualisasi tetap sama seperti notebook acuan, dengan satu baris per objective agar Regression, CORAL, dan CORN tidak bertumpuk.
    """,
)

replace(
    18,
    '''
    PALETTE = dict(zip(EXPERIMENTS, sns.color_palette("colorblind", len(EXPERIMENTS))))
    METRICS = ["R2", "Spearman", "MAE", "RMSE", "QWK"]


    def show_model_summary(backbone):
        columns = ["dataset", "objective", "experiment", *METRICS, "training_seconds", "inference_seconds"]
        table = results_df[results_df["backbone"] == backbone][columns]
        display(table.reset_index(drop=True).style.format({
            **{metric: "{:.4f}" for metric in METRICS},
            "training_seconds": "{:.1f}", "inference_seconds": "{:.3f}",
        }))


    def plot_metrics(backbone):
        for dataset_name in DATASET_FILES:
            subset = results_df.query("backbone == @backbone and dataset == @dataset_name")
            figure, axes = plt.subplots(3, 5, figsize=(24, 13), squeeze=False)
            for row, objective in enumerate(OBJECTIVES):
                objective_data = subset[subset["objective"] == objective]
                for axis, metric in zip(axes[row], METRICS):
                    sns.barplot(
                        data=objective_data, x="experiment", y=metric, hue="experiment",
                        order=EXPERIMENTS, hue_order=EXPERIMENTS, palette=PALETTE,
                        legend=False, ax=axis,
                    )
                    axis.set_title(metric + (" ↓" if metric in ["MAE", "RMSE"] else " ↑"))
                    axis.set_xlabel("")
                    axis.tick_params(axis="x", rotation=25)
                    if axis is axes[row, 0]:
                        axis.set_ylabel(objective)
            figure.suptitle(f"{backbone} — {dataset_name}", y=1.01, fontsize=15)
            figure.tight_layout()
            filename = DATASET_FILES[dataset_name][:-4]
            figure.savefig(OUTPUT_DIR / f"metrics_{backbone.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()


    def plot_training(backbone):
        for dataset_name in DATASET_FILES:
            subset = history_df.query("backbone == @backbone and dataset == @dataset_name")
            figure, axes = plt.subplots(3, 3, figsize=(18, 13), squeeze=False)
            for row, objective in enumerate(OBJECTIVES):
                objective_data = subset[subset["objective"] == objective]
                rankup = objective_data[objective_data["experiment"].str.startswith("RankUp")]
                rank_losses = rankup.melt(
                    id_vars=["epoch", "experiment"],
                    value_vars=["rank_labeled", "rank_unlabeled"],
                    var_name="component", value_name="value",
                )
                sns.lineplot(data=objective_data, x="epoch", y="loss", hue="experiment", palette=PALETTE, ax=axes[row, 0])
                sns.lineplot(data=rank_losses, x="epoch", y="value", hue="experiment", style="component", palette=PALETTE, ax=axes[row, 1])
                sns.lineplot(data=rankup, x="epoch", y="accepted", hue="experiment", palette=PALETTE, ax=axes[row, 2])
                axes[row, 0].set_ylabel(objective)
                axes[row, 0].set_title("Total training loss")
                axes[row, 1].set_title("Breakdown ranking loss")
                axes[row, 2].set_title("Pasangan pseudo-label diterima")
                axes[row, 2].set_ylim(0, 1)
            figure.suptitle(f"{backbone} — {dataset_name}", y=1.01, fontsize=15)
            figure.tight_layout()
            filename = DATASET_FILES[dataset_name][:-4]
            figure.savefig(OUTPUT_DIR / f"loss_{backbone.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()


    def plot_predictions(backbone):
        for dataset_name in DATASET_FILES:
            subset = predictions_df.query("backbone == @backbone and dataset == @dataset_name")
            figure, axes = plt.subplots(3, 2, figsize=(14, 17), squeeze=False)
            for row, objective in enumerate(OBJECTIVES):
                objective_data = subset[subset["objective"] == objective]
                low = objective_data[["y_true", "y_score"]].min().min()
                high = objective_data[["y_true", "y_score"]].max().max()
                sns.scatterplot(data=objective_data, x="y_true", y="y_score", hue="experiment", palette=PALETTE, alpha=0.45, ax=axes[row, 0])
                axes[row, 0].plot([low, high], [low, high], "--", color="black", linewidth=1)
                axes[row, 0].set_title(f"{objective}: prediksi vs target")
                sns.histplot(data=objective_data, x="residual", hue="experiment", palette=PALETTE, kde=True, bins=20, alpha=0.35, ax=axes[row, 1])
                axes[row, 1].axvline(0, linestyle="--", color="black", linewidth=1)
                axes[row, 1].set_title(f"{objective}: distribusi residual")
            figure.suptitle(f"{backbone} — {dataset_name}", y=1.01, fontsize=15)
            figure.tight_layout()
            filename = DATASET_FILES[dataset_name][:-4]
            figure.savefig(OUTPUT_DIR / f"prediction_{backbone.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()


    def plot_tsne(backbone):
        rng = np.random.default_rng(SEED)
        for dataset_name, data in prepared.items():
            n_samples = min(TSNE_MAX_SAMPLES, len(data["y_test"]))
            sample_idx = np.sort(rng.choice(len(data["y_test"]), n_samples, replace=False))
            targets = data["y_test"][sample_idx]
            figure, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True, squeeze=False)
            for row, objective in enumerate(OBJECTIVES):
                for column, experiment in enumerate(EXPERIMENTS):
                    features = representations[(dataset_name, backbone, objective, experiment)][sample_idx]
                    embedding = TSNE(
                        n_components=2, perplexity=min(30, n_samples - 1), init="pca",
                        learning_rate="auto", random_state=SEED,
                    ).fit_transform(features)
                    scatter = axes[row, column].scatter(
                        embedding[:, 0], embedding[:, 1], c=targets,
                        cmap="viridis", s=18, alpha=0.8,
                    )
                    axes[row, column].set_title(f"{objective} — {experiment}")
                    axes[row, column].set_xlabel("t-SNE 1")
                    axes[row, column].set_ylabel("t-SNE 2")
            figure.colorbar(scatter, ax=axes, label="Target ordinal")
            figure.suptitle(f"Representasi fitur {backbone} — {dataset_name}", y=1.02, fontsize=15)
            filename = DATASET_FILES[dataset_name][:-4]
            figure.savefig(OUTPUT_DIR / f"tsne_{backbone.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()
    ''',
)


def supervised_cell(backbone, full=False):
    class_name = {"MLP": "MLP", "TabM": "TabMRankUp", "RealMLP": "RealMLP"}[backbone]
    suffix = "fully_supervised" if full else "supervised"
    experiment = "Fully-Supervised" if full else "Supervised"
    x_key = "x_train" if full else "x_labeled"
    scaled_key = "y_train_scaled" if full else "y_labeled_scaled"
    class_key = "y_train_class" if full else "y_labeled_class"
    batch = "SUPERVISED_BATCH_SIZE" if full else "BATCH_SIZE"
    optimizer = (
        "torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)"
        if backbone == "TabM" else "torch.optim.Adam(model.parameters(), lr=1e-3)"
    )
    return f'''
    def train_{backbone.lower()}_{suffix}(data, objective):
        set_seed()
        model = {class_name}(data["n_features"], data["num_classes"], objective).to(DEVICE)
        optimizer = {optimizer}
        loader = make_loader(
            data["{x_key}"], data["{scaled_key}"], data["{class_key}"],
            batch_size={batch},
        )
        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            for x, y_scaled, y_class in loader:
                x = x.to(DEVICE)
                y_scaled = y_scaled.to(DEVICE)
                y_class = y_class.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = prediction_loss(logits, y_scaled, y_class, objective, data["num_classes"])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            history.append({{"epoch": epoch, "loss": np.mean(losses)}})
            if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == EPOCHS:
                print(f"  epoch {{epoch:03d}}/{{EPOCHS}} | loss={{np.mean(losses):.4f}}")
        return model, history


    experiment = "{experiment}"
    for objective in OBJECTIVES:
        print(f"\\nObjective: {{objective}}")
        for dataset_name, data in prepared.items():
            print(f"\\n{backbone} | {{objective}} | {{experiment}} | {{dataset_name}}")
            started = time.perf_counter()
            model, history = train_{backbone.lower()}_{suffix}(data, objective)
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            score = predict(model, data, objective)
            inference_seconds = time.perf_counter() - inference_started
            metrics = evaluate(data["y_test"], score, data["class_min"], data["class_max"])
            representations[(dataset_name, "{backbone}", objective, experiment)] = extract_features(model, data)
            results.append({{"dataset": dataset_name, "backbone": "{backbone}", "objective": objective, "experiment": experiment, "seed": SEED, "hardware": HARDWARE, "training_seconds": training_seconds, "inference_seconds": inference_seconds, **metrics}})
            predictions.extend({{"dataset": dataset_name, "backbone": "{backbone}", "objective": objective, "experiment": experiment, "row": i, "y_true": true, "y_score": pred, "residual": pred - true}} for i, (true, pred) in enumerate(zip(data["y_test"], score)))
            histories.extend({{"dataset": dataset_name, "backbone": "{backbone}", "objective": objective, "experiment": experiment, **row}} for row in history)
            print(pd.Series(metrics).round(4).to_dict())
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)
    history_df = pd.DataFrame(histories)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)
    display(results_df.query("backbone == '{backbone}' and experiment == @experiment").tail(15))
    '''


def rankup_cell(backbone, augmentation):
    class_name = {"MLP": "MLP", "TabM": "TabMRankUp", "RealMLP": "RealMLP"}[backbone]
    slug = augmentation.lower()
    experiment = f"RankUp + {augmentation}"
    optimizer = (
        "torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)"
        if backbone == "TabM" else "torch.optim.Adam(model.parameters(), lr=1e-3)"
    )
    view_setup = 'edges = festa_edges(data["x_train"]).to(DEVICE)' if augmentation == "FESTA" else ""
    weak = "festa_view(x_u, edges, 0.1)" if augmentation == "FESTA" else "tabcutmix_view(x_u, 0.1)"
    strong = "festa_view(x_u, edges, 0.3)" if augmentation == "FESTA" else "tabcutmix_view(x_u, 0.5)"
    return f'''
    def train_{backbone.lower()}_rankup_{slug}(data, objective):
        set_seed()
        model = {class_name}(data["n_features"], data["num_classes"], objective).to(DEVICE)
        optimizer = {optimizer}
        labeled_loader = make_loader(
            data["x_labeled"], data["y_labeled_scaled"], data["y_labeled_class"]
        )
        unlabeled_loader = make_loader(data["x_unlabeled"])
        {view_setup}
        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            totals, rank_labs, rank_unlabs, accepted = [], [], [], []
            labeled_batches = cycle(labeled_loader)
            for (x_u,) in islice(unlabeled_loader, MAX_RANKUP_STEPS):
                x_l, y_scaled, y_class = next(labeled_batches)
                x_l = x_l.to(DEVICE)
                y_scaled = y_scaled.to(DEVICE)
                y_class = y_class.to(DEVICE)
                x_u = x_u.to(DEVICE)
                weak = {weak}
                strong = {strong}

                optimizer.zero_grad(set_to_none=True)
                pred_l, rank_l = model(x_l)
                loss_prediction = prediction_loss(
                    pred_l, y_scaled, y_class, objective, data["num_classes"]
                )
                loss_rank_lab = supervised_rank_loss(rank_l, y_class.float().unsqueeze(1))
                with torch.no_grad():
                    _, rank_weak = model(weak)
                _, rank_strong = model(strong)
                loss_rank_unlab, accept_rate = pseudo_rank_loss(rank_weak, rank_strong)
                loss = loss_prediction + W_RANK * loss_rank_lab + W_UNLABELED * loss_rank_unlab
                loss.backward()
                optimizer.step()
                totals.append(loss.item())
                rank_labs.append(loss_rank_lab.item())
                rank_unlabs.append(loss_rank_unlab.item())
                accepted.append(accept_rate)
            history.append({{"epoch": epoch, "loss": np.mean(totals), "rank_labeled": np.mean(rank_labs), "rank_unlabeled": np.mean(rank_unlabs), "accepted": np.mean(accepted)}})
            if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == EPOCHS:
                print(f"  epoch {{epoch:03d}}/{{EPOCHS}} | loss={{np.mean(totals):.4f}} | accepted={{np.mean(accepted):.1%}}")
        return model, history


    experiment = "{experiment}"
    for objective in OBJECTIVES:
        print(f"\\nObjective: {{objective}}")
        for dataset_name, data in prepared.items():
            print(f"\\n{backbone} | {{objective}} | {{experiment}} | {{dataset_name}}")
            started = time.perf_counter()
            model, history = train_{backbone.lower()}_rankup_{slug}(data, objective)
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            score = predict(model, data, objective)
            inference_seconds = time.perf_counter() - inference_started
            metrics = evaluate(data["y_test"], score, data["class_min"], data["class_max"])
            representations[(dataset_name, "{backbone}", objective, experiment)] = extract_features(model, data)
            results.append({{"dataset": dataset_name, "backbone": "{backbone}", "objective": objective, "experiment": experiment, "seed": SEED, "hardware": HARDWARE, "training_seconds": training_seconds, "inference_seconds": inference_seconds, **metrics}})
            predictions.extend({{"dataset": dataset_name, "backbone": "{backbone}", "objective": objective, "experiment": experiment, "row": i, "y_true": true, "y_score": pred, "residual": pred - true}} for i, (true, pred) in enumerate(zip(data["y_test"], score)))
            histories.extend({{"dataset": dataset_name, "backbone": "{backbone}", "objective": objective, "experiment": experiment, **row}} for row in history)
            print(pd.Series(metrics).round(4).to_dict())
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)
    history_df = pd.DataFrame(histories)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)
    display(results_df.query("backbone == '{backbone}' and experiment == @experiment").tail(15))
    '''


for index, backbone, kind in [
    (21, "MLP", "supervised"), (23, "MLP", "fully"),
    (25, "MLP", "FESTA"), (27, "MLP", "TabCutMix"),
    (36, "TabM", "supervised"), (38, "TabM", "fully"),
    (40, "TabM", "FESTA"), (42, "TabM", "TabCutMix"),
    (51, "RealMLP", "supervised"), (53, "RealMLP", "fully"),
    (55, "RealMLP", "FESTA"), (57, "RealMLP", "TabCutMix"),
]:
    if kind == "supervised":
        replace(index, supervised_cell(backbone))
    elif kind == "fully":
        replace(index, supervised_cell(backbone, full=True))
    else:
        replace(index, rankup_cell(backbone, kind))

for index, backbone, function in [
    (29, "MLP", "show_model_summary"), (30, "MLP", "plot_metrics"),
    (31, "MLP", "plot_training"), (32, "MLP", "plot_predictions"), (33, "MLP", "plot_tsne"),
    (44, "TabM", "show_model_summary"), (45, "TabM", "plot_metrics"),
    (46, "TabM", "plot_training"), (47, "TabM", "plot_predictions"), (48, "TabM", "plot_tsne"),
    (59, "RealMLP", "show_model_summary"), (60, "RealMLP", "plot_metrics"),
    (61, "RealMLP", "plot_training"), (62, "RealMLP", "plot_predictions"), (63, "RealMLP", "plot_tsne"),
]:
    replace(index, f'{function}("{backbone}")')

replace(
    66,
    '''
    expected_result_rows = 180
    if len(results_df) != expected_result_rows:
        raise ValueError(f"Hasil belum lengkap: {len(results_df)}/{expected_result_rows} baris")

    comparison_columns = [
        "backbone", "objective", "experiment", "R2", "Spearman", "MAE", "RMSE", "QWK",
        "training_seconds", "inference_seconds",
    ]
    for dataset_name in DATASET_FILES:
        table = results_df[results_df["dataset"] == dataset_name][comparison_columns].copy()
        table["backbone"] = pd.Categorical(table["backbone"], BACKBONES, ordered=True)
        table["objective"] = pd.Categorical(table["objective"], OBJECTIVES, ordered=True)
        table["experiment"] = pd.Categorical(table["experiment"], EXPERIMENTS, ordered=True)
        table = table.sort_values(["backbone", "objective", "experiment"]).reset_index(drop=True)
        styled = (
            table.style.set_caption(dataset_name).format({
                **{metric: "{:.4f}" for metric in METRICS},
                "training_seconds": "{:.1f}", "inference_seconds": "{:.3f}",
            })
            .highlight_max(subset=["R2", "Spearman", "QWK"], color="#c6efce")
            .highlight_min(subset=["MAE", "RMSE"], color="#c6efce")
        )
        display(styled)
    ''',
)

replace(
    68,
    '''
    metric_titles = {
        "R2": "R² ↑", "Spearman": "Spearman ↑", "MAE": "MAE ↓",
        "RMSE": "RMSE ↓", "QWK": "QWK ↑",
    }
    for dataset_name in DATASET_FILES:
        dataset_data = results_df[results_df["dataset"] == dataset_name]
        for objective in OBJECTIVES:
            subset = dataset_data[dataset_data["objective"] == objective]
            figure, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.ravel()
            handles = labels = None
            for axis, metric in zip(axes, METRICS):
                sns.barplot(
                    data=subset, x="backbone", y=metric, hue="experiment",
                    order=BACKBONES, hue_order=EXPERIMENTS, palette=PALETTE, ax=axis,
                )
                axis.set_title(metric_titles[metric], fontweight="bold")
                axis.set_xlabel("")
                axis.grid(axis="y", alpha=0.25)
                if handles is None:
                    handles, labels = axis.get_legend_handles_labels()
                legend = axis.get_legend()
                if legend is not None:
                    legend.remove()
            axes[-1].axis("off")
            figure.legend(handles, labels, title="Eksperimen", loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
            figure.suptitle(f"{objective} — {dataset_name}", fontsize=16, fontweight="bold")
            figure.tight_layout(rect=(0, 0.08, 1, 0.95))
            filename = DATASET_FILES[dataset_name][:-4]
            figure.savefig(OUTPUT_DIR / f"comparison_metrics_{objective.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()
    ''',
)

replace(
    70,
    '''
    for dataset_name in DATASET_FILES:
        dataset_data = history_df[history_df["dataset"] == dataset_name]
        for objective in OBJECTIVES:
            subset = dataset_data[dataset_data["objective"] == objective]
            figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
            for axis, backbone in zip(axes, BACKBONES):
                sns.lineplot(
                    data=subset[subset["backbone"] == backbone], x="epoch", y="loss",
                    hue="experiment", hue_order=EXPERIMENTS, palette=PALETTE, ax=axis,
                )
                axis.set_title(backbone)
                axis.set_ylabel("Training loss")
                if axis is not axes[0]:
                    legend = axis.get_legend()
                    if legend is not None:
                        legend.remove()
            axes[0].legend(title="Eksperimen", bbox_to_anchor=(0, 1.25), loc="upper left", ncol=2)
            figure.suptitle(f"{objective} — {dataset_name}", y=1.08, fontsize=15)
            figure.tight_layout()
            filename = DATASET_FILES[dataset_name][:-4]
            figure.savefig(OUTPUT_DIR / f"loss_{objective.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()
    ''',
)

replace(
    72,
    '''
    for dataset_name in DATASET_FILES:
        dataset_data = predictions_df[predictions_df["dataset"] == dataset_name]
        for objective in OBJECTIVES:
            subset = dataset_data[dataset_data["objective"] == objective]
            grid = sns.relplot(
                data=subset, x="y_true", y="y_score", row="backbone", col="experiment",
                row_order=BACKBONES, col_order=EXPERIMENTS, kind="scatter", alpha=0.35,
                height=3, aspect=1, facet_kws={"sharex": True, "sharey": True},
            )
            low, high = subset[["y_true", "y_score"]].min().min(), subset[["y_true", "y_score"]].max().max()
            for axis in grid.axes.flat:
                axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
            grid.figure.suptitle(f"{objective}: prediksi vs target — {dataset_name}", y=1.02)
            filename = DATASET_FILES[dataset_name][:-4]
            grid.figure.savefig(OUTPUT_DIR / f"prediction_{objective.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()

            figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
            for axis, backbone in zip(axes, BACKBONES):
                sns.kdeplot(
                    data=subset[subset["backbone"] == backbone], x="residual", hue="experiment",
                    hue_order=EXPERIMENTS, palette=PALETTE, common_norm=False, ax=axis,
                )
                axis.axvline(0, linestyle="--", color="black", linewidth=1)
                axis.set_title(backbone)
            figure.suptitle(f"{objective}: distribusi residual — {dataset_name}", y=1.02)
            figure.tight_layout()
            figure.savefig(OUTPUT_DIR / f"residual_{objective.lower()}_{filename}.png", dpi=150, bbox_inches="tight")
            plt.show()
    ''',
)

replace(
    73,
    """
    # 14. File output

    Setelah **Save & Run All**, unduh folder `rankup_coral_corn_results` dari tab Output Kaggle. Isinya:

    - `metrics.csv`: backbone, objective, eksperimen, lima metrik, hardware, serta waktu training dan inference.
    - `predictions.csv`: target, expected ordinal score, dan residual setiap sampel test.
    - `history.csv`: loss per epoch.
    - PNG metrik, loss, prediksi, residual, dan t-SNE untuk setiap backbone, objective, dan dataset.

    Pada TabM, loss dihitung per anggota ensemble dan probabilitas dirata-ratakan saat inference.
    """,
)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(f"Wrote {OUTPUT} ({len(notebook['cells'])} cells)")
