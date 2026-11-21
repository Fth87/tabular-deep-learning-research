# Task 3 — ordinal RankUp, RankUp-CLS, and SSL baselines

Task 3 uses two groups of data. Ordinal RankUp is tested on the five Task 2 datasets. The RankUp-CLS adaptation and the four baseline SSL methods (SSLAE, VIME, STUNT, GFTab) are tested on five OpenML classification datasets. The four baseline SSL methods are also tested on the five ordinal datasets. The classification CSVs are stored separately in `dataset/classification/`.

`modal_app.py` sets up the image/GPU and selects the runner; `prepare_data.py` prepares the shared splits.

## Cloned paper code

| Method | Source code | Adaptation notes |
|---|---|---|
| VIME | [`experiments/vime/paper_repo`](experiments/vime/paper_repo) · [authors' repo](https://github.com/jsyoon0823/VIME) | [`experiments/vime/run.py`](experiments/vime/run.py) uses the shared split, the corruption from the authors' utilities, and the VIME objective with a TensorFlow 2 adapter. |
| STUNT | [`experiments/stunt/paper_repo`](experiments/stunt/paper_repo) · [authors' repo](https://github.com/jaehyun513/STUNT) | [`experiments/stunt/run.py`](experiments/stunt/run.py) uses synthetic clustering tasks built from train-unlabeled and prototypical few-shot evaluation from labeled-train. The adapter uses MiniBatchKMeans to handle the feature and data sizes of Task 3. |
| GFTab | [`experiments/gftab/paper_repo`](experiments/gftab/paper_repo) · [authors' repo](https://github.com/Yoontae6719/Geodesic-Flow-Kernels-for-Semi-Supervised-Learning-on-Mixed-Variable-Tabular-Dataset) | [`experiments/gftab/run.py`](experiments/gftab/run.py) keeps the numeric/categorical types from the shared split CSVs, the upstream corruption/GFK, and a tree embedding trained only on labeled train. |
| SSLAE | [`experiments/sslae/paper_repo`](experiments/sslae/paper_repo) · [authors' repo](https://github.com/sintija-s/sslae) | The Task 3 runner uses the authors' classifier API through [`experiments/sslae/run.py`](experiments/sslae/run.py) and `modal_app.py`. |
| ReCom | Not cloned | The authors' public repository URL could not be verified. Paper: [DOI](https://doi.org/10.1016/j.ins.2021.12.076). |
| SAWTab | Not cloned | The authors' public repository URL could not be verified. Paper: [DOI](https://doi.org/10.1007/978-981-97-2259-4_24). |

> The `paper_repo/` folders are not tracked by git (the code and license belong to the original authors). Download them with `./siapkan_paper_repo.sh`: the script fetches the same commits that were used in the experiments and applies the local changes to GFTab and SSLAE from `paper_repo_patch/`.

The Task 3 runner code and adjustments are kept outside the clones. The four SSL methods read the shared splits. For ordinal targets, the runner treats each level as a class, then maps the predictions back to ordered values to compute MAE and ordinal metrics. STUNT uses prototypical few-shot evaluation, so its protocol differs from the other baselines and from RankUp.

## Datasets and splits

The ten datasets are listed in [`config/datasets.json`](config/datasets.json). The five Task 2 datasets are marked `ordinal_regression`; the five OpenML datasets are in `dataset/classification/` and marked `classification`. Download metadata is in `dataset/openml_manifest.json`; download again with `python download_openml_datasets.py`. Details on counts, feature types, and checks are in [`dataset_compatibility.md`](dataset_compatibility.md).

`prepare_data.py` makes a stratified split with seed 42 for all targets:

- 70% train pool, 15% validation, and 15% test.
- From the train pool, 10% becomes labeled and 90% becomes unlabeled for SSL.
- Imputation, scaling, and one-hot encoding are fit on train only, covering both labeled and unlabeled train.

Ordinal targets stay numeric and keep their order; train-unlabeled does not store the target in the model files. The categorical columns of CMC, Bank Marketing, and Adult are set explicitly from the OpenML metadata. Missing Adult feature values are handled in the train-only preprocessor.

Each dataset produces `.npz` files for the model, CSVs for inspection, the preprocessor, and `manifest.json` in the output folder. The five ordinal datasets use MAE as the main metric; RMSE is still reported as an extra metric along with R², Spearman, and QWK. For OpenML classification, the main metric is Accuracy if the classes are balanced and Macro-F1 if not. Ten percent of the train pool labeled means about 7% of all data labeled.

## Run preparation and baselines on Modal

Install local dependencies with `pip install -r requirements.txt`, then log in with `modal setup`. The CSVs from the `dataset/` folder, including `dataset/classification/`, are included in the Modal image. From the `3-pembanding-ssl` folder, run:

```bash
modal run modal_app.py --dataset all --action prepare
modal run modal_app.py --dataset all --action sslae
modal run modal_app.py --dataset all --action vime
modal run modal_app.py --dataset all --action stunt
modal run modal_app.py --dataset all --action gftab
modal run modal_app.py --dataset all --action rankup
modal run modal_app.py --dataset classification --action rankup_cls
modal run modal_app.py --dataset all --action all --max-parallel-jobs 4
```

`--action prepare --dataset all` prepares all datasets. For the five ordinal datasets, use `--dataset ordinal`; for the five OpenML classification datasets, use `--dataset classification`. Run `prepare` first. `--action all --dataset all` schedules the four SSL baselines on ten datasets, six ordinal RankUp runs on five ordinal datasets, and six RankUp-CLS runs on five classification datasets. Jobs are sent in parallel per batch; `--max-parallel-jobs` sets the active GPU limit and defaults to 4. Modal may queue jobs if the account's GPU quota is not enough. More parallelism uses more GPUs at the same time.

The Volume `task3-data` stores the prepared splits. Metrics and predictions are written to the Volume `task3-results`. Download results with:

```bash
modal volume get task3-results lev/sslae_metrics.json ./sslae_metrics.json
modal volume get task3-results lev/vime_metrics.json ./vime_metrics.json
modal volume get task3-results adult_income/stunt_metrics.json ./stunt_metrics.json
modal volume get task3-results adult_income/gftab_metrics.json ./gftab_metrics.json
modal volume get task3-results lev/rankup_range_limited_metrics.json ./rankup_metrics.json
```

The classification runners report Accuracy, Macro-Precision, Macro-Recall, and Macro-F1. The main metric is Accuracy if the largest/smallest class ratio in labeled train is <= 1.5, otherwise Macro-F1. Ordinal RankUp selects the checkpoint by validation MAE and reports MAE, RMSE, R², Spearman, and QWK.

## Run preparation locally

Install `requirements.txt`, fill in valid CSV paths, then run:

```bash
python prepare_data.py --dataset lev --output ./prepared
```

All Modal training runners use a T4 GPU and stop with an error if no GPU is detected, so they never silently run on CPU. Split preparation and pandas/scikit-learn preprocessing still run on CPU. GFTab runs the GFK SVD and XGBoost trees on GPU; some data loading and category encoding stay on CPU before batches are sent to the GPU. All runners write metrics and predictions to `task3-results`.

To run the four SSL baselines on the same five ordinal datasets as RankUp:

```bash
modal run modal_app.py --dataset ordinal --action sslae
modal run modal_app.py --dataset ordinal --action vime
modal run modal_app.py --dataset ordinal --action stunt
modal run modal_app.py --dataset ordinal --action gftab
```

These four models remain classifiers: the ordinal target is turned into classes for training, then the class predictions are mapped back to the original target values. MAE is the main metric so the results can be compared with RankUp. This is a classification adaptation for ordinal targets, not a native ordinal implementation from the papers. STUNT also keeps its few-shot protocol.

## RankUp-style ordinal regression

RankUp has one standalone file per experiment: `experiments/rankup/tabcutmix.py`, `scarf.py`, `range_limited.py`, `gaussian.py`, `zero_out.py`, and `tabcutmix_plus.py`. Each file contains the augmentation, model, loss, Optuna, training, evaluation, and CLI for its method. Optuna searches the weak/strong strength with validation MAE as the objective; test is used only after the best parameters are chosen. Defaults follow Task 2: 200 trials (SCARF 150; limited to six pairs on the four-feature dataset), 40 epochs per trial, then final training for at most 120 epochs. Set the numbers with `--rankup-trials`, `--rankup-optuna-epochs`, and `--rankup-epochs`.

A summary of the methods, how they work, their limitations, and the paper sources is in [`related_work.md`](related_work.md).

Experiments run directly from the terminal with Modal; notebooks are not needed. To run all six augmentations on the five ordinal datasets:

```bash
modal run modal_app.py --dataset all --action rankup --augmentation all
```

For one augmentation, replace `all` with `tabcutmix`, `scarf`, `range_limited`, `gaussian`, `zero_out`, or `tabcutmix_plus`. For a smoke run, for example `modal run modal_app.py --dataset lev --action rankup --augmentation scarf --rankup-trials 2 --rankup-optuna-epochs 3 --rankup-epochs 3`. Each dataset stores `rankup_<method>_optuna_trials.csv` and `rankup_<method>_best_params.json` in addition to the metrics, predictions, and final training history.

## RankUp-style classification

RankUp-CLS runs the same six augmentations on the classification datasets. Each method has a standalone runner in `experiments/rankup_classification/`, complete with augmentation, training, Optuna, evaluation, and CLI. The runner uses cross-entropy for class targets, then uses high-confidence pseudo-labels from the weak view to train the strong view. The ordinal ARC is not used because nominal classes have no order. So the result is a weak-to-strong RankUp adaptation for classification, not a reproduction of the original RankUp. The validation metric follows Accuracy or Macro-F1 depending on class balance.

```bash
modal run modal_app.py --dataset classification --action rankup_cls --augmentation all
```

For one dataset and augmentation, for example `modal run modal_app.py --dataset diabetes_pima --action rankup_cls --augmentation scarf --rankup-trials 2 --rankup-optuna-epochs 3 --rankup-epochs 3`. Results use the prefix `rankup_cls_<method>` in the dataset folder of the Volume `task3-results`.
