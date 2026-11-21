# Thesis experiment: RankUp and tabular data augmentation

This experiment compares six augmentations for semi-supervised learning on three kinds of tasks: **ordinal, classification, and continuous regression**. Each dataset is tested with 1%, 10%, and 30% of the labels from the train pool. The initial mode runs Optuna tuning and five seeds on a single split. The extra mode uses 5-fold with one seed per fold; the two modes are stored separately.

The [dataset EDA](eda/eda_datasets.ipynb) notebook is run locally. Experiment training uses GPUs on Modal; the local dependencies do not install PyTorch.

## Folder map

| Location | Contents |
|---|---|
| `RINGKASAN_SEMUA_EKSPERIMEN.md` | **Full summary of all experiments** (settings, results, findings, limitations; in Indonesian) |
| `dataset/`, `config/`, `prepare_data.py` | Raw data, dataset list, and split preparation script |
| `eda/` | EDA notebook |
| `eksperimen/visualisasi_eksperimen.ipynb` | **Visual summary of all results** (start here) |
| `eksperimen/hasil/gambar/` | Result figures, ready for slides |
| `eksperimen/hasil/` | Run results: `ringkasan-5seed-40trial.csv` (Optuna + 5 seeds), `usulan-semua/` (13 datasets without Optuna), `pembanding/` (other SSL methods), `log/` |
| `eksperimen/catatan/` | Research notes, source audit, and exploration reports |
| `eksperimen/experiments/`, `eksperimen/tests/` | Training code and tests |
| `eksperimen/modal_*.py`, `eksperimen/ringkas_*.py` | Scripts to run experiments on Modal and summarize their results |
| `../5-augmentasi-latent/` | Follow-up: augmentation in the TabM latent space (moved to its own folder) |
| `usulan/` | Archive of exploration of another augmentation (pilot, not used in the main results) |

## Datasets and metrics

| Task | Dataset | Rows | Features | Target | Main metric | Direction |
|---|---|---:|---:|---|---|---|
| Ordinal | NHANES | 5,223 | 30 | Five ordinal levels | Quadratic Weighted Kappa (QWK) | Maximize |
| Classification | Pen Digits | 10,992 | 16 | Digits 0–9 | Macro-F1 | Maximize |
| Continuous regression | Superconductivity | 21,263 | 81 | Critical temperature `critical_temp` | Mean Squared Error (MSE) | Minimize |

- **NHANES:** the Task 2 ordinal dataset from [TOC-UCO](https://www.uco.es/grupos/ayrna/materials/tocuco/). The features have anonymous names, so the meaning of each feature cannot yet be interpreted substantively. In the 1% split, the rarest level has only one labeled example.
- **Pen Digits:** [UCI Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), Alpaydin and Alimoglu (1996), [DOI 10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K). The ten classes are weighted equally in macro-F1. The dataset is licensed CC BY 4.0.
- **Superconductivity:** [UCI Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data), Hamidieh (2018), [DOI 10.24432/C53P47](https://doi.org/10.24432/C53P47). The target stays continuous and the dataset is licensed CC BY 4.0.

The main metric decides the Optuna trial and the best checkpoint based on **validation**. Ordinal training still uses the regression MSE loss; predictions are rounded to the nearest ordinal level when computing QWK. Classification training uses cross-entropy, while regression uses the MSE loss. Evaluation MSE is computed on the original target scale.

## Data split

Validation and test stay fixed across all label fractions and the five training seeds. The label fraction is computed from the train pool, not from the whole dataset. The rest of the train pool becomes unlabeled data.

| Dataset | Train pool | 1% labeled | 10% labeled | 30% labeled | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| NHANES | 3,655 | 36 | 365 | 1,096 | 784 | 784 |
| Pen Digits | 5,995 | 59 | 599 | 1,798 | 1,499 | 3,498 |
| Superconductivity | 15,006 | 150 | 1,500 | 4,501 | 3,176 | 3,081 |

- NHANES uses a stratified split of about 70% / 15% / 15% for train pool / validation / test.
- Pen Digits keeps the official UCI test set, which comes from different writers. Validation is taken stratified from 20% of the official train set.
- Superconductivity separates groups of rows with identical feature vectors so they do not appear in the train pool, validation, and test at the same time. Identical rows are kept; similar materials whose features are not identical can still be spread across partitions. Within the train pool, rows with identical features can still be split between labeled and unlabeled.

Feature preprocessing is fit on the train pool, including the unlabeled part, then applied to validation and test. The prepared unlabeled file does not store the target. The EDA notebook analyzes the whole dataset for description; model selection still uses train and validation.

## Experiment design

There are **3 datasets × 6 augmentations × 3 label fractions = 54 combinations**. In the initial mode, each combination runs one Optuna study and five final trainings: **54 tuning studies and 270 final trainings** in total, not counting Optuna trials. The 5-fold mode is described in a separate section below.

| Augmentation | Method ID | How it is used in the experiment |
|---|---|---|
| TabCutMix | `tabcutmix` | Swaps some features with a donor from the mini-batch that has the same pseudo-label group. |
| SCARF | `scarf` | Replaces selected features with a donor value independent for each feature. |
| Range-Limited Augmentation | `range_limited` | Draws feature values from a local quantile range. |
| Gaussian | `gaussian` | Adds Gaussian noise to numeric features. |
| Zero Out | `zero_out` | Replaces selected features with zero after standardization. |
| TabCutMix Plus | `tabcutmix_plus` | Swaps groups of features based on correlation with a matching donor. |

On **NHANES and Superconductivity**, the implementation adapts RankUp with a tabular model and an auxiliary ranking classifier (ARC). On **Pen Digits**, `RankUp-CLS` uses class pseudo-labels with weak-to-strong augmentation without ARC; the approach is closer to FixMatch and is not the original regression RankUp. This implementation also does not use the optional RDA from the RankUp paper.

The augmentation operators are adapted for unlabeled data, not the whole pipeline of the source paper. TabCutMix uses pseudo-labels to find donors; if no donor is available, the sample is not swapped. On continuous regression, TabCutMix and TabCutMix Plus use quantile groups of the prediction to choose donors, without changing the continuous target and without a donor confidence estimate. RLA samples uniformly within the range; on discrete features, the result can become non-discrete. Zero after standardization is close to the train mean, not always a missing value.

The current series compares those six variants; there is no supervised-only or no-augmentation baseline yet. So the results are not enough to conclude how large the improvement over such a baseline is.

## Optuna and five seeds

For each combination of dataset, augmentation, and label fraction:

1. Optuna searches the **weak** and **strong** augmentation strengths based on the validation metric. The default is at most **40 trials**, each at most **40 epochs**; pruning can stop a trial early. The default tuning seed is 42.
2. Pick the best parameters from that study. The learning rate, loss weights, and other training parameters use fixed values and are not tuned.
3. Retrain with seeds **42, 123, 456, 789, and 101112**, at most **120 epochs** per seed. The best parameters and the same data split are used for all five; each seed chooses its checkpoint based on validation.
4. Compute the mean and sample standard deviation of the five seed scores for labeled train, validation, and test separately. This variation describes training variation on one split, not variation between data splits.

Other default settings: AdamW, learning rate `0.002`, weight decay `0.0003`, batch size `32`, at most `100` steps per epoch, patience `15`, ranking weight `0.2`, unlabeled weight `0.2`, confidence cutoff `0.95`, and donor confidence `0.8`. Some parameters apply only to methods or tasks that need them. The search ranges and the actual configuration are stored with the tuning and training results.

## Extra mode: 5-fold, single seed

The `kfold` mode keeps the same test set for all folds. The data outside test is split into five folds: each fold takes a turn as validation, and the other four folds become the train pool. The 1%, 10%, and 30% fractions are computed from each fold's train pool. The preprocessor is refit on each fold's train pool, including unlabeled data; test and validation are not used for fitting.

NHANES and Pen Digits use stratified 5-fold; Pen Digits still uses the official test set. Superconductivity uses GroupKFold on identical feature vectors, with a separate test set split by feature groups. The test split and the stratified fold shuffling use seed 42; GroupKFold is deterministic without shuffling. The default tuning and training seed is 42. In each fold, Optuna runs up to 40 trials based on the validation metric, then the model is trained once with the best parameters and seed 42. In total, **54 combinations × 5 folds = 270 Optuna studies and 270 final trainings**, not counting Optuna trials.

To prepare all folds and then send all jobs to Modal:

```bash
uv run modal run eksperimen/modal_app.py --action prepare --protocol kfold --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --protocol kfold --dataset all --method all --label-percent all --run-id ta-kfold-001
```

Preparation writes the splits to `experiment-ta-data/kfold/label_XX/fold_XX/<dataset>/`; results go to `experiment-ta-results/<run-id>/kfold/label_XX/fold_XX/`. These two folders are separate from the single-split, five-seed mode. Wait for the message `Queued all 270 jobs in Modal` before closing the terminal. By default at most four GPUs run at the same time. Use `--fold 1` to `--fold 5` for one fold or `--fold all` for all of them. `--seed` sets the tuning and training seed in all folds; the data split uses the seed in `config/datasets.json`.

Each fold's results include the labeled train, validation, and test metrics. Download and summarize:

```bash
mkdir -p eksperimen/hasil
uv run modal volume get experiment-ta-results ta-kfold-001 eksperimen/hasil
uv run python eksperimen/ringkas_kfold.py eksperimen/hasil/ta-kfold-001 --output eksperimen/hasil/ta-kfold-001-ringkasan.csv
```

The summary presents each fold's score and the mean and sample standard deviation for train, validation, and test when all five folds are complete. The test scores come from five models on the **same test set**, so they are not five independent tests. Use validation for parameter selection and comparison; do not choose a fold based on test. Fold variation does not measure seed variation.

Resume works per combination and fold: reuse the same run ID and settings after the previous job has stopped. Saved trials and complete seeds are reused; trials or seeds interrupted midway are repeated. Do not run the same job at the same time. This mode does not yet resume training from the last epoch.

## Local EDA

The notebook [eda/eda_datasets.ipynb](eda/eda_datasets.ipynb) contains statistics, target distributions, missing values, duplicates, and visualizations for the three datasets. Output is stored in `eda/hasil/`; details and notes on the earlier datasets are in the [EDA README](eda/README.md).

From the `4-augmentasi-input` directory:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace eda/eda_datasets.ipynb
```

In an IDE, use the `4-augmentasi-input/.venv/bin/python` interpreter as the kernel. `uv sync` installs the light EDA and Modal dependencies, while PyTorch is installed in the Modal image for training.

## Running experiments on Modal

From the `4-augmentasi-input` directory, prepare the data then run all combinations:

```bash
uv sync
uv run modal run eksperimen/modal_app.py --action prepare --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --dataset all --method all --label-percent all --run-id ta-001
```

Use `--dataset nhanes`, `pendigits`, or `superconductivity`; `--label-percent 1`, `10`, or `30`; and `--method` with an ID from the augmentation table to limit the experiment. The `all` option runs all values. `--trials`, `--optuna-epochs`, and `--epochs` can change the default limits. `--seed` changes the tuning seed, not the five final training seeds.

Example smoke test of one combination, which still runs the five final seeds:

```bash
uv run modal run eksperimen/modal_app.py --action prepare --dataset nhanes --label-percent 1
uv run modal run --detach eksperimen/modal_app.py --action train --dataset nhanes --method range_limited --label-percent 1 --run-id smoke-001 --trials 2 --optuna-epochs 3 --epochs 3
```

Training uses T4 GPUs with at most four jobs at once and a 12-hour timeout per job. `--detach` makes Modal continue all jobs already in the queue after the local process ends; wait for the message `Queued all ... jobs in Modal` before closing the terminal. If a job fails or exceeds the timeout, check the logs and rerun with the same run ID and settings; finished Optuna trials and seeds are reused. Use a new `--run-id` to avoid overwriting results. Running prepare again for the same dataset and fraction overwrites the split in the data Volume.

## Resuming an interrupted experiment

After the previous job has stopped, run the same training command again with the same `--run-id` and options. You can also limit the dataset, method, and fraction to repeat specific combinations. Do not run the same combination at the same time with the same run ID: the Modal Volume provides no locking between containers. Results from versions before the resume feature cannot be resumed automatically; use a new run ID. Avoid running prepare again before resuming because the split file checksums can change.

Each finished Optuna trial saves a snapshot of the study and sampler, then commits the Volume. Resume continues the remaining trials; a trial that was running when the interruption happened is recomputed. Each finished seed has a marker and checksums of the result files and best parameters; complete and valid seeds are skipped. A seed interrupted midway through training restarts from the beginning, not from the last epoch. When the code, library versions, data split, number of trials, epochs, or tuning seed change, use a new run ID so results do not get mixed. Do not modify or run pickle snapshot files from untrusted sources.

The dashboard link appears when a job is submitted. Check the status with `uv run modal app list` and the logs with `uv run modal app logs <app-id>`. If a job is still running, wait or stop it before resubmitting the same combination. Keep the run ID to find the results in the Volume.

## Results and reproduction

The prepared data is stored in the Modal Volume `experiment-ta-data/label_XX/<dataset>/`, including the split, manifest, and feature preprocessor. Results are stored in the Volume `experiment-ta-results/<run-id>/`:

| Location | Contents |
|---|---|
| `label_XX/tuning/<dataset>/` | Optuna trial history, study snapshot for resume, resume configuration, best parameters, search ranges, and tuning seed. |
| `label_XX/seed_<seed>/<dataset>/` | Initial configuration and library versions, labeled train / validation / test metrics, per-epoch loss and validation metric history, test predictions, and the best model checkpoint. |
| `label_XX/summary/<dataset>/` | Each seed's values plus the mean and standard deviation for train, validation, and test. |

The feature preprocessor is in the data Volume, not inside the checkpoint; keep both for reproduction. The per-epoch history records loss and the validation metric; the labeled train score is computed after the best checkpoint is chosen.

Download and combine the summaries:

```bash
mkdir -p eksperimen/hasil
uv run modal volume get experiment-ta-results ta-001 eksperimen/hasil
uv run python eksperimen/ringkas_hasil.py eksperimen/hasil/ta-001 --output eksperimen/hasil/ta-001-ringkasan.csv
```

The summary marks combinations that are not yet available as `missing` or `partial`; the mean and standard deviation are shown only when all five seeds are available. Instructions for light checks and file details are in the [experiment README](eksperimen/README.md).

## Paper and code references

| Reference | Use in the experiment |
|---|---|
| Huang, Fu, and Tsao, **RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier**, NeurIPS 2024. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c26a8494fe31695db965ae8b7244b7c1-Paper-Conference.pdf) · [Official code](https://github.com/pm25/Semi-Supervised-Regression) | The basis of RankUp and ARC for regression and ordinal. |
| Sohn et al., **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**, NeurIPS 2020. [Paper](https://arxiv.org/abs/2001.07685) · [Official code](https://github.com/google-research/fixmatch) | The basis of pseudo-labels and weak-to-strong consistency in the classification adaptation. |
| Gorishniy et al., **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**. [Paper](https://arxiv.org/abs/2410.24210) · [Official code](https://github.com/yandex-research/tabm) | The tabular architecture used in the model. |
| Fang et al., **Understanding and Mitigating Memorization in Diffusion Models for Tabular Data**, ICML 2025. [Paper](https://proceedings.mlr.press/v267/fang25f.html) · [Official code](https://github.com/fangzy96/TabCutMix) | Source of TabCutMix and TabCutMix Plus; its use here is adapted for pseudo-labels. |
| Bahri et al., **SCARF: Self-Supervised Contrastive Learning Using Random Feature Corruption**, ICLR 2022. [Paper](https://openreview.net/pdf?id=CuV_qYkmKb3) | Source of the SCARF feature corruption. The authors' official repo is not verified; the project uses its operator, not the whole SCARF pretraining. |
| Lee et al., **Range-limited Augmentation for Few-shot Learning in Tabular Data with Comprehensive Benchmark**, KDD 2025. [Paper](https://openreview.net/pdf?id=kTH3bEH6hW) · [Official FeSTA code](https://github.com/kyungeun-lee/festa) | Source of the RLA idea; this implementation uses sampling within the quantile range. |
| GsRCL. [Paper](https://academic.oup.com/bfg/article/23/4/441/7577350) · [Authors' code](https://github.com/ibrahimsaggaf/GsRCL) | Reference for using Gaussian noise; this experiment only uses the Gaussian operator. |
| Ucar et al., **SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning**, NeurIPS 2021. [Paper](https://papers.nips.cc/paper/2021/file/9c8661befae6dbcd08304dbf4dcaf0db-Paper.pdf) · [Official code](https://github.com/AstraZeneca/SubTab) | Reference for zero-out; this experiment does not run the whole SubTab. |
| Akiba et al., **Optuna: A Next-generation Hyperparameter Optimization Framework**, KDD 2019. [Paper](https://arxiv.org/abs/1907.10902) · [Official code](https://github.com/optuna/optuna) | Optimization of the augmentation strength. |

The [source audit notes](eksperimen/catatan/temuan_sumber.md) record an initial comparison with the original implementations. Some findings in those notes have been fixed; check the status note at the top of the file when reading them.
