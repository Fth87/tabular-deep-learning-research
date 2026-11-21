# RankUp experiments

This guide describes the initial single-split, five-seed mode. The additional **5-fold, single-seed** mode, with its commands and result summary, is in the [main README](../README.md#extra-mode-5-fold-single-seed). The two use separate data and result directories.

Six augmentations are compared on three datasets and three labeled-data fractions: **1%, 10%, and 30% of the train pool**. There are **54 combinations** in total. Each combination runs Optuna once, then five final trainings with the best parameters: 54 tuning processes and 270 final trainings in total.

1. TabCutMix (`tabcutmix`)
2. SCARF (`scarf`)
3. Range Limited Augmentation / RLA (`range_limited`)
4. Gaussian (`gaussian`)
5. Zero Out (`zero_out`)
6. TabCutMix Plus (`tabcutmix_plus`)

Superconductivity uses RankUp for continuous regression with MSE. NHANES uses RankUp for ordinal regression with QWK as the main metric. Pen Digits uses the RankUp-CLS adaptation for classification with macro-F1; this adaptation uses class pseudo-labels and does not use the ordinal ARC.

## Data and metrics

NHANES uses a stratified split of about 70% train pool, 15% validation, 15% test. Pen Digits keeps the official UCI test set (different authors); validation is taken stratified at 20% of the official train set. Superconductivity uses a split by groups of identical feature vectors to prevent the same features from appearing across train, validation, and test; the row proportions are close to 70/15/15. Validation and test stay the same across fractions. The 1%, 10%, and 30% fractions are drawn from the train pool; the rest becomes train-unlabeled. Preprocessing is fit on the train pool and train-unlabeled is stored without targets. The five seeds change the training, not the data split.

| Dataset | 1% labeled | 10% labeled | 30% labeled | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Superconductivity | 150 | 1,500 | 4,501 | 3,176 | 3,081 |
| NHANES | 36 | 365 | 1,096 | 784 | 784 |
| Pen Digits | 59 | 599 | 1,798 | 1,499 | 3,498 |

In NHANES 1%, the rarest level has only one labeled example. This can make the results unstable.

Superconductivity uses MSE for training, trial/checkpoint selection, and test. The continuous `critical_temp` target is kept intact. Only for TabCutMix and TabCutMix Plus, the predictions are grouped by quantiles of the train labels to choose donors; this is a temporary adaptation and not a change of target or a required approach for all regression. There is no donor confidence estimate in this adaptation.

For ordinal, the regression loss during training is still MSE on the standardized target. Trial and checkpoint selection use QWK on validation; the ordinal metric reported on test is QWK. Continuous predictions are mapped to the nearest ordinal level before QWK is computed. Pen Digits uses cross-entropy for training, macro-F1 for trial and checkpoint selection, then reports macro-F1 and other classification metrics on test.

SCARF takes an independent donor per feature. The ranking consistency loss is computed as the mean of the masked loss over all pairs, following the RankUp normalization. TabCutMix and RLA remain adaptations for unlabeled data. TabCutMix Plus uses hierarchical average-linkage as in the source code; probabilistic group selection allows a sample to stay unchanged; details are in the [source findings](catatan/temuan_sumber.md). There is no supervised-only baseline in this series.

## Testing

A light split check can be run locally with `uv run python -m unittest discover -s eksperimen/tests -p test_splits.py -v`. Tests that need PyTorch run on Modal with `uv run modal run eksperimen/modal_app.py --action check`; this command uses CPU and small synthetic data, without running the full experiment.

## Run on Modal

From the `4-augmentasi-input` folder:

```bash
uv sync
uv run modal run eksperimen/modal_app.py --action prepare --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --dataset all --method all --label-percent all --run-id ta-001
```

For a smoke test, after preparing the matching split:

```bash
uv run modal run eksperimen/modal_app.py --action prepare --dataset nhanes --label-percent 1
uv run modal run --detach eksperimen/modal_app.py --action train --dataset nhanes --method range_limited --label-percent 1 --run-id smoke-001 --trials 2 --optuna-epochs 3 --epochs 3
```

`--dataset` accepts `all`, `nhanes`, `pendigits`, or `superconductivity`; `--label-percent` accepts `all`, `1`, `10`, or `30`; `--method` accepts `all` or the augmentation names above. Optuna searches the weak and strong augmentation strengths using each dataset's validation metric; the learning rate and loss weights use fixed values and are not tuned. Tuning uses seed 42 by default, 40 trials, and 40 epochs per trial. After the best parameters are found, the model is retrained with seeds **42, 123, 456, 789, and 101112**, at most 120 epochs per seed; each seed uses the best parameters and the same data split. The summary stores the mean and sample standard deviation of the five labeled-train, validation, and test scores. `--seed` changes the tuning seed, not the list of five seeds. At most four jobs use T4 GPUs at once, and each job's timeout is 12 hours. All jobs are sent to the Modal queue; `--detach` keeps the jobs running after the local process ends. Wait for the message `Queued all ... jobs in Modal` before closing the terminal. Jobs that fail or exceed the timeout can be resubmitted with the same run ID and settings; saved trials and seeds are resumed. Do not resubmit the same combination while the previous job is still running.

Splits are stored in the Volume `experiment-ta-data/label_XX/<dataset>`; running prepare for the same fraction and dataset overwrites that split. Tuning results are stored in `experiment-ta-results/<run-id>/label_XX/tuning/<dataset>`, per-seed results in `label_XX/seed_<seed>/<dataset>`, and the five-seed summary in `label_XX/summary/<dataset>`. Use a new run ID for every repetition so files are not overwritten.

## Resuming after an interruption

Rerun the training command with the same run ID and settings after the previous job has stopped. The Optuna snapshot is saved and committed per trial; a trial that was running when the interruption happened is recomputed. Seeds that are complete and pass the checksum are skipped, while a seed interrupted in the middle of training restarts from the beginning. The code, data splits, library versions, and settings must be the same; use a new run ID if they change. See the [main guide](../README.md#resuming-an-interrupted-experiment).

## Contents of each experiment's results

In the folder `tuning/<dataset>/`, the `*_optuna_trials.csv` file records each trial and its status. `*_best_params.json` records the best parameters, the search ranges used in the best trial, the number of trials by status, the tuning seed, and the label fraction. The default is **40 trials per combination**; pruning can stop some trials before 40 epochs.

In each `seed_<seed>/<dataset>/`, the `*_run_config.json` file stores initial arguments such as learning rate, batch size, epochs, loss weights, cutoff, seed, data manifest, and library versions. The `*_metrics.json` file stores the train, validation, and test metrics; the train score is computed on **labeled** train data after the best checkpoint is chosen. `*_history.csv` stores the loss history and validation metric per epoch; `*_predictions.npz` stores the test predictions. `*_model.pt` stores the best checkpoint weights and model metadata; for regression and ordinal it also stores the target scaler parameters. The feature preprocessor and splits are stored in the data Volume, not in the model file. The Superconductivity split groups identical features; similar but non-identical materials are still spread across partitions.

## Download and summarize results

```bash
mkdir -p eksperimen/hasil
uv run modal volume get experiment-ta-results ta-001 eksperimen/hasil
uv run python eksperimen/ringkas_hasil.py eksperimen/hasil/ta-001 --output eksperimen/hasil/ta-001-ringkasan.csv
```

The summary marks combinations that are not yet available as `missing` or `partial`; the mean and standard deviation are shown only when all five seeds are available. The old Task 3 ordinal results were chosen by MAE, so they are not results from this QWK protocol.
