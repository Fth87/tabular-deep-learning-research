# Task 3 experiment results

This folder contains all 100 completed dataset/model runs across the 10 datasets.

## Contents

- One directory per dataset, with model metrics, held-out test predictions, and saved model artifacts where provided.
- RankUp has Optuna trial CSVs, best-parameter JSON files, and training-history CSVs for each of six augmentations.
- `experiment_summary.csv` contains one row per run and links each result to its detailed metrics JSON.

## Experimental setup

- Split: 70% train, 15% validation, 15% test; seed 42.
- Labeled subset: 10% of the training split; remaining training rows are unlabeled.
- Ordinal datasets use MAE as the primary metric; RMSE is also retained as a secondary metric.
- Classification datasets use Macro-F1 as the primary metric and report accuracy, macro precision, and macro recall.
- RankUp used up to 40 Optuna trials per dataset/augmentation, with pruning enabled; the trial CSV records completed and pruned trials.
- RankUp final training used 120 epochs. Jobs used Modal T4 GPUs.
- Baseline defaults: GFTab 10 epochs, SSLAE 100 epochs, STUNT 1,000 steps, and VIME 1,000 iterations with 10 pretraining epochs.

## Model coverage

- RankUp ordinal: six augmentations on each of five ordinal datasets.
- RankUp-CLS: six augmentations on each of five classification datasets.
- GFTab, SSLAE, STUNT, and VIME: each on all 10 datasets.

Hyperparameter selection uses validation metrics; final reported metrics use the held-out test split.
