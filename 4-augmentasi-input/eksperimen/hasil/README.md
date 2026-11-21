# Five-seed experiment results

Results of the run `ta-5seed-40trial-20261105`, taken from Modal (workspace `lagi`). The run covers 3 datasets, 6 augmentations, 3 label percentages, at most 40 Optuna trials, and 5 seeds per combination.

- `ringkasan-5seed-40trial.csv`: mean and standard deviation of the labeled-train, validation, and test metrics for the 54 combinations.
- `ta-5seed-40trial-20261105/`: best parameters, trial history, configuration, per-seed metrics, training history, test predictions, and JSON summaries.
- `data-5seed/`: data splits and preprocessors for each dataset and label percentage.

The model checkpoints `*_model.pt` and the Optuna study snapshots `*_study.pkl` were not copied into this folder. The original files remain in the Modal Volume `experiment-ta-results`. The `*_complete.json` marker files record the checksum of the full result on Modal, including the model checkpoints that were not copied locally.
