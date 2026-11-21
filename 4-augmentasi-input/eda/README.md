# Experiment dataset EDA

The [EDA notebook](eda_datasets.ipynb) loads NHANES, Pen Digits, and Superconductivity. The notebook has already been executed locally; to rerun it from the `4-augmentasi-input` folder:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace eda/eda_datasets.ipynb
```

In an IDE, choose the `4-augmentasi-input/.venv/bin/python` interpreter as the kernel. The EDA over the whole dataset is descriptive only; feature and model selection must use train/validation.

| | NHANES | Pen Digits | Superconductivity |
|---|---:|---:|---:|
| Rows | 5,223 | 10,992 | 21,263 |
| Numeric features | 30 | 16 | 81 |
| Missing values | 0 | 0 | 0 |
| Fully duplicated rows | 0 | 0 | 66 |
| Constant features | 0 | 0 | 0 |

**NHANES:** the target has five ordinal levels. The rarest level has only one labeled example in the current 1% split.

**Pen Digits:** the target has ten digit classes. The official UCI test set is kept; validation is drawn from the official train set.

**Superconductivity:** `critical_temp` is a continuous target. There are 6,093 rows that repeat, judging by all features, beyond their first occurrence. The split groups identical features so they do not mix across the train pool, validation, and test. Repeated rows are kept; similar but non-identical materials can still be spread across different partitions.

Plots, descriptive statistics, and JSON summaries are stored in [`hasil/`](hasil/). The earlier EDA results for Satimage and Grid Stability are kept in [`hasil/arsip_dataset_sebelumnya/`](hasil/arsip_dataset_sebelumnya/).
