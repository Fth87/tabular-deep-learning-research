# Stage 2 — RankUp on ordinal data (Task 1 and Task 2, July–September 2026)

Tests RankUp on **five ordinal datasets** (Lev, ESL, NHANES, Student Performance, Melbourne Airbnb) with several tabular models, ordinal objectives, and weak/strong augmentations. Almost all experiments are self-contained Kaggle notebooks; the results are stored in the notebook outputs.

## Folder contents

| Folder | Contents |
|---|---|
| `dataset/` | The five ordinal datasets (CSV) |
| `kode/` | Source notebooks: EDA, ordinal tabular model benchmark, RankUp on all datasets (MLP, TabM, RealMLP), and the CORAL/CORN version |
| `hasil/` | Executed notebooks (with outputs), including the 5-seed benchmark and the CORAL/CORN version |
| `hasil-tabm/` | RankUp + TabM with 1%, 10%, and 30% labeled fractions, plus the visualization notebook |
| `task1_weak_strong_optuna/` | **Task 1**: Optuna searches the weak/strong strength for TabCutMix, SCARF, Range-Limited, and Gaussian (objective: validation MAE) |
| `hasil-optuna/` | Task 1 run results (one folder per augmentation) |
| `gambar/` | Visual summary: profile of the five ordinal datasets, best weak/strong parameters from Optuna, and Optuna convergence plots |
| `catatan-rapat-2026-08-26.md` | Supervisor's guidance: how to choose weak/strong strength, paper survey, number of seeds |
| `docs/superpowers/` | Specification and plan for the CORAL/CORN notebook (19 August) |
| `scripts/`, `tests/` | CORAL/CORN notebook generator and its test |

The folder names are in Indonesian: `kode` = code, `hasil` = results, `gambar` = figures, `catatan-rapat` = meeting notes.

## Main experiments

- **Ordinal tabular model benchmark:** TabM, RealMLP-TD-S, XGBoost, LightGBM, and CatBoost; a 1-seed baseline, Optuna tuning, then retraining with 5 seeds.
- **RankUp on all datasets:** MLP, TabM, and RealMLP compared in four scenarios: Supervised, Fully-Supervised (upper bound), RankUp + range-limited augmentation, and RankUp + TabCutMix.
- **CORAL and CORN:** ordinal objectives added alongside regression.
- **Label-fraction sweep (TabM):** 1%, 10%, and 30% labels with four augmentations (Range-Limited, SCARF, TabCutMix, Gaussian).

## Running

The notebooks are designed for Kaggle (GPU). Locally, `uv sync` installs the basic dependencies (`pyproject.toml`). To see the results, open the notebooks in `hasil/` and `hasil-tabm/`; there is no need to rerun them.

Next: [`../3-pembanding-ssl/`](../3-pembanding-ssl/).
