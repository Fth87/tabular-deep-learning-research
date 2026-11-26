# Latent augmentation in the TabPFN-3.5 embedding space (Kaggle, 2×T4)

Hypothesis: latent augmentation did not help in stage 5 because TabM's latent space, trained from 36–4,500 labels, has little structure, and because the pseudo-labels come from the model that is still learning. Here the latent space is a frozen TabPFN-3.5 embedding (the default model of `tabpfn` 10.x) and the targets for unlabeled rows come from TabPFN as a teacher. The latent operators are the same code and strengths as stage 5 (`latent.perturb`, `GRID`).

| Name | What runs |
|---|---|
| `tabpfn` | TabPFN-3.5 fit on labeled rows, prediction only (the bar to beat) |
| `head_sup` | MLP head on the embedding, labeled loss only |
| `head_distill` | + distillation to the teacher on unlabeled rows, no augmentation |
| `head_<op>` | + latent operator on the head's hidden layer (strong view): `gaussian`, `zero_out`, `scarf`, `tabcutmix`, `dropout`, `fnoise`, `angular` |

Embeddings of labeled rows are taken out-of-fold (5 folds), so no row is embedded by a context that contains it. Checkpoints are chosen on validation; test is only reported. Metrics: NHANES QWK, Pen Digits macro-F1, Superconductivity MSE.

## Running on Kaggle

All code lives in [`kaggle_tabpfn_latent.ipynb`](kaggle_tabpfn_latent.ipynb); the Kaggle Dataset holds only the data splits, so it never needs re-uploading when the code changes. The notebook's `common.py` cell is an exact copy of the project functions it needs (`load` and `scores` from `experiments/usulan.py`, `same_rank_donors` from `experiments/rankup/tabcutmix.py`, the latent operators from `latent.py`), and the operator strengths are the `latent_strong` values from `grid.py`. If those files change, update the copy in the notebook too.

1. Locally: `./7-tabpfn-latent/siapkan_kaggle.sh` builds `kaggle_data.zip` (`label_01/`, `label_10/`, `label_30/` from `5-augmentasi-latent/data/`). Upload it once as a Kaggle Dataset.
2. Import the notebook into Kaggle: add the dataset, Accelerator **GPU T4 x2**, Internet **on**. The TabPFN-3.5 checkpoint (`tabpfn-v3.5-20260909.safetensors`, 876 MB, license `tabpfn-3-5-license-v1.0`, non-commercial) is downloaded from [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5), which is not gated. If the package still asks for a token, add a Kaggle secret `TABPFN_TOKEN` with a Prior Labs API key from [platform.priorlabs.ai](https://platform.priorlabs.ai).
3. Run the cells in order. The `%%writefile` cells write `common.py` and `tabpfn_latent.py` to `/kaggle/working/` so that the run can use one process per GPU. Run the smoke test first (2 groups, 200 steps).

The full run is 27 groups (3 datasets × 3 label % × 3 seeds) × 10 variants. Results go to `/kaggle/working/hasil/results_gpu{0,1}.jsonl`, and TabPFN embeddings are cached in `/kaggle/working/cache/`. Finished runs are skipped within a session; `/kaggle/working` is emptied in a new session, so download the zip from the last cell.

## Results of the TabPFN 3.5 and 2.5 runs

Both versions have 270 matched final runs. The [full comparison](RINGKASAN_2_5_DAN_3_5.md) covers all variants, validation/test wins by seed, the validation-calibrated NHANES QWK, and the TabM controls from stages 5–6.

| TabPFN teacher, mean test score | 2.5 · 1/10/30% | 3.5 · 1/10/30% |
|---|---|---|
| NHANES, calibrated QWK ↑ | 0.215 / 0.329 / 0.376 | **0.274 / 0.338 / 0.424** |
| Pen Digits, macro-F1 ↑ | 0.838 / 0.948 / 0.973 | **0.873 / 0.970 / 0.984** |
| Superconductivity, MSE ↓ | **306.5** / 165.8 / 122.4 | 318.1 / **163.4 / 116.4** |

Pen Digits 3.5 wins all nine seed–label pairs against 2.5. SSR-RCUS from stage 6 still scores higher at 1% and 10% labels (0.885 and 0.973). On Superconductivity, angular augmentation raises the 3.5 head's MSE from 315 to 780 at 1% labels. On NHANES, angular raises raw QWK while Spearman barely moves and MSE rises. Calibration of the 3.5 teacher raises test QWK to 0.274 / 0.338 / 0.424, so stage 8 uses validation-calibrated QWK as its ordinal selection metric.

### Earlier 2.5 comparison

The run in [`versi 2.5/kaggle_data/`](versi%202.5/kaggle_data/) (270 rows, 27 groups × 10 variants, no duplicates) was made with an earlier version of the notebook, so its rows carry no `tabpfn_version` field; the version comes from the folder name, and the embedding size (192) fits the v2 family. Test scores, mean of 3 seeds:

| | NHANES 1/10/30% (QWK) | Pen Digits 1/10/30% (macro-F1) | Superconductivity 1/10/30% (MSE) |
|---|---|---|---|
| `tabpfn` | 0.089 / 0.216 / 0.320 | 0.838 / 0.948 / 0.973 | 306 / 166 / 122 |
| `head_distill` | 0.143 / 0.255 / 0.328 | 0.839 / 0.947 / 0.969 | 304 / 167 / 124 |
| `head_angular` | 0.237 / 0.298 / 0.371 | 0.842 / 0.947 / 0.970 | 831 / 228 / 150 |
| `head_scarf` | 0.237 / 0.306 / 0.373 | 0.842 / 0.944 / 0.970 | 633 / 213 / 151 |
| best of stage 5 (TabM) | 0.280 / 0.275 / 0.319 | 0.893 / 0.953 / 0.976 | 365 / 202 / 146 |

- **Superconductivity:** TabPFN alone lowers MSE by 16–20% against the best stage-5 run. Mild operators (`gaussian`, `dropout`, `fnoise`) match `head_distill`; strong ones (`zero_out`, `scarf`, `angular`) multiply MSE by 2–3.
- **Pen Digits:** no gain; stage 5 is better at 1% labels.
- **NHANES:** the QWK gains of `scarf`, `angular`, and `zero_out` are a prediction-spread artifact. Spearman is unchanged (0.35 at 10% for both `head_distill` and `head_angular`) while MSE rises (0.93 → 1.31). Plain `tabpfn` with class cutoffs tuned on validation (validation-prediction quantiles matched to validation class proportions), computed from the cached teacher predictions, reaches 0.215 / 0.329 / 0.376.

Follow-up design: [`../8-tabpfn-ssl/`](../8-tabpfn-ssl/).

