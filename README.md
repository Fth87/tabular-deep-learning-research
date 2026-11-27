# Semi-supervised learning and augmentation for tabular data

A thesis research project comparing **semi-supervised learning methods** and measuring the **additional benefit of augmentation** on tabular data. It starts with RankUp (Huang et al., NeurIPS 2024) and follows comparisons with other SSL methods, TabM and TabPFN. The main experiments cover ordinal, classification, and regression tasks with 1%, 10%, and 30% labeled data.

**Current experiment:** [Full methodology and results](8-tabpfn-ssl/dokumen/EKSPERIMEN_SSL_TABULAR.md) · [SSR-RCUS vs TabPFN 3.5 charts](8-tabpfn-ssl/output/perbandingan_ssr_rcus.md)

![Architecture: weak/strong augmentation in the TabM latent space](5-augmentasi-latent/gambar/arsitektur/arsitektur_medium.png)

## Background

Labels are expensive and unlabeled data is cheap. **Semi-supervised learning (SSL)** uses both. Methods such as FixMatch pick pseudo-labels with a confidence threshold, but this does not work for regression because a regression prediction carries no confidence. **RankUp** solves this with an *auxiliary ranking classifier* (ARC) that guesses the order of two samples and uses only the high-confidence guesses.

```
Labeled data   ──► model ──► supervised loss (MSE / cross-entropy) + labeled ranking loss
Unlabeled data:
   weak view   (original data)    ──► model ──► "answer key" (pseudo-label / pair order)
   strong view (augmented data)   ──► model ──► must stay consistent with the answer key
Total loss = supervised + 0.2 × consistency   (only pairs with confidence ≥ 0.95)
```

The augmentation decides how much, and in which direction, the strong view differs from the original, so augmentation is the variable compared in this research.

**Research questions**

1. Which SSL methods improve on label-only baselines for each task and amount of labeled data?
2. Holding the SSL objective fixed, which augmentations improve results over identical weak and strong views?
3. How do input, latent, and TabPFN context augmentations compare, and how sensitive are their gains to the task and the metric?

## Results summary

Full numbers are in each stage's folder.

**Input-level augmentation** ([stage 4](4-augmentasi-input/RINGKASAN_SEMUA_EKSPERIMEN.md)):

- No single augmentation is always best. The winner changes between datasets, label percentages, and between validation and test.
- RankUp is generally better than no RankUp, especially with few labels (no Optuna, 1 seed, best augmentation: NHANES 1% QWK 0.150 → 0.251; Pen Digits 1% Macro F1 0.838 → 0.877).
- Variation between seeds can be as large as, or larger than, the difference between augmentations.
- Optuna does not always help, and many conditions are insensitive to augmentation because the best checkpoint is chosen in an early epoch, before the SSL part has time to work.

**Latent-level augmentation** ([stage 5](5-augmentasi-latent/README.md)): 945 runs, 3 seeds.

- RankUp helps mainly when labeled data is very scarce (1% labels).
- Input-level augmentation tends to fall below RankUp without augmentation, while latent-level augmentation is neutral or slightly above it. An added benefit is not proven, and the gap between operators is smaller than the standard deviation across seeds.
- Angular augmentation with controlled cosine (the supervisor's idea) is on par with the other latent operators.
- Limitations: 3 seeds, one data split, and the augmentation strength is not tuned.

## Research flow

Folders are ordered by when the work was done.

| Stage | Period | Folder | What was done |
|---|---|---|---|
| 1 | Jun–Jul 2026 | [`1-eksplorasi-rankup/`](1-eksplorasi-rankup/) | Learned RankUp and tried a small implementation on Energy Efficiency (Supervised vs RankUp vs Fully-Supervised) |
| 2 | Jul–Sep 2026 | [`2-rankup-ordinal/`](2-rankup-ordinal/) | RankUp on 5 ordinal datasets with MLP, TabM, and RealMLP; tabular model benchmark; CORAL/CORN objectives; 1/10/30% label sweep; **Task 1**: Optuna for weak/strong strength (TabCutMix, SCARF, Range-Limited, Gaussian) |
| 3 | Aug–Oct 2026 | [`3-pembanding-ssl/`](3-pembanding-ssl/) | **Task 3**: ordinal RankUp and RankUp-CLS compared with other tabular SSL methods (SSLAE, VIME, STUNT, GFTab) on 10 datasets: 100 runs on Modal |
| 4 | Nov 2026 (4–17) | [`4-augmentasi-input/`](4-augmentasi-input/) | **Thesis experiment**: 6 input-level augmentations on three main datasets; Optuna with 40 trials and 5 seeds, plus variants without Optuna, 13 datasets, and SSL baselines |
| 5 | Nov 2026 (17–) | [`5-augmentasi-latent/`](5-augmentasi-latent/) | Weak/strong augmentation moved to the TabM latent space (supervisor's guidance, meeting of 13 Nov) |
| 6 | Nov 2026 (21–) | [`6-ssr-rcus/`](6-ssr-rcus/) | Official SSR-RCUS code on the three tabular datasets with TabM (classification counterpart for Pen Digits): baseline, with angular augmentation, and with the earlier augmentations |
| 7 | Nov 2026 (26–) | [`7-tabpfn-latent/`](7-tabpfn-latent/) | Frozen TabPFN embeddings and latent augmentation on Kaggle. [Paired results for TabPFN 2.5 and 3.5](7-tabpfn-latent/RINGKASAN_2_5_DAN_3_5.md), with TabM comparisons from stages 5–6 |
| 8 | Nov 2026 (26–) | [`8-tabpfn-ssl/`](8-tabpfn-ssl/) | Controlled comparison of label-only, teacher distillation, and EMA self-training with identity, latent, and context views. Kaggle reports include historical RankUp, SSR-RCUS, and TabPFN controls |

How the stages connect: the five ordinal datasets from stage 2 are reused in stages 3 and 4 (`4-augmentasi-input/config/datasets_semua.json` points to `3-pembanding-ssl/dataset/`), and stage 5 uses the RankUp training loop and data splits from stage 4 (copied into `5-augmentasi-latent/` so it can run on its own).

Folder and file names are in Indonesian (`eksplorasi` = exploration, `pembanding` = baselines, `augmentasi` = augmentation, `hasil` = results, `gambar` = figures, `kode` = code, `referensi` = references, `tulisan` = writing).

## Experiment design (stages 4 and 5)

**Datasets**

| Task | Dataset | Rows | Features | Target | Main metric |
|---|---|---:|---:|---|---|
| Ordinal | NHANES ([TOC-UCO](https://www.uco.es/grupos/ayrna/materials/tocuco/)) | 5,223 | 30 | 5 levels | QWK (↑) |
| Classification | Pen Digits ([UCI](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)) | 10,992 | 16 | digit 0–9 | Macro F1 (↑) |
| Regression | Superconductivity ([UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data)) | 21,263 | 81 | critical temperature | MSE (↓) |

**Data split.** The label percentage is computed from the *train pool*; the rest becomes unlabeled data. Validation and test are the same for all label percentages and all seeds.

| Dataset | Train pool | 1% labeled | 10% labeled | 30% labeled | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| NHANES | 3,655 | 36 | 365 | 1,096 | 784 | 784 |
| Pen Digits | 5,995 | 59 | 599 | 1,798 | 1,499 | 3,498 |
| Superconductivity | 15,006 | 150 | 1,500 | 4,501 | 3,176 | 3,081 |

**Model and training**

- **TabM** backbone (3 blocks × 512, 32 parallel submodels). NHANES and Superconductivity use RankUp (regression head + ARC); Pen Digits uses RankUp-CLS, a FixMatch-style adaptation without ARC.
- AdamW (lr 0.002), batch of 32 labeled + 32 unlabeled, at most 120 epochs with early stopping (patience 15) on the validation metric. Unlabeled loss weight 0.2, confidence cutoff 0.95.
- Configuration and checkpoint selection always use **validation**; test is only reported.
- Training on a T4 GPU through [Modal](https://modal.com/).

**Baselines:** `supervised` (no RankUp) and `rankup_noaug` (RankUp with no augmentation at all).

**Stage 4: six input-level augmentations** (strength = fraction of features changed; for Gaussian: σ)

| Augmentation | How it works |
|---|---|
| TabCutMix | Swaps some features with a donor in the mini-batch that has the same pseudo-label |
| TabCutMix+ | Like TabCutMix, but swaps groups of correlated features |
| SCARF | Replaces selected features with values from another random row |
| Range-limited | Draws a random value within the feature's local quantile range |
| Gaussian | Adds Gaussian noise to the standardized features |
| Zero-out | Replaces selected features with 0 |

Each combination (3 datasets × 3 label percentages × 6 augmentations = 54) was run with Optuna (40 trials) to find the weak and strong strengths, then retrained with 5 seeds.

**Stage 5: latent-level augmentation.** The perturbation is attached through a hook on the TabM backbone output (`h`, size B × 32 × 512), only on unlabeled data during training. Weak = unchanged, strong = 0.5, no Optuna. The same operators (Gaussian, zero-out, SCARF, TabCutMix) are compared at two positions (input and latent), plus dropout, F-noise, and angular augmentation with controlled cosine. In total 35 configurations × 27 conditions (3 datasets × 3 label percentages × 3 seeds) = 945 runs. The `input_*` configurations give scores identical to the stage 4 experiment without Optuna (seed 42), as a check that the new code does not change the training loop.

## Folder structure

```
.
├── 1-eksplorasi-rankup/      small notebooks (Jun–Jul)
├── 2-rankup-ordinal/         Tasks 1 and 2: Kaggle notebooks, results, Optuna
├── 3-pembanding-ssl/         Task 3: Modal runners, results, siapkan_paper_repo.sh
├── 4-augmentasi-input/       thesis experiment: code, data, results, RINGKASAN_SEMUA_EKSPERIMEN.md
├── 5-augmentasi-latent/      latent augmentation: code, data, results, figures, documents
├── 6-ssr-rcus/               SSR-RCUS baseline: adaptor for the official code, Modal runner, siapkan_paper_repo.sh
├── 7-tabpfn-latent/          latent operators in TabPFN embeddings, Kaggle notebook and results for 2.5 and 3.5
├── 8-tabpfn-ssl/            TabPFN SSL: Kaggle code, experiment report, charts and architecture
├── referensi/                45 papers (PDF) with their sources, a review of each, and the official RankUp code (MIT)
└── tulisan/                  thesis drafts (Chapter II, IEEE and Markdown formats)
```

Each stage has its own README and environment, so there is no single environment for the whole repo.

## How to run

General prerequisites: Python 3.12 for stages 2 and 4 (3.10 or newer for the others), [`uv`](https://docs.astral.sh/uv/), and a [Modal](https://modal.com/) account for GPU training in stages 3, 4, and 5 (`pip install modal`, then `modal setup`). Stage 2 consists of Kaggle notebooks.

### Stage 8: context augmentation on Kaggle

Import [the stage 8 notebook](8-tabpfn-ssl/kaggle_tabpfn_konteks.ipynb) and attach the data already used for stage 7. The notebook starts with a smoke run on Kaggle. The [stage 8 README](8-tabpfn-ssl/README.md) covers the full grid, output archive, and resume.

### Stage 5: latent-level augmentation

```bash
cd 5-augmentasi-latent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd kode

PYTHONPATH=. python test_latent.py -v     # unit tests (CPU)
python ringkas.py                         # builds ../hasil/ringkasan.md from results.jsonl
jupyter lab visualisasi.ipynb             # result plots; run all cells
```

Run the grid on Modal (replace `<profile>` with your Modal profile, or remove `MODAL_PROFILE`):

```bash
MODAL_PROFILE=<profile> modal run modal_latent.py                                  # tests + 2-epoch smoke run
MODAL_PROFILE=<profile> modal run modal_latent.py --action run --names fase1       # one group of configurations
MODAL_PROFILE=<profile> modal run modal_latent.py --action run --names angular --seeds 42 --percents 10
```

Runs already in `hasil/results.jsonl` are skipped, so the same command is safe to repeat if it gets interrupted. The list of configuration names and their explanations is in [`5-augmentasi-latent/README.md`](5-augmentasi-latent/README.md); the plan and design decisions are in [`dokumen/PLAN_EKSPERIMEN_LATENT.md`](5-augmentasi-latent/dokumen/PLAN_EKSPERIMEN_LATENT.md).

### Stage 4: input-level augmentation

```bash
cd 4-augmentasi-input
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace eda/eda_datasets.ipynb     # local EDA

# prepare the data splits, then run all combinations on Modal
uv run modal run eksperimen/modal_app.py --action prepare --dataset all --label-percent all
uv run modal run --detach eksperimen/modal_app.py --action train --dataset all --method all --label-percent all --run-id ta-001

# quick test of one combination
uv run modal run eksperimen/modal_app.py --action prepare --dataset nhanes --label-percent 1
uv run modal run --detach eksperimen/modal_app.py --action train --dataset nhanes --method range_limited \
  --label-percent 1 --run-id smoke-001 --trials 2 --optuna-epochs 3 --epochs 3

# download and summarize results
uv run modal volume get experiment-ta-results ta-001 eksperimen/hasil
uv run python eksperimen/ringkas_hasil.py eksperimen/hasil/ta-001 --output eksperimen/hasil/ta-001-ringkasan.csv
```

If a job is interrupted, rerun the same `train` command with the same `--run-id`; finished trials and seeds are reused. Visualizations of all results are in `eksperimen/visualisasi_eksperimen.ipynb`, and the full summary of all experiments is in [`RINGKASAN_SEMUA_EKSPERIMEN.md`](4-augmentasi-input/RINGKASAN_SEMUA_EKSPERIMEN.md).

### Stage 3: SSL baselines

```bash
cd 3-pembanding-ssl
pip install -r requirements.txt
./siapkan_paper_repo.sh                   # downloads the VIME, STUNT, SSLAE, GFTab code at the pinned commits

modal run modal_app.py --dataset all --action prepare
modal run modal_app.py --dataset all --action rankup          # ordinal RankUp
modal run modal_app.py --dataset classification --action rankup_cls
modal run modal_app.py --dataset all --action vime            # also: sslae, stunt, gftab
```

The full commands (including `--action all` and how to download results from the Modal Volume) are in [`3-pembanding-ssl/README.md`](3-pembanding-ssl/README.md).

### Stages 1 and 2: notebooks

These are self-contained notebooks designed for Kaggle (GPU). Results are stored in the notebook outputs, so they only need to be opened (`2-rankup-ordinal/hasil/` and `hasil-tabm/`). For local stage 2 dependencies: `cd 2-rankup-ordinal && uv sync`. Each stage's README explains what each notebook contains.

## References

[`referensi/`](referensi/) contains 45 papers (PDF) with links to their sources ([`referensi/README.md`](referensi/README.md)), a review of each paper ([`referensi/papers/DAFTAR_PAPER.md`](referensi/papers/DAFTAR_PAPER.md), in Indonesian), and a copy of the official RankUp code (MIT).

## Credits

This research stands on the work of many researchers. Thank you to the authors.

**Papers.** The full list of 45 papers, with authors, venues, and source links, is in [`referensi/README.md`](referensi/README.md); the BibTeX is in [`referensi/referensi.bib`](referensi/referensi.bib). The works this research builds on directly:

| Work | Used for |
|---|---|
| Huang, Fu, Tsao. [*RankUp*](https://arxiv.org/abs/2410.22124), NeurIPS 2024 | The semi-supervised method studied throughout this research |
| Gorishniy, Kotelnikov, Babenko. [*TabM*](https://arxiv.org/abs/2410.24210), ICLR 2025 | Model backbone |
| Bahri et al. [*SCARF*](https://arxiv.org/abs/2106.15147), ICLR 2022; Fang et al., *TabCutMix*, ICML 2025; Lee et al., *Range-limited (FeSTA)*, KDD 2025 | Input-level augmentations compared (stages 2–4) |
| Yang et al. [*UniMatch*](https://arxiv.org/abs/2208.09910), CVPR 2023; Wu & Cui [*IFMatch*](https://arxiv.org/abs/2408.12614), ECCV 2024; Ouali et al. [*CCT*](https://arxiv.org/abs/2003.09005), CVPR 2020 | Basis for feature-level perturbation (stage 5) |
| Eo et al. [*RaTab*](https://doi.org/10.1609/aaai.v39i11.33265), AAAI 2025; Fang et al. [*SDAT*](https://doi.org/10.1145/3511808.3557699), CIKM 2022; Li et al. [*SSR-RCUS*](https://doi.org/10.1609/aaai.v40i28.39487), AAAI 2026 | Closest related work on latent augmentation for tabular data and RankUp |
| Hollmann et al. [*TabPFN v2*](https://doi.org/10.1038/s41586-024-08328-6), Nature 2025; Grinsztajn et al. [*TabPFN-2.5*](https://arxiv.org/abs/2511.08667), 2025; Margeloiu et al. [*TabMDA*](https://arxiv.org/abs/2406.01805), ICML 2024 Workshop on In-Context Learning | Frozen encoder and teacher (stage 7); in-context subsetting as the augmentation (stage 8 design) |

**Code.** The official RankUp code ([pm25/Semi-Supervised-Regression](https://github.com/pm25/Semi-Supervised-Regression), MIT license, copyright Microsoft Corporation and Pin-Yen Huang) is included in [`referensi/Semi-Supervised-Regression/`](referensi/Semi-Supervised-Regression/) together with its license. The SSR-RCUS baseline (stage 6) uses the official code ([changchunli/SSR-RCUS](https://github.com/changchunli/SSR-RCUS), MIT license), which is not included here and is downloaded unchanged with `6-ssr-rcus/siapkan_paper_repo.sh`. The stage 3 baselines use the original authors' code, which is not included here and is downloaded with `3-pembanding-ssl/siapkan_paper_repo.sh`: [VIME](https://github.com/jsyoon0823/VIME), [STUNT](https://github.com/jaehyun513/STUNT), [SSLAE](https://github.com/sintija-s/sslae), and [GFTab](https://github.com/Yoontae6719/Geodesic-Flow-Kernels-for-Semi-Supervised-Learning-on-Mixed-Variable-Tabular-Dataset).

**Data.** NHANES from [TOC-UCO](https://www.uco.es/grupos/ayrna/materials/tocuco/); [Pen Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) and [Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data) from the UCI Machine Learning Repository; Satimage, Diabetes Pima, CMC, Bank Marketing, and Adult from [OpenML](https://www.openml.org/) (IDs 182, 37, 23, 1461, 1590).

**Paper copyright.** The PDF copies in `referensi/papers/` are included only for academic and reference purposes. The copyright of each paper remains with its original authors and publisher. If you are a rights holder and want a copy removed, open an issue in this repo.

## Writing

Drafts of Chapter II (Literature Review) are in [`tulisan/`](tulisan/), in Markdown and IEEE LaTeX formats.
