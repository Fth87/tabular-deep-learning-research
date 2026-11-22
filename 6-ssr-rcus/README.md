# SSR-RCUS baseline on tabular data (TabM)

The official [SSR-RCUS](https://github.com/changchunli/SSR-RCUS) code (Li et al., AAAI 2026, MIT license) run on this project's three tabular datasets with a TabM backbone: NHANES (ordinal, QWK), Pen Digits (classification, macro-F1), and Superconductivity (regression, MSE). This answers the supervisor's items 1–3:

| Name | Supervisor's item | What runs |
|---|---|---|
| `rcus` | 1. SSR-RCUS baseline with TabM, nothing else changed | Official SSR-RCUS; weak = strong = original row, so latent manifold mixup is the only perturbation |
| `rcus_angular`, `rcus_angular_weak` | 2. SSR-RCUS with angular/cosine augmentation | Same, plus the latent `angular` operator on the strong view (cos 0.3–0.5), and in `_weak` also on the weak view (cos 0.7–0.95) |
| `rcus_input_{gaussian,zero_out,scarf}` | 3. SSR-RCUS with the earlier augmentations | Same, plus the earlier input operator (strength 0.5) on the strong view; each dataset uses its own task family's operator code |
| `rankup` | control | Official RankUp from the same repo and pipeline, so SSR-RCUS is compared under identical conditions |

Operator settings are taken unchanged from `GRID` in [`../5-augmentasi-latent/kode/grid.py`](../5-augmentasi-latent/kode/grid.py), and the operator code is the same code (`latent.perturb`, `experiments/rankup/<op>.augment_method`).

## How the official code is used

`paper_repo/` is a clean clone of the official repo at commit `e79e53f`, made by `./siapkan_paper_repo.sh` (same pattern as `3-pembanding-ssl/`). It is **not modified** and is not committed here, since the code and its license belong to the original authors. All adaptation is in [`tabular_rcus.py`](tabular_rcus.py):

| Piece | Why it is needed |
|---|---|
| `TabMNet` | Gives TabM the semilearn network interface (`logits`, `feat`, `only_feat`, `only_fc`), like the official WideResNet |
| `TabularData.set_dataset` | Loads the `.npz` splits; the official code only has image/text/audio loaders |
| `LatentViews.train_step` | Attaches the latent operator to the weak/strong views with a forward hook, then calls the algorithm's `train_step` unchanged |
| `TabularCls`, `TabularRankUpCls`, `TabularRCUSCls` | Classification counterparts for Pen Digits (see below) |
| `run_one` | Mirrors `train.py`'s `main_worker` (one GPU), then scores validation and test |

A unit test checks that, for regression and ordinal data, `TabularRCUS` and `TabularRankUp` keep the official `train_step` and `arc_manifold_mixup`, and that only `set_dataset` is replaced.

### Pen Digits (classification)

The official code is regression-only: min-max target scaling, a single regression output, MAE-based checkpoint selection, and a ranking head whose mixup target relies on the targets having an order. Digit classes have no order, so Pen Digits uses the closest counterparts. They are built from the official parts: the masking and pseudo-label hooks, `ce_loss`, `cls_consistency_loss`, EMA, the λ sampling, and the loss weights.

| Name | On Pen Digits |
|---|---|
| `rankup` | RankUp-CLS as in stages 4–5: cross-entropy on labeled rows + 0.2 × cross-entropy on the strong view against confident (≥ 0.95) weak-view class pseudo-labels |
| `rcus` and its variants | **SSR-RCUS-CLS**: every pair (i, j) of strong-view features is mixed as `λ·z_j + (1−λ)·z_i`, with λ drawn exactly as in the official code. The target is `λ·y_j + (1−λ)·y_i` over one-hot pseudo-classes, and a pair is used only if both samples are confident. Labeled rows get the same mixup with true labels. The weights are the official SSR-RCUS ones: 0.2 for the labeled mixup and 1.0 for the unlabeled mixup |

SSR-RCUS's monotonicity assumption (a mixed sample ranks between its sources) is replaced here by the usual mixup assumption (the label changes linearly between two samples). This is an adaptation and not the official method, and it should be reported as such. Targets are class indices (identity scaler). Checkpoints are selected on validation macro-F1: the official hook keeps the lowest `eval/mae`, so the classification `evaluate` reports `1 − macro-F1` under that key.

### Settings

- **Kept from the official config:** alpha 0.5, T 0.5, p_cutoff 0.95, ARC loss weights 0.2/1.0, EMA 0.999, batch 32, uratio 7, L1 loss, min-max target scaling, labeled rows included in the unlabeled pool.
- **Changed for the tabular data and backbone:**
  - Budget is 120 epochs × 100 steps, with validation every 100 steps (the official config uses 262,144 steps for images).
  - Optimizer is AdamW with lr 2e-3, weight decay 3e-4, and a constant LR. This is the project's TabM recipe; the official SGD 0.01 is for WideResNet.
  - No Optuna.

### Two fixes in the evaluation protocol

These are protocol fixes only; the algorithm itself is not touched.

1. **Model selection on test.** The official config passes the test set as `eval`, so the best checkpoint is chosen on test. Here `eval` is the validation set, and test is only scored once, at the end.
2. **Stale EMA weights at test time.** After training, the official hook reloads `model_best.pth`, but `evaluate()` still uses the EMA shadow from the *last* step. `run_one` re-syncs the shadow (`alg.ema.load(alg.ema_model)`), so validation and test scores really come from the best checkpoint.

### Known differences and limits

- **TabM ensemble:** TabM trains one head and one loss per submodel (k = 32). The official algorithm needs one feature vector and one prediction, so the 32 submodel embeddings are averaged into `feat` (B × 512) and the official heads sit on top. The latent operators still act per submodel, before the average, as in stage 5.
- **Regression consistency term:** the official code inherits `reg_loss = sup_loss + ulb_loss_ratio` from RankUp, so the regression consistency term is inactive and only the ARC terms use unlabeled data. It is kept as is.
- **Model selection metric:** for NHANES and Superconductivity, the best checkpoint is chosen by validation MAE (official), not by QWK as in stages 4–5. Test QWK/MSE are computed with the same `scores` function as stage 5. For that reason, compare against the `rankup` control here rather than against stage 5's numbers.
- **No TabCutMix in item 3:** TabCutMix needs the model's pseudo-labels, so it cannot be an input transform without changing the algorithm.

## Running

Unit tests need torch, so they run on Modal:

```bash
cd 6-ssr-rcus
./siapkan_paper_repo.sh                                                          # once
MODAL_PROFILE=lombaaaa modal run modal_rcus.py                                    # unit tests + 200-step smoke run
MODAL_PROFILE=lombaaaa modal run modal_rcus.py --action run --names rankup,rcus --seeds 42 --percents 1   # pilot
MODAL_PROFILE=lombaaaa modal run modal_rcus.py --action run --names all           # 7 variants × 3 datasets × 3 labels × 3 seeds = 189 runs
```

Results are appended to `hasil/results.jsonl`, one line per run. The format matches stage 5: `validation` and `test` metrics, `best_it`, and `seconds`. Runs already in the file are skipped, so an interrupted command can be repeated.

## Results

All 189 runs finished (7 variants × 3 datasets × 3 label fractions × 3 seeds, one split, about 21 GPU-hours on T4). `python ringkas.py` builds [`hasil/ringkasan.md`](hasil/ringkasan.md) from `hasil/results.jsonl`: test scores (mean ± std over seeds) and, per dataset, how many of the 9 paired blocks each variant wins. No variant is chosen from test.

Scores here come from the official SSR-RCUS pipeline (EMA, L1 loss, validation-MAE checkpoint, averaged TabM features), so they are lower than stage 5's numbers and should only be compared with the `rankup` control in this folder.
