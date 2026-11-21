# Augmentation candidate: bounded local transport

This candidate is an **exploratory hypothesis**, not a method proven to be better and not a final novelty claim. It combines local-neighbor interpolation, a distance bound, and pseudo-label gating. Interpolation and nearby pairs have been studied, for example in [ICT](https://www.ijcai.org/Proceedings/2019/0504.pdf), [C-Mixup](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html), and [SSR-RCUS](https://ojs.aaai.org/index.php/AAAI/article/view/39487). The literature map and code comparison are in the [research notes](../eksperimen/catatan/riset_augmentasi_usulan.md).

## The problem being tested

The six earlier operators do not have the same validation winner across all combinations. Feature swapping and noise can produce combinations that break the structure, while a donor whose class is guessed by the model can be wrong. These constraints differ between datasets. In the 270 old runs, the mean pseudo-rank acceptance for NHANES 30% at the selected epoch is only **0.0043**; this is an observation on the current code and split, not proof of the cause of the low performance. Run `uv run python usulan/analisis_riwayat.py` to recompute the audit.

A preliminary check **on labeled train data only** shows that Euclidean closeness also does not guarantee target closeness. For example, on the 1% subset, the nearest labeled neighbor has the same NHANES level for 16/36 anchors (44.4%), and the same Pen Digits class for 48/59 anchors (81.4%). This is only a descriptive statistic from a small subset, not an estimate of donor accuracy on unlabeled data. For Superconductivity, the mean absolute target difference between 150 labeled anchors and their nearest labeled neighbor is about 13.04, with a labeled target standard deviation of 32.44. This computation does not use validation or test labels.

## Method definition

For a mini-batch `x_i ∈ R^d` standardized with the train pool:

1. Compute the per-feature distance `D(i,j) = ||x_i - x_j||₂ / √d` and forbid the donor `j = i`.
2. For classification, find the nearest donor with the **same class pseudo-label** and initial confidence ≥ 0.8 for both the receiver and the donor. For ordinal, use the rounded ordinal label and the project code's built-in heuristic confidence. These pseudo-labels can be wrong and their confidence is not assumed to be calibrated.
3. For continuous regression, the baseline pipeline does not pass the donor's prediction or uncertainty to the operator. This prototype uses the nearest geometric donor **without target gating**; it does not claim that the donor has the same target or ranking.
4. If no donor qualifies, use the identity `x'_i = x_i`. If one does, use `x'_i = x_i + a min(1, τ / max(D(i,j), ε)) (x_j - x_i)` with `τ = 0.5`, `a_weak = 0.1`, and `a_strong = 0.3`. The search is limited to the unlabeled mini-batch; the batch members change as the DataLoader shuffles the data.

Interpolating the whole vector preserves a certain linear relation between two points, but it **does not guarantee** that the sample lies on the data manifold or is semantically valid. Fractional values on discrete features, categorical/one-hot features, nonlinear physical constraints, and geometric neighbors with different targets are risks; this prototype is only for the three numeric datasets tested. There is no guarantee that the mini-batch nearest neighbor is the nearest neighbor in the whole dataset. The identity fallback can also make the change too small and the consistency less informative.

For ordinal and regression, the operator is attached to the weak/strong views in the existing tabular RankUp pipeline, including the auxiliary ranking classifier. For nominal classification, the project pipeline **has no ARC**; the operator is tested on a FixMatch-style class-consistency adaptation, not the original regression RankUp. Implementation: [local_transport.py](local_transport.py).

## Conditional mathematical bound

With `0 ≤ a ≤ 1` and `τ > 0`, the formula above algebraically guarantees `||x'_i-x_i||₂ / √d ≤ aτ`, up to numerical tolerance. With the pilot parameters, the bound for the strong view is `0.3 × 0.5 = 0.15` in RMS of standardized features. The tests check this bound and the donor fallback.

If, as an extra unverified assumption, the true target function `f*` is `L`-Lipschitz along that segment, then `|f*(x'_i)-f*(x_i)| ≤ L aτ√d`. For two samples with the same bound on target change, the target order is preserved when the true target margin exceeds `2 L aτ√d`. `L` and the unlabeled labels are unknown, and the assumption can fail. This bound is **not proof** that the augmentation preserves labels/ranks on real data, or that QWK, F1, or MSE will improve. For classification, a separate class-margin assumption is needed.

## Cheap pilot

One seed 42, no Optuna, and no reading of test: 3 datasets × 3 label fractions × 9 methods = 81 jobs. The methods are the six old augmentations, an identity control with SSL, a labeled-objective control with unlabeled weight zero, and this candidate. All use at most 30 epochs, 40 steps per epoch, batch 32, early stopping 15, and fixed weak/strong strengths of 0.1/0.3. The supervised model on ordinal/regression still has a labeled ranking loss; preprocessing is still transductive because it was fit on the whole train pool earlier. So it is not a fully supervised model without access to the unlabeled features.

Use the validation values: higher QWK and macro-F1 are better; lower MSE is better. Strength 0.1/0.3 does not have the same meaning or level of damage for each operator, so this is **equal-budget screening**, not a calibrated final comparison. The old methods were tuned with 40 trials and trained with a different budget, so the pilot numbers cannot be compared directly as evidence of superiority over the old result tables. Do not choose the design from test; the project's test has been seen before, so strong confirmation needs a new/external holdout.

Per-job timeout is 300 seconds, with at most 3 T4 GPUs at the same time. This limits jobs but is not a hard billing cap; startup, CPU/memory, infrastructure retries, and storage can add cost. This pilot is not five seeds, does not prove statistical stability, and does not provide between-split uncertainty. For a claim: fix the protocol before the confirmation experiment, compare with SSR-RCUS and relevant tabular methods, ablate the gating/distance/interpolation, run repeated validation across splits and paired seeds, use independent data that was not used to choose the method, and report differences and uncertainty intervals with the correct evaluation unit.

The pilot results in the Modal Volume `experiment-ta-results/pilot-local-transport-20261109/` record scores and validation history, epochs, acceptance rate, input checksums, and transport diagnostics (donor coverage, distance bound, RMS change). In this task's local data, the validation labels are used for model selection; the test labels were not opened by the pilot. Model checkpoints were not saved in the pilot; for the final research, complete tracking of model artifacts and the environment needs to be added.
