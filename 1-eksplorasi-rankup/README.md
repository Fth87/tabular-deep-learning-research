# Stage 1 — RankUp exploration (June–July 2026)

Learning RankUp (Huang et al., NeurIPS 2024), a semi-supervised regression method with an auxiliary ranking classifier, and trying a small implementation before moving on to the ordinal datasets in stage 2. The official RankUp code was used as a reference; its copy is in `../referensi/Semi-Supervised-Regression`.

| Notebook | Contents |
|---|---|
| `rankup_minimal_energy_efficiency.ipynb` | Minimal RankUp on the Energy Efficiency dataset (UCI): Supervised, minimal RankUp, and Fully-Supervised |
| `rankup_energy_efficiency_clean.ipynb` | A cleaned-up version of the notebook above |
| `inspect_convnextv2_errors.ipynb` | Error analysis of ConvNeXtV2 predictions (test images with label 0 predicted as 2) |

Next: [`../2-rankup-ordinal/`](../2-rankup-ordinal/).
