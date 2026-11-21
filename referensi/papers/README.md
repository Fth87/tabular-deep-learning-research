> The full list of all papers with their source links is in [`../README.md`](../README.md). This document holds the relevance table and best practices for the latent experiment.

# References: augmentation in latent / embedding space for SSL (tabular focus)

Collected on 17 November 2026. All PDFs in this folder have been checked: each file is a real PDF and the title on its first page matches. RaTab, SDAT, and MODALS were downloaded manually and read in full; SDAT is a scan, so its text was extracted with OCR.

A full explanation per paper (full title, core idea, method) is in **`DAFTAR_PAPER.md`** (in Indonesian).

Markers in the "Evidence" column:
- ✔ content checked directly from the PDF;
- ◐ from the abstract or search results only.

## 1. Latent augmentation specific to tabular data (`1_latent_tabular/`)

| Paper | Venue | What it does in the latent space | Relevance for us | Evidence |
|---|---|---|---|---|
| **Eo et al., RaTab: Representation Space Augmentation for Effective Self-Supervised Learning on Tabular Data** | AAAI 2025 | Augmented view = a repeated forward pass with the **encoder's last-layer weights replaced by a truncated-SVD version** (rank 50/60/70% of full; 70% is most often best) + dropout 0.1 in the encoder. InfoNCE + reconstruction loss (λ = 0.5), then fine-tuning. 13 OpenML datasets (classification + regression), MLP / FT-Transformer / T2G-Former backbones | **Closest prior art (self-supervised).** The claim "first latent augmentation for tabular" cannot be used. Useful details: (1) perturbing the **last layer** is best, perturbing many layers adds nothing; (2) contrastive mixup in the latent beats input augmentation (masking, shuffling, CutMix); (3) it perturbs the **weights**, not the vector z directly, so it can serve as an "RaTab-style" baseline on TabM | ✔ |
| Darabi et al., Contrastive Mixup | arXiv 2021 | Mixup in a low-dimensional latent (manifold assumption), because mixup at the input produces table rows that cannot exist. Unlabeled data is used through label propagation | An example of **semi-supervised tabular** with latent augmentation | ✔ |
| Somepalli et al., SAINT | arXiv 2021 | Contrastive pre-training: CutMix at the input + **mixup in the embedding space** | A mix of input and latent augmentation | ✔ |
| Levin et al., Transfer Learning with Deep Tabular Models | ICLR 2023 | Positive pairs: CutMix at the input + Mixup in the embedding (following SAINT) | Same as above | ✔ |
| Wu et al., SwitchTab | AAAI 2024 | Asymmetric autoencoder that separates "mutual" and "salient" features in the latent, then swaps them between pairs | An example of tabular latent manipulation (swap) | ✔ |
| **Onishi, Rethinking Data Augmentation for Tabular Data in Deep Learning** | arXiv 2023 | Compares augmentations for self-supervised tabular learning | **Important finding:** "learning with Manifold Mixup had the only negative effect". Interpolation in the latent does not automatically help on tabular data | ✔ |

## 2. Feature/latent augmentation for semi-supervised learning (general, mostly vision) (`2_latent_ssl_umum/`)

| Paper | Venue | Core idea | Lesson for our design | Evidence |
|---|---|---|---|---|
| **Yang et al., UniMatch** | CVPR 2023 | FixMatch + **an extra feature-perturbation stream**: `feat_fp = Dropout(0.5)(g(x_weak))`. The pseudo-label comes from the unperturbed weak view; feature perturbation and input augmentation are considered **complementary** | Latent perturbation is used as an **addition**, not a replacement for input augmentation. Dropout 0.5 is already strong enough as a baseline | ✔ |
| **Wu & Cui, Image-Feature Weak-to-Strong Consistency (IFMatch)** | ECCV 2024 | **Weak and strong** feature perturbation **at two different positions**; "movement", "dropout", "value" strategies; deliberately **sample-agnostic** so it is not hit by wrong pseudo-labels; three-branch structure | **Closest to the supervisor's guidance** (weak/strong at the feature level). Latent operators that depend on pseudo-labels (latent TabCutMix) are risky | ✔ |
| Ouali et al., Cross-Consistency Training (CCT) | CVPR 2020 | Perturbation on the encoder output. F-Noise: z + z·U(−0.3, 0.3). F-Drop: drop with a threshold ∈ [0.6, 0.9]. Dropout 0.5, VAT. Consistency between the main decoder and auxiliary decoders | A list of standard latent operators + default values that have been tested | ✔ |
| Kuo et al., FeatMatch | ECCV 2020 | Feature augmentation learned from intra-/inter-class prototypes (memory bank), used in a consistency loss | A more complex variant; depends on class pseudo-labels | ✔ |
| Verma et al., Manifold Mixup | ICML 2019 | Interpolation of hidden states | Theoretical basis of latent augmentation; bad results on tabular data (see Onishi) | ✔ |
| DeVries & Taylor, Dataset Augmentation in Feature Space | ICLR-W 2017 | Noise, interpolation, **extrapolation** between neighbors in feature space: c′ = (cⱼ − cₖ)λ + cⱼ, λ = 0.5 | Extrapolation is worth testing in addition to interpolation | ✔ |
| Osada et al., Latent Space VAT | arXiv 2020 | VAT in the latent space (from a generative model); adversarial samples are more flexible than at the input | An adversarial latent perturbation option (later phase) | ✔ |
| Wang et al., ISDA | NeurIPS 2019 | Shifts features along semantic directions from the per-class covariance | The perturbation direction can be chosen in a structured way, not necessarily random | ✔ |
| Cheung & Yeung, MODALS | ICLR 2021 | **Supervised**, not SSL. 4 latent operations: interpolation toward a hard same-class sample, extrapolation away from the class center, Gaussian noise with per-class σ, difference transform z + λ(zⱼ − zₖ) from two same-class samples. Probability/magnitude searched with PBA (λ ∈ [0, 0.9]). Plus a triplet loss + adversarial loss to keep the latent compact. Tabular: 5 small UCI datasets, 2-layer MLP, latent 128, 3 trials | All its operations **need class labels**; in our SSL that means pseudo-labels (risky, see IFMatch). A class-free version (σ from the batch, difference transform from two random rows) can be used. Gains on small tabular data (e.g. Abalone 64.6 → 65.7) and no statistical test. Latent augmentation alone gives +1.6 points on average; the rest comes from the extra losses | ✔ |
| **Fang et al., SDAT** | CIKM 2022 | **Semi-supervised tabular** (classification). A VAE encoder is shared with the classifier; augmented view = sample z ~ N(μ, σ) whose **σ is enlarged** through β regularization, then decoded into x̂. KL consistency KL(SG(h(x)) ‖ h(x̂)): the **prediction on the original, unperturbed data + stop-gradient** is the target. 6 datasets (64/128 labels), 5 trials + paired t-test | **Closest semi-supervised prior art** (located in folder 1_latent_tabular). Relevant findings: (1) Gaussian noise at the **input** (γ = 0.5·std) barely helps (UDA, ReMixMatch ≈ DNN); (2) a separately trained VAE ≈ Gaussian noise; the benefit comes from **joint training**, because the latent becomes task-related; (3) the best β is very small (0.002): latent perturbation must be small. Classification only, no ordinal/regression | ✔ (OCR) |
| **Li et al., SSR-RCUS** | AAAI 2026 | An extension of **RankUp** (same code and 0.95 threshold). Mixup in the latent z = e(x) to build ranking pairs between nearby samples; the order is guaranteed through a monotonic assumption. Data: UTKFace, Yelp Review, BVCC (not tabular) | Closest to our work. Uses latent mixup in place of weak/strong consistency, so it can be an extra latent operator and a baseline | ✔ |
| Fan et al., FeatDistLoss | IJCV 2023 | Consistency in the classifier, **equivariance** (pushing feature distances apart) at the feature level; augmentation stays at the input | Pushing features apart between views can help; a reference for ablating weak/strong similarity | ◐ |
| Kurtulus et al., Tied-Augment | arXiv 2023 | A feature-similarity loss term between two views with shared weights; includes Tied-FixMatch (images) | Similarity is controlled through a loss, not through vector construction like angular | ◐ |
| Yang et al., UniMatch V2 | arXiv 2024 | Input and feature streams merged; Complementary Dropout (two complementary feature sets for two strong views); pretrained ViT encoder | Two strong views with complementary feature perturbations; supports the pretrained-encoder scenario | ◐ |
| Wang et al., CLFP | Algorithms 2026 | Weak/strong augmentation at the input + two feature dropouts (0.5) + complementary label; CIFAR-10 and STL-10 | An input + feature hybrid, similar to our input + latent condition | ◐ |

## 3. General tabular augmentation and SSL (`3_tabular_ssl_augmentasi/`)

| Paper | Venue | Relevance |
|---|---|---|
| Wang et al., A Survey on SSL for Non-Sequential Tabular Data | ACML 2024 | A map of tabular SSL methods (predictive, contrastive, hybrid). For Chapter II |
| Ren et al., Deep Learning within Tabular Data | arXiv 2025 | A survey of 127 papers, including augmentation |
| Cui et al., Class-Conditioned and Feature-Correlation Based Augmentation | arXiv 2024 | Pseudo-label-conditioned input corruption + correlation-based masks; the authors found no strong evidence that correlation masks help |
| Lawson et al., Mutual Information-Guided Corruption | ICML 2026 Workshop | Input corruption per group of high-MI features, new values from a conditional VAE. In line with the TabCutMix+ idea |
| Thimonier et al., T-JEPA | ICLR 2025 | Tabular SSL **without augmentation**: predicts latent representations. A conceptual alternative |
| Bahri et al., SCARF | ICLR 2022 | The input augmentation baseline we already use |

## 4. Backbones, adapters, and base methods (`4_backbone_adapter/`)

| Paper | Relevance |
|---|---|
| Huang et al., RankUp (NeurIPS 2024) | The SSL framework used |
| Gorishniy et al., TabM (ICLR 2025) | The current encoder |
| Chen et al., AdaptFormer (NeurIPS 2022) | An adapter for pretrained Vision Transformers. Mentioned by the supervisor; only meaningful for a Transformer backbone |
| Zhu et al., XTab (ICML 2023) | A candidate **pretrained tabular encoder** (cross-table Transformer) for Scenario 2 |

## 5. Prior art for the angular / cosine augmentation idea (`5_angular_prior_art/`)

| Paper | Core idea | Impact on the claim |
|---|---|---|
| **Kobayashi, Rotation Regularization Without Rotation** | ECCV 2022. Rotates a feature vector toward a random direction by a **fixed angle α**, forming a cone around the feature that gives an angular margin. Used as classifier regularization (supervised) | **Closest prior art for the supervisor's idea.** Feature rotation with a controlled angle already exists. Possible differences: (a) used as the **weak/strong** views in SSL consistency, (b) different cosine ranges for weak and strong, (c) in tabular RankUp for ordinal/regression. Must be cited |

A search for "angular / cosine-controlled perturbation + consistency + tabular" did **not** find an identical paper. But this search is not exhaustive, so it is not proof of novelty.

## 6. Status of manual downloads

| Paper | Status |
|---|---|
| RaTab (AAAI 2025) | Present: `1_latent_tabular/Eo2025_RaTab_Representation_Space_Augmentation_AAAI.pdf` |
| SDAT (CIKM 2022) | Present: `1_latent_tabular/Fang2022_SDAT_SSL_Data_Augmentation_Tabular_CIKM_scan.pdf` (scan, no text layer) |
| MODALS (ICLR 2021) | Present: `2_latent_ssl_umum/Cheung2021_MODALS_Latent_Space_Augmentation_ICLR.pdf` |
| Mutual Information-Guided Corruption (ICML 2026 Workshop) | Present: `3_tabular_ssl_augmentasi/Lawson2026_MI_Guided_Corruption_Tabular_SSL.pdf`. **Input** augmentation (feature groups from mutual information + CVAE), self-supervised |
| SSR-RCUS, UniMatch V2, FeatDistLoss, Tied-Augment, CLFP | Present in `2_latent_ssl_umum/` (downloaded 18 Nov 2026). Two duplicate files (IFMatch and TabM) were deleted because they were identical to existing ones |

---

## 7. Best practices to take (and their impact on the plan)

1. **Latent augmentation is most often used as an addition, not a replacement for input augmentation.**
   - UniMatch: feature stream + input stream, both guided by one weak view.
   - IFMatch: input and feature perturbation are combined in every branch.
   - SAINT and Levin: input CutMix + embedding mixup.

   → The plan needs an **"input + latent"** condition in addition to "latent only".
2. **Pseudo-labels are taken from the weak view without feature perturbation** (UniMatch). Feature perturbation exists only in the branch being trained.
   → "Weak = identity in the latent" is the default supported by the literature. A lightly perturbed weak latent (the supervisor's guidance) is still tested as an ablation.
3. **Use operators that do not depend on pseudo-labels** (IFMatch). Class-based operators can be pushed by wrong pseudo-labels.
   → Prioritize noise, dropout, and angular rotation. The latent version of TabCutMix (needs pseudo-labels) becomes a baseline only.
4. **Standard latent operators and their default values already exist:**
   - Dropout 0.5 (UniMatch, CCT);
   - multiplicative noise z + z·U(−0.3, 0.3) (CCT F-Noise);
   - F-Drop (CCT);
   - Gaussian noise, interpolation/extrapolation (DeVries, MODALS);
   - latent VAT (Osada).

   → Include **dropout 0.5 and F-Noise** as latent baselines, because both have been tested in the literature.
5. **Interpolation/mixup in the latent does not necessarily help on tabular data** (Onishi: the only one with a negative effect).
   → Low priority. If tested, include extrapolation too (DeVries).
6. **Varying the intensity and form of feature perturbation** (IFMatch) supports the supervisor's ablation of the weak/strong cosine ranges.
7. **The novelty must be bounded:**
   - RaTab: tabular representation augmentation (self-supervised);
   - SDAT: latent augmentation for **semi-supervised** tabular (classification) already exists, so the claim "first latent augmentation in tabular SSL" is also not possible;
   - Kobayashi: feature rotation with a fixed angle;
   - UniMatch/IFMatch: weak-to-strong at the feature level.

   The contribution that is still possible: **angular augmentation with controlled cosine as the weak/strong views in tabular RankUp (ordinal, classification, regression)**, compared with standard latent operators at equal perturbation size. Differences from SDAT: no generative model, perturbation directly in the backbone latent, perturbation size measured by cosine, and tested on ordinal/regression.
8. **Latent perturbation should be small and task-related** (SDAT: best β 0.002; a separate VAE ≈ Gaussian noise). Perturbation in the backbone latent that is trained jointly with RankUp is already "task-related". This supports measuring the actual cos(h, h′) and a strength grid that includes small values.
9. **Perturbation location: the encoder's last layer** (RaTab, layer ablation). This fits our choice (the TabM backbone output).
10. **MODALS operators that are safe without labels:** extrapolation away from the batch mean, difference transform z + λ(zⱼ − zₖ) from two random rows, Gaussian noise with σ from the batch (already present as `gaussian`). The per-class versions are for comparison only.
11. **Latent mixup is already used inside RankUp** (SSR-RCUS, AAAI 2026, on images, text, and audio). The authors replace weak/strong consistency with ranking on mixup pairs in the latent.
   → Mixup between two rows in the latent can be a comparison operator. Unlike item 5, here mixup is used to build ranking pairs, not as the strong view.
12. **Pushing features apart between views can help** (FeatDistLoss). This gives a reason to test (not evidence) that a strong view with low cosine (0.3–0.5) does not always hurt; in FeatDistLoss the augmentation is at the input, not in the latent.
