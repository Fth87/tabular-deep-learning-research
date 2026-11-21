# Figures

The diagrams are made with LaTeX (TikZ), so they stay sharp in slides and documents. Each figure has three files with the same name:
`.tex` (source), `.pdf` (vector, for documents), `.png` (300 dpi, for slides).

To rebuild after editing a `.tex` file (needs `pdflatex` and `pdftoppm`):

```bash
./bangun.sh                # all figures
./bangun.sh cosine_skema   # one figure
```

## `arsitektur/` — RankUp flow with augmentation in the latent space

| Figure | Contents |
|---|---|
| `arsitektur` | Short: data → encoder → weak/strong → head → loss. For slides |
| `arsitektur_medium` | The same, with a note in each box |
| `arsitektur_detail` | Full: training flow, TabM internals, and angular augmentation. For papers |

## `cosine/` — the supervisor's cosine method (V, V1, V2)

Suggested reading order:

| Figure | Contents |
|---|---|
| `cosine_dasar` | What cosine similarity `c` is, with a 2D example |
| `cosine_hitung` | How to compute V1 from V and `c`, step by step |
| `cosine_skema` | Scheme V → weak V1 / strong V2 → Head, without numbers |
| `cosine_latent` | The same scheme with a numerical example |
| `cosine_dosen` | Full version following the supervisor's whiteboard: flow, steps, and geometry |
| `cosine_langkah` | Four panels (cosine, 2D example, cone in 512 dimensions, recipe). The cone part appears only here |

## `paper/` — methods from reference papers

| Figure | Contents |
|---|---|
| `ssr_rcus` | The SSR-RCUS method (Li et al., AAAI 2026), in the same style as `arsitektur_medium` for comparison |

File names are in Indonesian: `arsitektur` = architecture, `dasar` = basics, `hitung` = compute, `skema` = scheme, `dosen` = supervisor, `langkah` = steps, `bangun` = build.
