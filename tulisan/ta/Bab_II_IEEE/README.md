# Full IEEE template with Chapter II

Use **`template_lengkap.tex`** as the main file and `template_lengkap.pdf` as its compiled output. This is a copy of the full IEEE conference template: title, six sample authors, abstract, keywords, introduction, acknowledgment, sample references, and the template's guidance text are all kept. Section II of the template is replaced with the research's Chapter II from `bab_ii_isi.tex`.

The Chapter II references are added to the bibliography through `referensi_bab_ii_items.tex`; the template's 11 sample references are kept. The sample citation numbers shift automatically because the Chapter II references appear first. A few LaTeX packages were added for tables and URLs. `IEEEtran.cls` and `template_asli/` come from the original template that was copied; the source Markdown is unchanged.

Compile from this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error template_lengkap.tex
```

`bab_ii.tex` and `bab_ii.pdf` are an older version that shows only an excerpt of Chapter II. They share `bab_ii_isi.tex` and `referensi_bab_ii_items.tex` with the full template.

The title, author names, affiliations, abstract, introduction, acknowledgment, sample references, and the red instruction text are the **template's default text**, kept on purpose as requested. Replace or delete all sample text before submitting a paper; Chapter II alone is not yet a complete manuscript. Also follow the language and page-limit rules of the target conference.

The author of the Pen Digits dataset reference in the LaTeX copy was corrected from I. Alpaydin to E. Alpaydin, and the year 1996 was added, following the [UCI metadata](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits). The other references were transcribed from their sources; not all of their metadata and claims have been verified. Source notes are in `catatan_verifikasi.md`.

The latest revision tidied four tables into a single column and tightened the Chapter II text. Details of the changes and their limitations are in `CATATAN_REVISI.md`; the pre-revision version is in `arsip_sebelum_revisi/`.
