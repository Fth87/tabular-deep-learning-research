# RankUp CORAL/CORN Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membuat notebook Kaggle baru yang mempertahankan struktur dan eksperimen `rankup_all_datasets.ipynb`, lalu menjalankan objective Regression, CORAL, dan CORN pada MLP, TabM, dan RealMLP.

**Architecture:** Notebook baru dibangun dari 74 cell notebook acuan agar urutan dan pemisahan eksperimennya tidak berubah. Cell model, target, loss, inference, loop objective, penyimpanan hasil, dan visualisasi diganti secara mekanis; ranking head, FESTA, TabCutMix, split, optimizer, dan metrik tetap mengikuti acuan.

**Tech Stack:** Python, Jupyter JSON, PyTorch, `tabm`, `coral-pytorch`, pandas, seaborn, scikit-learn.

## Global Constraints

- Hasil akhir: `kode/rankup_all_datasets_coral_corn.ipynb`.
- Satu seed `42`, lima dataset, tiga backbone, tiga objective, empat skenario.
- Total hasil yang wajib terbentuk: 180 baris.
- Notebook harus dapat dijalankan melalui Kaggle Save & Run All.
- CORAL/CORN memakai target integer `0..K-1`; Regression mempertahankan target scaler acuan.
- Loss TabM dihitung per ensemble member; inference merata-ratakan probabilitas member.
- Metrik tetap R², Spearman, MAE, RMSE, dan QWK.

---

### Task 1: Kontrak notebook

**Files:**
- Create: `tests/test_rankup_coral_corn_notebook.py`
- Target: `kode/rankup_all_datasets_coral_corn.ipynb`

**Interfaces:**
- Consumes: notebook JSON dengan key `cells`.
- Produces: pemeriksaan struktur, sintaks, marker objective, dan jumlah kombinasi.

- [ ] **Step 1: Tulis pemeriksaan yang gagal sebelum notebook dibuat**

```python
import ast
import json
from pathlib import Path

NOTEBOOK = Path("kode/rankup_all_datasets_coral_corn.ipynb")


def source(cell):
    return "".join(cell.get("source", []))


def test_notebook_contract():
    notebook = json.loads(NOTEBOOK.read_text())
    assert len(notebook["cells"]) == 74
    all_source = "\n".join(map(source, notebook["cells"]))
    for marker in [
        "coral-pytorch", "CoralLayer", "coral_loss", "corn_loss",
        'OBJECTIVES = ["Regression", "CORAL", "CORN"]',
        "expected_result_rows = 180",
    ]:
        assert marker in all_source
    assert all_source.count("RankUp + FESTA") >= 3
    assert all_source.count("RankUp + TabCutMix") >= 3
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            code = "\n".join(
                line for line in source(cell).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            ast.parse(code, filename=f"cell {index}")
```

- [ ] **Step 2: Jalankan pemeriksaan dan pastikan RED**

Run: `pytest -q tests/test_rankup_coral_corn_notebook.py`

Expected: FAIL karena `kode/rankup_all_datasets_coral_corn.ipynb` belum ada.

### Task 2: Generator dan notebook ordinal

**Files:**
- Create: `scripts/build_rankup_coral_corn_notebook.py`
- Create: `kode/rankup_all_datasets_coral_corn.ipynb`

**Interfaces:**
- Consumes: `kode/rankup_all_datasets.ipynb`.
- Produces: notebook 74 cell tanpa output lama.
- Model: `Model(n_features, n_classes, objective)` mengembalikan `(prediction_logits, rank_logits)`.
- Loss: `prediction_loss(logits, y_scaled, y_class, objective, num_classes)`.
- Inference: `predict(model, data, objective)` mengembalikan skor kontinu skala kelas asli.

- [ ] **Step 1: Buat generator minimal**

Generator membaca notebook acuan dengan `json`, mengganti source cell berdasarkan indeks, menghapus output dan execution count, lalu menulis notebook baru. Ia mempertahankan seluruh markdown heading dan empat cell eksperimen per backbone.

Cell setup harus memasang dan mengimpor:

```python
%pip install -q tabm coral-pytorch
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
OBJECTIVES = ["Regression", "CORAL", "CORN"]
```

Cell preprocessing harus menyimpan target berikut:

```python
"y_labeled_scaled": tensor(y_scaled[labeled_idx]),
"y_labeled_class": tensor(y[labeled_idx].astype(np.int64)),
"y_train_scaled": tensor(y_scaled[train_idx]),
"y_train_class": tensor(y[train_idx].astype(np.int64)),
"num_classes": len(np.unique(y)),
```

MLP dan RealMLP memakai `CoralLayer` untuk CORAL, output `K-1` untuk CORN, dan output satu nilai untuk Regression. TabM memakai output scalar + `CoralLayer(1, K)` untuk CORAL, serta output langsung `K-1` untuk CORN. Ranking head selalu dua logits.

Loss objective harus menangani output biasa `(B, D)` dan ensemble `(B, k, D)`:

```python
def prediction_loss(logits, y_scaled, y_class, objective, num_classes):
    if objective == "Regression":
        return regression_loss(logits, y_scaled)
    if logits.ndim == 3:
        k = logits.shape[1]
        logits = logits.flatten(0, 1)
        y_class = y_class.repeat_interleave(k)
    if objective == "CORAL":
        levels = levels_from_labelbatch(y_class, num_classes).to(logits.device)
        return coral_loss(logits, levels)
    return corn_loss(logits, y_class, num_classes)
```

Inference ordinal harus memakai cumulative probabilities:

```python
def ordinal_probabilities(logits, objective):
    probability = torch.sigmoid(logits)
    return probability if objective == "CORAL" else probability.cumprod(dim=-1)


def ordinal_score(logits, objective):
    probability = ordinal_probabilities(logits, objective)
    if probability.ndim == 3:
        probability = probability.mean(dim=1)
    return probability.sum(dim=-1)
```

Setiap cell skenario menjalankan `for objective in OBJECTIVES`, mencatat `backbone`, `objective`, dan `experiment`, serta menyimpan key representasi `(dataset, backbone, objective, experiment)`.

- [ ] **Step 2: Tambahkan self-check notebook**

```python
demo_logits = torch.tensor([[2.0, 1.0, -1.0]])
assert ordinal_probabilities(demo_logits, "CORN").shape == demo_logits.shape
assert torch.all(ordinal_probabilities(demo_logits, "CORN")[:, 1:] <= ordinal_probabilities(demo_logits, "CORN")[:, :-1])
assert 0 <= ordinal_score(demo_logits, "CORAL").item() <= demo_logits.shape[1]
```

- [ ] **Step 3: Bangun notebook**

Run: `python scripts/build_rankup_coral_corn_notebook.py`

Expected: mencetak path output dan `74 cells`.

- [ ] **Step 4: Jalankan kontrak dan pastikan GREEN**

Run: `pytest -q tests/test_rankup_coral_corn_notebook.py`

Expected: `1 passed`.

### Task 3: Audit kesetaraan dan kesiapan Kaggle

**Files:**
- Verify: `kode/rankup_all_datasets.ipynb`
- Verify: `kode/rankup_all_datasets_coral_corn.ipynb`

**Interfaces:**
- Consumes: kedua notebook.
- Produces: bukti struktur heading, skenario, dataset, dan sintaks tetap konsisten.

- [ ] **Step 1: Bandingkan heading dan konfigurasi inti**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text())["cells"]

base = load("kode/rankup_all_datasets.ipynb")
new = load("kode/rankup_all_datasets_coral_corn.ipynb")
assert len(base) == len(new) == 74
base_headings = ["".join(c["source"]).splitlines()[0] for c in base if c["cell_type"] == "markdown"]
new_headings = ["".join(c["source"]).splitlines()[0] for c in new if c["cell_type"] == "markdown"]
assert base_headings == new_headings
print("PASS: 74 cell dan heading sama")
PY
```

Expected: `PASS: 74 cell dan heading sama`.

- [ ] **Step 2: Verifikasi semua output plot memiliki dimensi objective**

Run: `rg -n 'row=objective|OBJECTIVES|hue="objective"|objective_order=OBJECTIVES' kode/rankup_all_datasets_coral_corn.ipynb`

Expected: marker objective muncul pada helper dan perbandingan akhir.

- [ ] **Step 3: Jalankan seluruh pemeriksaan akhir**

Run: `pytest -q tests/test_rankup_coral_corn_notebook.py && python scripts/build_rankup_coral_corn_notebook.py && pytest -q tests/test_rankup_coral_corn_notebook.py`

Expected: dua kali `1 passed` dan generator melaporkan `74 cells`.

- [ ] **Step 4: Commit implementasi**

```bash
git add scripts/build_rankup_coral_corn_notebook.py tests/test_rankup_coral_corn_notebook.py kode/rankup_all_datasets_coral_corn.ipynb
git commit -m "feat: add CORAL and CORN RankUp notebook"
```

