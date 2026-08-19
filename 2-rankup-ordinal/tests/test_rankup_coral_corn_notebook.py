import ast
import json
import re
from pathlib import Path


BASE = Path("kode/rankup_all_datasets.ipynb")
NOTEBOOK = Path("kode/rankup_all_datasets_coral_corn.ipynb")


def source(cell):
    return "".join(cell.get("source", []))


def headings(cells):
    return [source(cell).splitlines()[0] for cell in cells if cell["cell_type"] == "markdown"]


def test_notebook_contract():
    base = json.loads(BASE.read_text())
    notebook = json.loads(NOTEBOOK.read_text())

    assert len(notebook["cells"]) == len(base["cells"]) == 74
    assert headings(notebook["cells"]) == headings(base["cells"])

    all_source = "\n".join(map(source, notebook["cells"]))
    for marker in [
        "coral-pytorch",
        "CoralLayer",
        "coral_loss",
        "corn_loss",
        'OBJECTIVES = ["Regression", "CORAL", "CORN"]',
        "expected_result_rows = 180",
    ]:
        assert marker in all_source

    assert all_source.count('experiment = "Supervised"') == 3
    assert all_source.count('experiment = "Fully-Supervised"') == 3
    assert all_source.count('experiment = "RankUp + FESTA"') == 3
    assert all_source.count('experiment = "RankUp + TabCutMix"') == 3

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "\n".join(
            line
            for line in source(cell).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(code, filename=f"cell {index}")


def test_base_experiment_parity():
    base = json.loads(BASE.read_text())
    notebook = json.loads(NOTEBOOK.read_text())

    assert source(base["cells"][4]) == source(notebook["cells"][4])
    assert source(base["cells"][10]) == source(notebook["cells"][10])

    constants = [
        "N_LABEL", "EPOCHS", "BATCH_SIZE", "SUPERVISED_BATCH_SIZE",
        "MAX_RANKUP_STEPS", "W_RANK", "W_UNLABELED", "P_CUTOFF",
        "FESTA_BINS", "TSNE_MAX_SAMPLES",
    ]
    base_setup = source(base["cells"][2])
    new_setup = source(notebook["cells"][2])
    for name in constants:
        pattern = rf"^{name} = (.+?)\s*$"
        assert re.search(pattern, base_setup, re.MULTILINE).group(1).strip() == re.search(
            pattern, new_setup, re.MULTILINE
        ).group(1).strip()

    experiment_cells = [21, 23, 25, 27, 36, 38, 40, 42, 51, 53, 55, 57]
    for index in experiment_cells:
        cell_source = source(notebook["cells"][index])
        assert "for objective in OBJECTIVES:" in cell_source
        assert '"objective": objective' in cell_source


if __name__ == "__main__":
    test_notebook_contract()
    test_base_experiment_parity()
    print("PASS: notebook contract and base parity")
