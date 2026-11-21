"""Daftar varian baseline SSR-RCUS (tanpa torch, supaya bisa dibaca entrypoint Modal lokal).

Kontrol = RankUp resmi di pipeline yang sama.
E1 = SSR-RCUS resmi (weak = strong = data asli; manifold mixup satu-satunya gangguan).
E2 = SSR-RCUS + angular di latent (view weak/strong). E3 = SSR-RCUS + augmentasi input sebelumnya.
Config operator diambil apa adanya dari GRID eksperimen latent, jadi sama persis dengan 945 run sebelumnya.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '5-augmentasi-latent' / 'kode'))
from grid import GRID  # noqa: E402

# Pen Digits (klasifikasi) memakai padanan RankUp-CLS / SSR-RCUS-CLS di tabular_rcus.py; dua lainnya kode resmi.
DATASETS = ('nhanes', 'pendigits', 'superconductivity')
INPUT_OPS = ('gaussian', 'zero_out', 'scarf')  # TabCutMix butuh pseudo-label model, tidak bisa jadi transform input
LATENT_OPS = ('gaussian', 'zero_out', 'scarf', 'dropout', 'fnoise', 'angular')  # yang tidak butuh pseudo-label/head

VARIANTS = {
    'rankup': {'algorithm': 'rankup', 'arc_loss_ratio': 0.2},  # kontrol: RankUp resmi (nilai config UTKFace RankUp)
    'rcus': {},
    'rcus_angular': GRID['angular'],
    'rcus_angular_weak': GRID['angular_weak'],
    **{f'rcus_input_{op}': GRID[f'input_{op}'] for op in INPUT_OPS},
}
