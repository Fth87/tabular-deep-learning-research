"""Semua konfigurasi eksperimen. Nama = posisi augmentasi + operator (+ varian).

Kunci config (semua opsional):
  input_op, input_weak, input_strong      augmentasi di input (operator lama, strength sama dengan E2)
  latent_op, latent_weak, latent_strong   augmentasi di output backbone TabM
  unlabeled_weight                        0 = tanpa RankUp (supervised saja)
Strength 0 atau tidak ada = identitas. Untuk 'angular', strength = (cos_min, cos_max).
"""
OPS = ('gaussian', 'zero_out', 'scarf', 'tabcutmix')  # operator yang ada versi input dan latent-nya
STRONG = 0.5   # sama dengan E2 (def_*)
WEAK = 0.1     # weak latent ringan, sesuai arahan pembimbing

GRID = {'supervised': {'unlabeled_weight': 0.0},
        'rankup_noaug': {}}  # kontrol: RankUp tanpa augmentasi apa pun (weak = strong = data asli)
for op in OPS:
    GRID[f'input_{op}'] = {'input_op': op, 'input_strong': STRONG}
    GRID[f'latent_{op}'] = {'latent_op': op, 'latent_strong': STRONG}
    GRID[f'latent_{op}_weak'] = {'latent_op': op, 'latent_weak': WEAK, 'latent_strong': STRONG}
    GRID[f'both_{op}'] = {'input_op': op, 'input_strong': STRONG, 'latent_op': op, 'latent_strong': STRONG}
# Baseline latent dari literatur, dengan nilai default paper-nya.
GRID['latent_dropout'] = {'latent_op': 'dropout', 'latent_strong': 0.5}   # UniMatch, CCT
GRID['latent_fnoise'] = {'latent_op': 'fnoise', 'latent_strong': 0.3}     # CCT
# Ide pembimbing: cosine terkontrol. 'angular' dan 'angular_weak' = contoh rentang dari rapat 13 Nov (Fase 2a);
# sisanya ablasi rentang (Fase 2b). Nama: angular[_weak|_weak90][_s10|_s50].
ANGULAR_WEAK = {'': None, '_weak': (0.7, 0.95), '_weak90': (0.9, 0.99)}
ANGULAR_STRONG = {'_s10': (0.1, 0.3), '': (0.3, 0.5), '_s50': (0.5, 0.7)}
for w, weak in ANGULAR_WEAK.items():
    for s, strong in ANGULAR_STRONG.items():
        GRID[f'angular{w}{s}'] = {'latent_op': 'angular', 'latent_strong': strong,
                                  **({'latent_weak': weak} if weak else {})}
# Fase 2e: rotasi sebelum ReLU blok terakhir (hasil tetap ≥ 0 seperti yang biasa dilihat head).
GRID['angular_prerelu'] = {**GRID['angular'], 'latent_at': 'pre_relu'}
GRID['angular_weak_prerelu'] = {**GRID['angular_weak'], 'latent_at': 'pre_relu'}


def matched_strength(op: str, cos: float) -> float:
    """Strength operator lama yang memberi cos(h, h') ≈ cos. Rumus dikalibrasi dari cos aktual Fase 1 (strength 0,5).

    zero_out/dropout: cos = √(1−p) (tepat, 0,705 di p = 0,5).
    gaussian: cos ≈ 1/√(1 + s²·r), r = 0,938 dari cos 0,90 di s = 0,5.
    scarf: cos ≈ 1 − p(1 − ρ), ρ = 0,25 dari cos 0,625 di p = 0,5.
    TabCutMix (cos minimum ±0,46) dan F-Noise (±0,87) tidak bisa mencapai cos rendah, jadi tidak dipakai.
    Cos aktual tetap dicatat per run dan dilaporkan di ringkasan.
    """
    strength = {'zero_out': 1 - cos ** 2, 'dropout': 1 - cos ** 2,
                'gaussian': ((1 / cos ** 2 - 1) / 0.938) ** 0.5, 'scarf': (1 - cos) / 0.75}[op]
    assert op == 'gaussian' or 0 < strength < 1, (op, cos, strength)
    return round(strength, 3)


# Fase 2c: operator lama dengan cos aktual sama dengan angular (strong cos 0,4, weak identitas). Kalau sama baiknya,
# yang berpengaruh adalah besar gangguan, bukan konstruksi trigonometrinya.
MATCHED_TO, MATCHED_COS = 'angular', 0.4
for op in ('zero_out', 'dropout', 'gaussian', 'scarf'):
    GRID[f'matched_{op}'] = {'latent_op': op, 'latent_strong': matched_strength(op, MATCHED_COS)}

# Fase 3: rotasi sadar-head (dokumen/USULAN_ROTASI_ADVERSARIAL.md). Strength = rentang angular distance
# d ∈ [0, 1], θ = π·d, pusat 0,05 / 0,10 / 0,20. advrot = arah adversarial closed form; headrot = tanda acak (kontrol).
ROT_D = {'05': (0.025, 0.075), '10': (0.05, 0.15), '20': (0.10, 0.30)}
for tag, d in ROT_D.items():
    GRID[f'advrot_{tag}'] = {'latent_op': 'advrot', 'latent_strong': d}
for tag in ('10', '20'):
    GRID[f'headrot_{tag}'] = {'latent_op': 'headrot', 'latent_strong': ROT_D[tag]}

GROUPS = {
    'fase1': [name for name in GRID if not name.startswith(('angular', 'matched_', 'advrot', 'headrot'))],
    'fase2': ['angular', 'angular_weak'],
    'fase2b': [name for name in GRID if name.startswith('angular') and name not in ('angular', 'angular_weak')],
    'fase2c': [name for name in GRID if name.startswith('matched_')],
    'fase3': [name for name in GRID if name.startswith(('advrot', 'headrot'))],
}
DATASETS = ('nhanes', 'pendigits', 'superconductivity')
PERCENTS = (1, 10, 30)
SEEDS = (42, 123, 456)
