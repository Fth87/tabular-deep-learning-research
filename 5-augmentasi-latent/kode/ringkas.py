"""Ringkas ../hasil/results.jsonl menjadi ../hasil/ringkasan.md.

Jalankan dari folder kode/:  python ringkas.py
Isi: (1) skor test per kondisi, (2) peringkat rata-rata (validation dan test),
(3) perbandingan posisi augmentasi per operator (uji Wilcoxon berpasangan), (4) cosine aktual.
Pemilihan konfigurasi memakai validation; test hanya dilaporkan.
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from grid import MATCHED_TO, OPS

HASIL = Path(__file__).resolve().parent.parent / 'hasil'
METRIC = {'nhanes': ('qwk', 1), 'pendigits': ('macro_f1', 1), 'superconductivity': ('mse', -1)}  # 1 = makin besar makin baik


def load() -> pd.DataFrame:
    rows = []
    for line in (HASIL / 'results.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        metric, sign = METRIC[r['dataset']]
        rows.append({'name': r['name'], 'dataset': r['dataset'], 'label': r['label_percent'], 'seed': r['seed'],
                     'val': r['validation'][metric], 'test': r['test'][metric], 'sign': sign,
                     'cos_weak': r['cos'].get('weak'), 'cos_strong': r['cos'].get('strong')})
    df = pd.DataFrame(rows).drop_duplicates(['name', 'dataset', 'label', 'seed'], keep='last')
    for split in ('val', 'test'):
        df[f'{split}_good'] = df[split] * df['sign']  # skor "makin besar makin baik" untuk peringkat dan uji
    return df


def score_table(df: pd.DataFrame) -> str:
    stats = df.groupby(['name', 'dataset', 'label'])['test'].agg(['mean', 'std', 'count']).reset_index()
    stats['cell'] = stats.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if r['count'] > 1
                                else f"{r['mean']:.3f}", axis=1)
    table = stats.pivot(index='name', columns=['dataset', 'label'], values='cell')
    table.columns = [f'{d} {p}%' for d, p in table.columns]
    return table.to_markdown()


def rank_table(df: pd.DataFrame) -> str:
    keys = ['dataset', 'label', 'seed']
    complete = df.groupby(keys)['name'].transform('count') == df['name'].nunique()  # hanya blok yang lengkap
    block = df[complete].copy()
    for split in ('val', 'test'):
        block[f'rank_{split}'] = block.groupby(keys)[f'{split}_good'].rank(ascending=False)
    ranks = block.groupby('name')[['rank_val', 'rank_test']].mean().sort_values('rank_val').round(2)
    blocks = block.groupby(keys).ngroups
    return f'Blok lengkap: {blocks} (dataset × label × seed). Urut menurut validation.\n\n' + ranks.to_markdown()


def position_table(df: pd.DataFrame) -> str:
    keys = ['dataset', 'label', 'seed']
    rows = []
    pairs = [(op, a, b) for op in OPS for a, b in (
        (f'input_{op}', f'latent_{op}'), (f'input_{op}', f'both_{op}'), (f'latent_{op}', f'latent_{op}_weak'),
        ('rankup_noaug', f'input_{op}'), ('rankup_noaug', f'latent_{op}'))]
    # Kontrol: apakah augmentasi (di mana pun) lebih baik dari RankUp tanpa augmentasi / tanpa RankUp.
    names = sorted(df['name'].unique())
    pairs += [('-', 'rankup_noaug', name) for name in names
              if name in ('latent_dropout', 'latent_fnoise') or name.startswith(('angular', 'matched_'))]
    pairs += [('angular', name.replace('_prerelu', ''), name) for name in names if name.endswith('_prerelu')]
    pairs += [('matched', MATCHED_TO, name) for name in names if name.startswith('matched_')]
    pairs += [('-', 'supervised', 'rankup_noaug')]
    for op, a, b in pairs:
        pair = df[df['name'] == a].merge(df[df['name'] == b], on=keys, suffixes=('_a', '_b'))
        if len(pair) < 2:
            continue
        for split in ('val', 'test'):
            diff = pair[f'{split}_good_b'] - pair[f'{split}_good_a']
            p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            rows.append({'operator': op, 'pembanding': f'{b} vs {a}', 'split': split, 'n': len(pair),
                         'B menang': f'{(diff > 0).mean():.0%}', 'p Wilcoxon': round(float(p), 4)})
    if not rows:
        return '(belum ada pasangan lengkap)'
    tests = len(rows) // 2  # satu uji per pembanding per split
    return (f'Ada {tests} pembanding per split; anggap p < {0.05 / tests:.4f} (Bonferroni) sebagai bukti kuat.\n\n'
            + pd.DataFrame(rows).to_markdown(index=False))


def cosine_table(df: pd.DataFrame) -> str:
    latent = df[df['cos_strong'].notna()]
    cos = latent.groupby('name')[['cos_weak', 'cos_strong']].mean().round(3).sort_values('cos_strong')
    return cos.to_markdown()


def main():
    df = load()
    text = f"""# Ringkasan eksperimen latent

Dibuat otomatis oleh `kode/ringkas.py` dari `results.jsonl` ({len(df)} run). Jangan diedit manual.

Metrik: NHANES = QWK (↑), Pen Digits = Macro F1 (↑), Superconductivity = MSE (↓).
Konfigurasi dipilih dari **validation**; angka test hanya dilaporkan.

## 1. Skor test (rata-rata ± std antar seed)

{score_table(df)}

## 2. Peringkat rata-rata (1 = terbaik)

{rank_table(df)}

## 3. Perbandingan berpasangan

Pasangan per (dataset, label, seed). "B menang" = proporsi pasangan saat konfigurasi pertama (B) lebih baik dari yang kedua.

{position_table(df)}

## 4. Cosine aktual cos(h, h′) di latent

Rata-rata selama training; kolom kosong = view itu tidak diganggu di latent.

{cosine_table(df)}
"""
    (HASIL / 'ringkasan.md').write_text(text, encoding='utf-8')
    print(f'ditulis: {HASIL / "ringkasan.md"} ({len(df)} run)')


if __name__ == '__main__':
    main()
