"""Ringkasan hasil/results.jsonl -> hasil/ringkasan.md. Jalankan: python ringkas.py (butuh pandas, tabulate).

Metrik: NHANES = QWK (↑), Pen Digits = Macro F1 (↑), Superconductivity = MSE (↓).
Angka test hanya dilaporkan; tidak ada pemilihan varian dari test.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
METRIC = {'nhanes': 'qwk', 'pendigits': 'macro_f1', 'superconductivity': 'mse'}
ORDER = ['rankup', 'rcus', 'rcus_angular', 'rcus_angular_weak', 'rcus_input_gaussian', 'rcus_input_zero_out',
         'rcus_input_scarf']

rows = [json.loads(line) for line in (HERE / 'hasil' / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
df = pd.DataFrame([{'name': r['name'], 'dataset': r['dataset'], 'label': r['label_percent'], 'seed': r['seed'],
                    'val': r['validation'][METRIC[r['dataset']]], 'test': r['test'][METRIC[r['dataset']]]}
                   for r in rows])
# skor "makin besar makin baik": MSE dibalik tandanya hanya untuk uji berpasangan
df['val_good'] = df.val.where(df.dataset != 'superconductivity', -df.val)
df['test_good'] = df.test.where(df.dataset != 'superconductivity', -df.test)

out = ['# Ringkasan SSR-RCUS tabular', '', f'Dibuat otomatis oleh `ringkas.py` dari `results.jsonl` ({len(df)} run).', '',
       '## 1. Skor test (rata-rata ± std antar 3 seed)', '']
for dataset, metric in METRIC.items():
    part = df[df.dataset == dataset]
    table = part.groupby(['name', 'label']).test.agg(lambda s: f'{s.mean():.3f} ± {s.std():.3f}').unstack()
    out += [f'**{dataset}** ({metric})', '', table.loc[ORDER].to_markdown(), '']

wide = df.pivot_table(index=['dataset', 'label', 'seed'], columns='name', values=['val_good', 'test_good'])
out += ['## 2. Perbandingan berpasangan: berapa blok yang dimenangkan', '',
        'Satu blok = satu (label, seed), jadi 9 blok per dataset. "Menang" = skor varian lebih baik dari pembanding. '
        'Hanya hitungan menang yang dilaporkan: selisih skor tidak dirata-ratakan lintas dataset karena skalanya berbeda '
        '(MSE vs QWK), dan dengan 9 blok per dataset belum ada uji yang berarti.', '',
        '| varian | vs | split | nhanes | pendigits | superconductivity | total |', '|---|---|---|---:|---:|---:|---:|']
for variant, base in [('rcus', 'rankup')] + [(v, 'rcus') for v in ORDER[2:]]:
    for split in ('test_good', 'val_good'):
        win = (wide[split][variant] - wide[split][base]) > 0
        per = win.groupby(level='dataset').sum()
        out.append(f'| {variant} | {base} | {split[:-5]} | ' + ' | '.join(f'{per[d]}/9' for d in METRIC)
                   + f' | {int(win.sum())}/27 |')
out.append('')
(HERE / 'hasil' / 'ringkasan.md').write_text('\n'.join(out), encoding='utf-8')
print('\n'.join(out))
