"""Combine conditional pilot and existing screening using validation only."""
import hashlib
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
old_root = root / 'eksperimen/hasil/data-5seed'
new_root = root / 'usulan/hasil/conditional-screen-20261110'
old_file = root / 'eksperimen/hasil/usulan-screening/results.jsonl'
metric = {'nhanes': 'qwk', 'pendigits': 'macro_f1', 'superconductivity': 'mse'}
rows = []
for line in old_file.read_text().splitlines():
    record = json.loads(line)
    if record['seed'] != 42:
        raise ValueError('Unexpected old seed')
    rows.append(dict(dataset=record['dataset'], percent=record['label_percent'], method=record['name'],
                     score=record['validation'][metric[record['dataset']]], epoch=record['best_epoch'], source='existing'))
for path in new_root.rglob('*_metrics.json'):
    record = json.loads(path.read_text())
    if record['seed'] != 42 or record['test_used']:
        raise ValueError('Unexpected new record')
    old_data = old_root / f"label_{record['label_percent']:02d}" / record['dataset']
    for name, digest in record['input_sha256'].items():
        if hashlib.sha256((old_data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input mismatch for {path}: {name}')
    rows.append(dict(dataset=record['dataset'], percent=record['label_percent'], method=record['method'],
                     score=record['validation'], epoch=record['best_epoch'], source='conditional'))
frame = pd.DataFrame(rows)
if len(frame) != 87 or frame.duplicated(['dataset', 'percent', 'method']).any():
    raise ValueError(f'Expected 87 unique jobs, got {len(frame)}')
utility = frame.score.where(frame.dataset != 'superconductivity', -frame.score)
frame['rank'] = utility.groupby([frame.dataset, frame.percent]).rank(ascending=False, method='min')
frame = frame.sort_values(['dataset', 'percent', 'rank'])
out = root / 'usulan/hasil/conditional-screen-perbandingan.csv'
frame.to_csv(out, index=False)
for (dataset, percent), group in frame.groupby(['dataset', 'percent']):
    print(f'\n{dataset} {percent}% ({metric[dataset]})')
    for _, row in group.iterrows():
        print(f'{int(row["rank"]):2} {row.method:23} {row.score:.6f} epoch={row.epoch}')
print('\nSaved:', out)
