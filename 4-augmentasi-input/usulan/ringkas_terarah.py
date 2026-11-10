"""Compare feature-guided RankUp with conditional augmentation pilots."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent / 'hasil'
old = root / 'conditional-screen-20261110'
new = root / 'feature-guided-20261110'
prior = pd.read_csv(root / 'conditional-screen-perbandingan.csv')
rows = []
for path in new.rglob('feature_guided_metrics.json'):
    record = json.loads(path.read_text())
    dataset, percent = record['dataset'], record['label_percent']
    baseline = json.loads((old / f'label_{percent:02d}' / dataset / 'conditional_random_metrics.json').read_text())
    if record['input_sha256'] != baseline['input_sha256'] or record['test_used'] or record['seed'] != 42:
        raise ValueError(f'Incompatible input or seed: {path}')
    rows.append({'dataset': dataset, 'percent': percent, 'method': 'feature_guided',
                 'score': record['validation'], 'epoch': record['best_epoch'], 'source': 'guided'})
if len(rows) != 9:
    raise ValueError(f'Expected 9 records, got {len(rows)}')
frame = pd.concat([prior.drop(columns='rank'), pd.DataFrame(rows)], ignore_index=True)
utility = frame.score.where(frame.dataset != 'superconductivity', -frame.score)
frame['rank'] = utility.groupby([frame.dataset, frame.percent]).rank(ascending=False, method='min')
frame = frame.sort_values(['dataset', 'percent', 'rank'])
output = root / 'feature-guided-perbandingan.csv'
frame.to_csv(output, index=False)
for (dataset, pct), group in frame.groupby(['dataset', 'percent']):
    best = group.iloc[0]
    print(f'\n{dataset} {pct}%: best {best.method} {best.score:.6f}')
    for _, row in group[group.method.isin(['feature_guided', 'conditional_random', 'conditional_group'])].iterrows():
        print(f"  {row.method:19} {row.score:.6f} rank={int(row['rank'])} epoch={row.epoch}")
print('\nSaved:', output)
