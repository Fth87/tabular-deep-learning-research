"""Compare both exploratory pilots after checking that their input splits match."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent / 'hasil'
folders = [root / 'pilot-local-transport-20261109', root / 'pilot-lanjutan-20261110']
rows = []
for folder in folders:
    for path in folder.rglob('*_metrics.json'):
        record = json.loads(path.read_text())
        if record['test_used'] or record['seed'] != 42:
            raise ValueError('Unexpected test use or seed')
        rows.append({'dataset': record['dataset'], 'label_percent': record['label_percent'],
                     'method': record['method'], 'metric': record['metric'],
                     'validation': record['validation'], 'best_epoch': record['best_epoch'],
                     'epochs_run': record['epochs_run'],
                     'hashes': json.dumps(record['input_sha256'], sort_keys=True)})
frame = pd.DataFrame(rows)
if len(frame) != 108 or frame.duplicated(['dataset', 'label_percent', 'method']).any():
    raise ValueError(f'Expected 108 unique records; got {len(frame)}')
if (frame.groupby(['dataset', 'label_percent']).hashes.nunique() != 1).any():
    raise ValueError('Inputs do not match for a comparison')
utility = frame.validation.where(frame.metric != 'mse', -frame.validation)
frame['rank'] = utility.groupby([frame.dataset, frame.label_percent]).rank(ascending=False, method='min')
frame = frame.drop(columns='hashes').sort_values(['dataset', 'label_percent', 'rank'])
out = root / 'pilot-lanjutan-perbandingan.csv'
frame.to_csv(out, index=False)
for (dataset, percent), group in frame.groupby(['dataset', 'label_percent']):
    print(f'\n{dataset} {percent}% {group.metric.iloc[0]}')
    for _, row in group.iterrows():
        print(f'{int(row["rank"]):2} {row.method:17} {row.validation:.6f} epoch={row.best_epoch}')
print('\nSaved:', out)
