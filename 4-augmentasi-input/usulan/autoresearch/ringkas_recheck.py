"""Summarize paired seed-43 checks after all Modal jobs have completed."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'hasil'
run = root / 'autoresearch-recheck-seed43-20261111' / 'label_01'
rows = []
for dataset in ('nhanes', 'pendigits', 'superconductivity'):
    baseline = json.loads((run / dataset / 'baseline_metrics.json').read_text())
    candidate = json.loads((run / dataset / 'candidate_metrics.json').read_text())
    if (baseline['seed'] != 43 or candidate['seed'] != 43 or baseline['test_used'] or candidate['test_used']
        or baseline['input_sha256'] != candidate['input_sha256'] or baseline['candidate_sha256'] != candidate['candidate_sha256']):
        raise ValueError(f'Incompatible pair for {dataset}')
    change = baseline['validation'] - candidate['validation'] if candidate['metric'] == 'mse' else candidate['validation'] - baseline['validation']
    rows.append({'dataset': dataset, 'label_percent': 1, 'seed': 43, 'metric': candidate['metric'],
                 'baseline': baseline['validation'], 'candidate': candidate['validation'], 'improvement': change})
frame = pd.DataFrame(rows)
out = root / 'autoresearch' / 'recheck-seed43-1pct.csv'
frame.to_csv(out, index=False)
print(frame.to_string(index=False))
print('Saved:', out)
