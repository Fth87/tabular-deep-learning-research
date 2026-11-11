"""Compare a completed candidate to the fixed validation-only baseline."""
import argparse
import json
import math
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'hasil'
p = argparse.ArgumentParser()
p.add_argument('run_id')
a = p.parse_args()
new = root / a.run_id
baseline = root / 'feature-guided-20261110'
rows, hashes = [], set()
for path in new.rglob('candidate_metrics.json'):
    r = json.loads(path.read_text())
    old = baseline / f"label_{r['label_percent']:02d}" / r['dataset'] / 'feature_guided_metrics.json'
    b = json.loads(old.read_text())
    if r['input_sha256'] != b['input_sha256'] or r['test_used'] or r['seed'] != b['seed']:
        raise ValueError(f'Incompatible input/seed or test use: {path}')
    if r['pilot_config']['epochs'] != 120 or r['pilot_config']['max_steps'] != 100:
        raise ValueError('Incorrect experiment budget')
    hashes.add(r['candidate_sha256'])
    change = (b['validation'] - r['validation']) if r['metric'] == 'mse' else (r['validation'] - b['validation'])
    if not math.isfinite(change):
        raise ValueError('Nonfinite validation score')
    rows.append({'dataset': r['dataset'], 'label_percent': r['label_percent'], 'metric': r['metric'],
                 'baseline': b['validation'], 'candidate': r['validation'], 'improvement': change,
                 'best_epoch': r['best_epoch']})
if len(rows) != 9 or len(hashes) != 1:
    raise ValueError(f'Expected 9 comparable results from one candidate, got {len(rows)}, {len(hashes)} versions')
frame = pd.DataFrame(rows).sort_values(['dataset', 'label_percent'])
keep = bool((frame.improvement >= 0).all() and (frame.improvement > 0).any())
output = root / 'autoresearch'
output.mkdir(exist_ok=True)
frame.to_csv(output / f'{a.run_id}-comparison.csv', index=False)
summary = {'run_id': a.run_id, 'candidate_sha256': next(iter(hashes)), 'conditions': len(rows),
           'wins': int((frame.improvement > 0).sum()), 'ties': int((frame.improvement == 0).sum()),
           'losses': int((frame.improvement < 0).sum()), 'decision': 'keep' if keep else 'discard',
           'baseline': 'feature-guided-20261110', 'test_used': False}
(output / f'{a.run_id}-decision.json').write_text(json.dumps(summary, indent=2) + '\n')
print(frame.to_string(index=False))
print(json.dumps(summary, indent=2))
