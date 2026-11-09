"""Inspect validation ranks and acceptance rates of the existing 270 runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
results = root / 'eksperimen/hasil/ta-5seed-40trial-20261105'
summary = pd.read_csv(root / 'eksperimen/hasil/ringkasan-5seed-40trial.csv')
summary['selection_score'] = np.where(summary.dataset == 'superconductivity',
                                      -summary.validation_mean, summary.validation_mean)
summary['validation_rank'] = summary.groupby(['dataset', 'label_percent']).selection_score.rank(ascending=False)
rows = []
for p in results.rglob('*_history.csv'):
    rel = p.relative_to(results).parts
    h = pd.read_csv(p)
    metrics_path = p.with_name(p.name.replace('_history.csv', '_metrics.json'))
    metrics = json.loads(metrics_path.read_text())
    best_epoch = int(metrics['best_epoch'])
    rows.append({'dataset': rel[2], 'label_percent': int(rel[0].split('_')[1]),
                 'mean_accept_rate': h.pseudo_accept_rate.mean(),
                 'best_epoch_accept_rate': h.pseudo_accept_rate.iloc[best_epoch - 1],
                 'best_epoch': best_epoch})
hist = pd.DataFrame(rows)
print('runs:',len(hist),'combinations:',len(summary))
print('\nMean validation rank (lower is better):')
print(summary.groupby('method').validation_rank.mean().sort_values().round(3).to_string())
print('\nAverage pseudo-label acceptance in the selected epoch:')
print(hist.groupby(['dataset','label_percent']).best_epoch_accept_rate.agg(['mean','min','max']).round(4).to_string())
