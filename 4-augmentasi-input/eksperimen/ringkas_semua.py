"""Summarise the all-dataset run (1 seed, no Optuna) as markdown tables.

Usage: uv run python eksperimen/ringkas_semua.py eksperimen/hasil/usulan-semua/results.jsonl [test|validation] [ours] [names]
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

OURS = 'gibbs_k5_s50'
METRICS = {'ordinal_regression': {'qwk': 1, 'mae': -1, 'accuracy': 1, 'spearman': 1},
           'classification': {'macro_f1': 1, 'accuracy': 1},
           'regression': {'mse': -1, 'mae': -1, 'r2': 1, 'spearman': 1}}
PRIMARY = {'ordinal_regression': 'qwk', 'classification': 'macro_f1', 'regression': 'mse'}
LABEL = {'none': 'Tanpa SSL', OURS: '**Block-Gibbs (usulan)**', 'def_tabcutmix': 'TabCutMix',
         'def_scarf': 'SCARF', 'def_range_limited': 'RLA', 'def_gaussian': 'Gaussian',
         'def_zero_out': 'Zero-out', 'def_tabcutmix_plus': 'TabCutMix+'}


def md(frame, index=True):
    frame = frame.reset_index() if index else frame
    cells = [[f'{v:.4g}' if isinstance(v, float) else str(v) for v in row] for row in frame.itertuples(index=False)]
    head = [str(c) for c in frame.columns]
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in cells])


def main():
    split = sys.argv[2] if len(sys.argv) > 2 else 'test'
    ours = sys.argv[3] if len(sys.argv) > 3 else OURS
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    if len(sys.argv) > 4:
        rows = [r for r in rows if r['name'] in sys.argv[4].split(',')]
    frame = pd.DataFrame([{'dataset': r['dataset'], 'pct': r['label_percent'], 'task': r['task'], 'name': r['name'],
                           'best_epoch': r['best_epoch'], **r[split]} for r in rows])
    frame = frame.drop_duplicates(['dataset', 'pct', 'name'], keep='last')
    ranks = []
    for (dataset, pct, task), part in frame.groupby(['dataset', 'pct', 'task']):
        for metric, sign in METRICS[task].items():
            rank = (part[metric] * sign).rank(ascending=False)
            for name, value, r in zip(part.name, part[metric], rank):
                ranks.append({'dataset': dataset, 'pct': pct, 'task': task, 'metric': metric, 'name': name,
                              'value': value, 'rank': r, 'primary': metric == PRIMARY[task]})
    ranks = pd.DataFrame(ranks)
    print(f"Split: {split}. Kondisi: {frame.groupby(['dataset', 'pct']).ngroups}, run: {len(frame)}\n")

    table = ranks.groupby(['name', 'primary'])['rank'].mean().unstack().rename(
        columns={True: 'Peringkat metrik utama', False: 'Peringkat metrik lain'})
    table['Menang (#1) metrik utama'] = ranks[ranks.primary & (ranks['rank'] == 1)].groupby('name').size()
    for task in METRICS:
        table[f'Utama: {task}'] = ranks[ranks.primary & (ranks.task == task)].groupby('name')['rank'].mean()
    table = table.fillna(0).sort_values('Peringkat metrik utama').rename(index=lambda n: LABEL.get(n, n))
    print(md(table.round(2)))

    wide = ranks[ranks.primary].pivot_table(index=['dataset', 'pct'], columns='name', values='rank').dropna()
    stat = friedmanchisquare(*[wide[c] for c in wide.columns])
    print(f'\nFriedman (metrik utama, {len(wide)} kondisi, {wide.shape[1]} metode): '
          f'chi2 = {stat.statistic:.2f}, p = {stat.pvalue:.4f}\n')

    tests = []
    primary = wide
    for name in primary.columns.drop(ours):
        diff = primary[name] - primary[ours]  # > 0: ours ranked better
        tests.append({'Pembanding': LABEL.get(name, name), 'Usulan menang': int((diff > 0).sum()),
                      'Seri': int((diff == 0).sum()), 'Usulan kalah': int((diff < 0).sum()),
                      'p Wilcoxon': wilcoxon(diff).pvalue if (diff != 0).any() else 1.0})
    print(md(pd.DataFrame(tests).round(4), index=False))

    print(f'\nMetrik utama per dataset (nilai {split}; peringkat dalam kurung):\n')
    cell = ranks[ranks.primary].assign(text=lambda f: f.value.map('{:.4g}'.format) + ' (' + f['rank'].astype(int).astype(str) + ')')
    detail = cell.pivot_table(index=['task', 'dataset', 'pct'], columns='name', values='text', aggfunc='first')
    order = [c for c in LABEL if c in detail.columns] + [c for c in detail.columns if c not in LABEL]
    print(md(detail[order].rename(columns=lambda n: LABEL.get(n, n))))

    out = Path(sys.argv[1]).parent
    ranks.to_csv(out / f'ranks_{split}_{ours}.csv', index=False)
    pd.DataFrame(tests).to_csv(out / f'wilcoxon_{split}_{ours}.csv', index=False)


if __name__ == '__main__':
    main()
