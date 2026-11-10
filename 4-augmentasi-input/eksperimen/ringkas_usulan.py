"""Compare the proposed runs with the tuned 5-seed baselines.

Usage: uv run python eksperimen/ringkas_usulan.py eksperimen/hasil/usulan-screening/results.jsonl
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

HASIL = Path(__file__).resolve().parent / 'hasil'
PRIMARY = {'nhanes': ('qwk', 1), 'pendigits': ('macro_f1', 1), 'superconductivity': ('mse', -1)}
METRICS = {'nhanes': {'qwk': 1, 'mae': -1, 'spearman': 1, 'accuracy': 1},
           'pendigits': {'macro_f1': 1, 'accuracy': 1},
           'superconductivity': {'mse': -1, 'mae': -1, 'r2': 1, 'spearman': 1}}


def baseline_rows():
    rows = []
    for path in glob.glob(str(HASIL / 'ta-5seed-40trial-20261105/label_*/seed_*/*/*_predictions.npz')):
        parts = Path(path).parts
        dataset, seed, percent = parts[-2], int(parts[-3][5:]), int(parts[-4][6:])
        method = Path(path).name.replace('rankup_cls_', '').replace('rankup_', '').replace('_predictions.npz', '')
        with np.load(path) as data:
            y = data['y_true']
            if dataset == 'pendigits':
                test = {'macro_f1': f1_score(y, data['y_pred'], average='macro'),
                        'accuracy': accuracy_score(y, data['y_pred'])}
            else:
                score = data['y_score']
                test = {'mse': mean_squared_error(y, score), 'mae': mean_absolute_error(y, score),
                        'r2': r2_score(y, score), 'spearman': spearmanr(y, score).statistic}
                if dataset == 'nhanes':
                    test.update(qwk=cohen_kappa_score(y, data['y_pred'], weights='quadratic'),
                                accuracy=accuracy_score(y, data['y_pred']))
        rows.append({'dataset': dataset, 'label_percent': percent, 'name': f'lama_{method}',
                     'seed': seed, **{f'test_{k}': v for k, v in test.items()}})
    return pd.DataFrame(rows)


def main():
    raw = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    ours = pd.DataFrame([{'dataset': r['dataset'], 'label_percent': r['label_percent'], 'name': r['name'],
                          'seed': r['seed'], 'accept': r['accept_rate_last5'], 'best_epoch': r['best_epoch'],
                          **{f'val_{k}': v for k, v in r['validation'].items()},
                          **{f'test_{k}': v for k, v in r['test'].items()}} for r in raw])
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 200)

    # Selection uses validation only: mean rank of the primary validation metric over the 9 conditions.
    ours['val_primary'] = [r[f'val_{PRIMARY[r["dataset"]][0]}'] * PRIMARY[r['dataset']][1]
                           for _, r in ours.iterrows()]
    per_condition = ours.groupby(['dataset', 'label_percent', 'name'])['val_primary'].mean().reset_index()
    per_condition['val_rank'] = per_condition.groupby(['dataset', 'label_percent'])['val_primary'].rank(ascending=False)
    print('Validation rank (lower is better):')
    print(per_condition.pivot_table(index='name', columns=['dataset', 'label_percent'], values='val_rank')
          .assign(mean=lambda f: f.mean(axis=1)).sort_values('mean').round(1))
    print('\nAccept rate (last 5 epochs):')
    print(ours.pivot_table(index='name', columns=['dataset', 'label_percent'], values='accept').round(2))

    both = pd.concat([ours, baseline_rows()], ignore_index=True)
    seeds = sorted(ours.seed.unique())
    both = both[both.seed.isin(seeds)]
    for dataset, metrics in METRICS.items():
        for metric, sign in metrics.items():
            table = both[both.dataset == dataset].pivot_table(
                index='name', columns='label_percent', values=f'test_{metric}')
            ranks = (table * sign).rank(ascending=False)
            print(f'\n{dataset} test {metric} (seeds {seeds}), rank in brackets:')
            print(table.round(4).astype(str) + ' [' + ranks.fillna(0).astype(int).astype(str) + ']')


if __name__ == '__main__':
    main()


def final(path, ours='gibbs_k5_s50'):
    """Mean over shared seeds, rank per condition, and paired Wilcoxon of ours vs each baseline."""
    from scipy.stats import wilcoxon
    raw = [json.loads(line) for line in Path(path).read_text().splitlines()]
    new = pd.DataFrame([{'dataset': r['dataset'], 'label_percent': r['label_percent'], 'name': r['name'],
                         'seed': r['seed'], **{f'test_{k}': v for k, v in r['test'].items()}} for r in raw])
    data = pd.concat([new[new.name.isin([ours, 'none'])], baseline_rows()], ignore_index=True)
    seeds = sorted(set(new[new.name == ours].seed))
    data = data[data.seed.isin(seeds)]
    rank_rows, tests = [], []
    for dataset, metrics in METRICS.items():
        for metric, sign in metrics.items():
            part = data[data.dataset == dataset]
            mean = part.pivot_table(index='name', columns='label_percent', values=f'test_{metric}')
            rank = (mean * sign).rank(ascending=False)
            for percent in mean.columns:
                for name in mean.index:
                    rank_rows.append({'dataset': dataset, 'metric': metric, 'label_percent': percent,
                                      'name': name, 'mean': mean.loc[name, percent], 'rank': rank.loc[name, percent]})
            for name in mean.index.drop(ours):
                a = part[part.name == ours].set_index(['label_percent', 'seed'])[f'test_{metric}']
                b = part[part.name == name].set_index(['label_percent', 'seed'])[f'test_{metric}']
                diff = ((a - b.reindex(a.index)) * sign).dropna()
                tests.append({'dataset': dataset, 'metric': metric, 'vs': name, 'n': len(diff),
                              'win_rate': float((diff > 0).mean()),
                              'p_wilcoxon': float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0})
    ranks = pd.DataFrame(rank_rows)
    print(f'\nSeeds {seeds}. Mean rank over all datasets, label fractions and metrics (1 = best of {ranks.name.nunique()}):')
    print(ranks.groupby('name')['rank'].agg(['mean', 'median', 'min', 'max']).sort_values('mean').round(2))
    print('\nMean rank per dataset x label fraction (averaged over metrics):')
    print(ranks.pivot_table(index='name', columns=['dataset', 'label_percent'], values='rank').round(1)
          .assign(mean=lambda f: f.mean(axis=1)).sort_values('mean'))
    print('\nPrimary metric means:')
    for dataset, (metric, _) in PRIMARY.items():
        print(dataset, metric)
        print(ranks[(ranks.dataset == dataset) & (ranks.metric == metric)]
              .pivot_table(index='name', columns='label_percent', values='mean').round(4))
    t = pd.DataFrame(tests)
    print(f'\n{ours} vs each baseline, paired over label fraction x seed (win_rate > 0.5 = ours better):')
    print(t.pivot_table(index='vs', columns=['dataset', 'metric'], values='win_rate').round(2))
    print('\nWilcoxon p-values:')
    print(t.pivot_table(index='vs', columns=['dataset', 'metric'], values='p_wilcoxon').round(3))
    ranks.to_csv(HASIL / 'usulan-final-ranks.csv', index=False)
    t.to_csv(HASIL / 'usulan-final-tests.csv', index=False)
