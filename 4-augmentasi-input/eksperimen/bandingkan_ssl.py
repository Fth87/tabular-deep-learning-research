"""Compare RankUp (seed 42) with supervised and SSL baselines on NHANES, Pen Digits and Superconductivity.

Classical baselines run here on CPU; deep SSL baselines come from hasil/pembanding/ (modal_pembanding.py).
Usage: uv run python eksperimen/bandingkan_ssl.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ringkas_usulan import baseline_rows  # noqa: E402

HASIL = Path(__file__).resolve().parent / 'hasil'
DATASETS = ('nhanes', 'pendigits', 'superconductivity')
PRIMARY = {'ordinal_regression': ('qwk', 1), 'classification': ('macro_f1', 1), 'regression': ('mse', -1)}


def load(folder, name):
    with np.load(folder / f'{name}.npz', allow_pickle=False) as split:
        return split['X'].astype(np.float64), (split['y'] if 'y' in split else None)


def scores(task, y, pred, levels=None):
    if task == 'classification':
        return {'macro_f1': f1_score(y, pred, average='macro', zero_division=0), 'accuracy': accuracy_score(y, pred)}
    out = {'mse': mean_squared_error(y, pred), 'mae': mean_absolute_error(y, pred), 'r2': r2_score(y, pred),
           'spearman': spearmanr(y, pred).statistic}
    if task == 'ordinal_regression':
        rounded = levels[np.abs(np.asarray(pred).reshape(-1, 1) - levels).argmin(axis=1)]
        out.update(qwk=cohen_kappa_score(y, rounded, weights='quadratic'), accuracy=accuracy_score(y, rounded))
    return out


def coreg(x_lab, y_lab, x_unlab, iterations=100, pool=100, k=3, seed=0):
    """COREG (Zhou & Li, 2005): two kNN regressors (p=2, p=5) teach each other confident pseudo-labels."""
    rng = np.random.default_rng(seed)
    sets = [[x_lab.copy(), y_lab.astype(float).copy()] for _ in range(2)]
    remaining = np.arange(len(x_unlab))
    for _ in range(iterations):
        added = False
        for view, p in enumerate((2, 5)):
            x_l, y_l = sets[view]
            if len(remaining) == 0:
                break
            candidates = rng.choice(remaining, size=min(pool, len(remaining)), replace=False)
            knn = NearestNeighbors(n_neighbors=min(k, len(x_l)), p=p).fit(x_l)
            neighbours = knn.kneighbors(x_unlab[candidates], return_distance=False)
            guess = y_l[neighbours].mean(1)
            gains = []
            for c, (idx, value) in enumerate(zip(neighbours, guess)):
                # Error change on the candidate's labelled neighbours when it joins the training set.
                own = knn.kneighbors(x_l[idx], return_distance=False)
                before = y_l[own].mean(1)
                dist_new = np.abs(x_l[idx] - x_unlab[candidates[c]]) ** p
                far = np.abs(x_l[idx][:, None] - x_l[own]) ** p
                replace = dist_new.sum(1) < far.sum(2).max(1)
                after = np.where(replace, (y_l[own].sum(1) - y_l[own][:, -1] + value) / own.shape[1], before)
                gains.append(np.sum((y_l[idx] - before) ** 2 - (y_l[idx] - after) ** 2))
            best = int(np.argmax(gains))
            if gains[best] > 0:
                other = sets[1 - view]
                other[0] = np.vstack([other[0], x_unlab[candidates[best]]])
                other[1] = np.append(other[1], guess[best])
                remaining = remaining[remaining != candidates[best]]
                added = True
        if not added:
            break
    models = [NearestNeighbors(n_neighbors=min(k, len(x)), p=p).fit(x) for (x, _), p in zip(sets, (2, 5))]
    return lambda x: np.mean([y[m.kneighbors(x, return_distance=False)].mean(1)
                              for m, (_, y) in zip(models, sets)], axis=0)


def self_training_regressor(x_lab, y_lab, x_unlab, rounds=3, share=0.1, seed=0):
    """Pseudo-label the most agreed-upon unlabeled rows (lowest bootstrap variance) for a few rounds."""
    rng = np.random.default_rng(seed)
    x_l, y_l, pool = x_lab, y_lab.astype(float), x_unlab
    for _ in range(rounds):
        boots = [HistGradientBoostingRegressor(random_state=b).fit(*(lambda i: (x_l[i], y_l[i]))(
            rng.integers(len(x_l), size=len(x_l)))) for b in range(5)]
        preds = np.stack([m.predict(pool) for m in boots])
        take = np.argsort(preds.std(0))[:max(1, int(share * len(pool)))]
        x_l, y_l = np.vstack([x_l, pool[take]]), np.append(y_l, preds.mean(0)[take])
        pool = np.delete(pool, take, axis=0)
    return HistGradientBoostingRegressor(random_state=0).fit(x_l, y_l).predict


def classical(dataset, percent):
    folder = HASIL / 'data-5seed' / f'label_{percent:02d}' / dataset
    manifest = json.loads((folder / 'manifest.json').read_text())
    task = manifest['task_type']
    (x_l, y_l), (x_u, _), (x_v, y_v), (x_t, y_t) = (load(folder, s) for s in
                                                    ('train_labeled', 'train_unlabeled', 'validation', 'test'))
    rows = []
    if task == 'classification':
        y_l, y_v, y_t = (np.asarray(y).astype(str) for y in (y_l, y_v, y_t))
        ids = {c: i for i, c in enumerate(sorted(set(y_l)))}
        y_l_id = np.array([ids[c] for c in y_l])
        back = np.array(sorted(ids))
        hgb = HistGradientBoostingClassifier(random_state=0)
        models = {
            'HGB (supervised)': hgb.fit(x_l, y_l).predict,
            'Logistic (supervised)': LogisticRegression(max_iter=3000).fit(x_l, y_l).predict,
            'Self-training HGB': (lambda m: lambda x: back[m.predict(x)])(SelfTrainingClassifier(
                HistGradientBoostingClassifier(random_state=0), threshold=0.9).fit(
                np.vstack([x_l, x_u]), np.r_[y_l_id, -np.ones(len(x_u), int)])),
            'Label Spreading': (lambda m: lambda x: back[m.predict(x)])(LabelSpreading(
                kernel='knn', n_neighbors=10, alpha=0.2, max_iter=100).fit(
                np.vstack([x_l, x_u]), np.r_[y_l_id, -np.ones(len(x_u), int)])),
        }
        levels = None
    else:
        y_l, y_v, y_t = (np.asarray(y, dtype=float) for y in (y_l, y_v, y_t))
        levels = np.asarray(manifest['target_values'], float) if task == 'ordinal_regression' else None
        models = {
            'HGB (supervised)': HistGradientBoostingRegressor(random_state=0).fit(x_l, y_l).predict,
            'Ridge (supervised)': RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_l, y_l).predict,
            'Self-training HGB': self_training_regressor(x_l, y_l, x_u),
            'COREG': coreg(x_l, y_l, x_u),
        }
    for name, predict in models.items():
        rows.append({'dataset': dataset, 'label_percent': percent, 'task': task, 'name': name, 'seed': 42,
                     'best_epoch': 0, 'validation': scores(task, y_v, predict(x_v), levels),
                     'test': scores(task, y_t, predict(x_t), levels)})
        print('done', dataset, percent, name, round(rows[-1]['test'][PRIMARY[task][0]], 4), flush=True)
    return rows


def deep_rows():
    rows = []
    for path in sorted((HASIL / 'pembanding').glob('*_*_*.npz')):
        method, dataset, percent = path.stem.rsplit('_', 2)[0], path.stem.rsplit('_', 2)[1], int(path.stem[-2:])
        task = 'ordinal_regression' if dataset == 'nhanes' else 'classification'
        manifest = json.loads((HASIL / 'data-5seed' / f'label_{percent:02d}' / dataset / 'manifest.json').read_text())
        with np.load(path) as data:
            levels = np.asarray(manifest['target_values'], float) if task != 'classification' else None
            test = scores(task, data['y_true'].astype(float) if levels is not None else data['y_true'],
                          data['y_pred'].astype(float) if levels is not None else data['y_pred'], levels)
        rows.append({'dataset': dataset, 'label_percent': percent, 'task': task, 'name': method.upper(),
                     'seed': 42, 'best_epoch': 0, 'validation': {}, 'test': test})
    return rows


def rankup_rows():
    names = {'none': 'TabM tanpa SSL', 'gibbs_k5_s50': 'RankUp + Block-Gibbs (usulan)'}
    names.update({f'def_{m}': f'RankUp + {m} (tanpa tuning)' for m in
                  ('tabcutmix', 'scarf', 'range_limited', 'gaussian', 'zero_out', 'tabcutmix_plus')})
    rows = [dict(r, name=names[r['name']]) for r in map(json.loads, (HASIL / 'usulan-semua' / 'results.jsonl')
            .read_text().splitlines()) if r['dataset'] in DATASETS and r['name'] in names and r['seed'] == 42]
    tasks = {'nhanes': 'ordinal_regression', 'pendigits': 'classification', 'superconductivity': 'regression'}
    lama = baseline_rows()
    for r in lama[lama.seed == 42].to_dict('records'):
        rows.append({'dataset': r['dataset'], 'label_percent': r['label_percent'], 'task': tasks[r['dataset']],
                     'name': f"RankUp + {r['name'][5:]} (Optuna)", 'seed': 42, 'best_epoch': 0, 'validation': {},
                     'test': {k[5:]: v for k, v in r.items() if k.startswith('test_') and pd.notna(v)}})
    return rows


def main():
    cache = HASIL / 'pembanding' / 'klasik.jsonl'
    if not cache.exists():
        rows = [row for d in DATASETS for p in (1, 10, 30) for row in classical(d, p)]
        cache.write_text('\n'.join(json.dumps(r, default=float) for r in rows) + '\n')
    rows = [json.loads(line) for line in cache.read_text().splitlines()] + deep_rows() + rankup_rows()
    out = HASIL / 'pembanding' / 'semua.jsonl'
    out.write_text('\n'.join(json.dumps(r, default=float) for r in rows) + '\n')
    frame = pd.DataFrame([{'dataset': r['dataset'], 'pct': r['label_percent'], 'name': r['name'],
                           'value': r['test'][PRIMARY[r['task']][0]], 'sign': PRIMARY[r['task']][1]} for r in rows])
    frame['rank'] = (frame.value * frame.sign).groupby([frame.dataset, frame.pct]).rank(ascending=False)
    table = frame.pivot_table(index='name', columns=['dataset', 'pct'], values='value')
    ranks = frame.pivot_table(index='name', columns=['dataset', 'pct'], values='rank')
    table['rata-rata peringkat'] = ranks.mean(axis=1)
    table['kondisi'] = ranks.notna().sum(axis=1)
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 60)
    print(table.sort_values('rata-rata peringkat').round(4))
    frame.to_csv(HASIL / 'pembanding' / 'peringkat.csv', index=False)


if __name__ == '__main__':
    main()
