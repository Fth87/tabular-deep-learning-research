"""Measure how well each augmentation keeps the data distribution and the target order.

Uses train and validation only. Run: uv run python eksperimen/analisis_operator.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent / 'hasil' / 'data-5seed' / 'label_30'
TASKS = {'nhanes': 'ordinal', 'pendigits': 'classification', 'superconductivity': 'regression'}
rng = np.random.default_rng(0)


def mask_of(x, strength):
    count = min(x.shape[1], max(1, round(strength * x.shape[1])))
    order = rng.random(x.shape).argsort(axis=1)
    mask = np.zeros(x.shape, bool)
    np.put_along_axis(mask, order[:, :count], True, axis=1)
    return mask


def gibbs(x, strength, train, k, pool=2048, chunk=256):
    mask = mask_of(x, strength)
    out = x.copy()
    for start in range(0, len(x), chunk):
        part, keep = x[start:start + chunk], ~mask[start:start + chunk]
        candidates = train[rng.integers(len(train), size=min(pool, len(train)))]
        distance = ((part[:, None] - candidates) ** 2 * keep[:, None]).sum(-1)
        distance[(part[:, None] == candidates).all(-1)] = np.inf
        near = np.argpartition(distance, k - 1, axis=1)[:, :k]
        donor = near[np.arange(len(part)), rng.integers(k, size=len(part))]
        out[start:start + chunk] = np.where(keep, part, candidates[donor])
    return out


def scarf(x, strength, train, _):
    picks = train[rng.integers(len(train), size=x.shape), np.arange(x.shape[1])]
    return np.where(mask_of(x, strength), picks, x)


def swap(x, strength, train, _):
    return np.where(mask_of(x, strength), train[rng.integers(len(train), size=len(x))], x)


def gaussian(x, strength, train, _):
    return x + rng.normal(scale=strength, size=x.shape)


def zero(x, strength, train, _):
    return np.where(mask_of(x, strength), 0.0, x)


OPERATORS = {'scarf': (scarf, None), 'random_swap': (swap, None), 'gaussian': (gaussian, None),
             'zero_out': (zero, None), 'gibbs_k5': (gibbs, 5), 'gibbs_k20': (gibbs, 20)}


def load(folder, name):
    with np.load(folder / f'{name}.npz', allow_pickle=False) as split:
        return split['X'].astype(np.float32), (split['y'] if 'y' in split else None)


def realism_auc(real, fake):
    """Classifier two-sample test: 0.5 means augmented rows look real."""
    n = min(len(real), len(fake), 3000)
    # A row and its augmented twin stay in one fold, otherwise the model recognises the twin.
    fit, held = train_test_split(np.arange(n), test_size=0.3, random_state=0)
    make = lambda idx: (np.vstack([real[idx], fake[idx]]), np.r_[np.ones(len(idx)), np.zeros(len(idx))])
    a, ya = make(fit)
    b, yb = make(held)
    model = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(a, ya)
    return roc_auc_score(yb, model.predict_proba(b)[:, 1])


def main():
    rows = []
    for dataset, task in TASKS.items():
        folder = ROOT / dataset
        x_lab, y_lab = load(folder, 'train_labeled')
        x_unlab, _ = load(folder, 'train_unlabeled')
        x_val, _ = load(folder, 'validation')
        train = np.vstack([x_lab, x_unlab])
        if task == 'classification':
            y_lab = np.asarray(y_lab).astype(str)
            oracle = HistGradientBoostingClassifier(random_state=0).fit(x_lab, y_lab)
        else:
            oracle = HistGradientBoostingRegressor(random_state=0).fit(x_lab, np.asarray(y_lab, float))
        clean = oracle.predict(x_val)
        base_corr = np.nan_to_num(np.corrcoef(train[:5000], rowvar=False))
        i, j = rng.integers(len(x_val), size=(2, 40000))
        for strength in (0.3, 0.5):
            for name, (fn, k) in OPERATORS.items():
                fake = fn(x_val, strength, train, k)
                pred = oracle.predict(fake)
                row = {'dataset': dataset, 'strength': strength, 'operator': name,
                       'realism_auc': realism_auc(x_val, fake),
                       'corr_error': np.sqrt(np.mean((np.nan_to_num(np.corrcoef(fake, rowvar=False)) - base_corr) ** 2))}
                if task == 'classification':
                    row['label_kept'] = np.mean(pred == clean)
                else:
                    other = oracle.predict(fn(x_val, strength, train, k))
                    keep = clean[i] != clean[j]
                    row['label_kept'] = np.mean(np.sign(pred[i] - other[j])[keep] == np.sign(clean[i] - clean[j])[keep])
                    row['spearman_clean'] = spearmanr(clean, pred).statistic
                rows.append(row)
                print(json.dumps({key: round(v, 4) if isinstance(v, float) else v for key, v in row.items()}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(__file__).resolve().parent / 'hasil' / 'analisis_operator.csv', index=False)
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    for column in ('realism_auc', 'corr_error', 'label_kept'):
        print(column)
        print(frame.pivot_table(index=['dataset', 'strength'], columns='operator', values=column).round(3))


if __name__ == '__main__':
    main()
