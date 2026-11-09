"""Describe nearest-neighbor target disagreement using only labeled train data."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

root = Path(__file__).resolve().parents[1]
data = root / 'eksperimen/hasil/data-5seed'
rng = np.random.default_rng(42)
rows = []
for dataset in ('nhanes', 'pendigits', 'superconductivity'):
    for percent in (1, 10, 30):
        with np.load(data / f'label_{percent:02d}' / dataset / 'train_labeled.npz') as f:
            X, y = f['X'], f['y']
        anchors = rng.choice(len(X), size=min(500, len(X)), replace=False)
        neighbors = NearestNeighbors(n_neighbors=3, n_jobs=1).fit(X)
        _, indices = neighbors.kneighbors(X[anchors])
        donors = np.array([next(i for i in row if i != anchor) for anchor, row in zip(anchors, indices)])
        record = {'dataset': dataset, 'label_percent': percent, 'n_labeled': len(X),
                  'n_anchors': len(anchors), 'same_label_rate': None, 'mean_abs_target_difference': None,
                  'labeled_target_std': None}
        if dataset == 'pendigits':
            record['same_label_rate'] = float(np.mean(y[anchors] == y[donors]))
        else:
            diff = np.abs(y[anchors] - y[donors])
            record['mean_abs_target_difference'] = float(np.mean(diff))
            record['labeled_target_std'] = float(np.std(y))
            if dataset == 'nhanes':
                record['same_label_rate'] = float(np.mean(diff == 0))
        rows.append(record)
out = root / 'usulan/diagnostik_tetangga_train.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f'Saved {len(rows)} summaries to {out}')
