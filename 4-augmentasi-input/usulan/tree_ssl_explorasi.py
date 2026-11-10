"""Cheap supervised-tree and teacher-consistency SSL exploration; validation only."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import cohen_kappa_score, f1_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors

root = Path(__file__).resolve().parents[1]
data = root / 'eksperimen/hasil/data-5seed'
output = root / 'usulan/hasil/tree_ssl_explorasi.csv'
rows = []
for dataset in ('nhanes', 'pendigits', 'superconductivity'):
    for pct in (1, 10, 30):
        folder = data / f'label_{pct:02d}' / dataset
        manifest = json.loads((folder / 'manifest.json').read_text())
        def load(name):
            with np.load(folder / f'{name}.npz', allow_pickle=False) as f:
                return f['X'], f['y'] if 'y' in f else None
        x, y = load('train_labeled')
        u, _ = load('train_unlabeled')
        val, truth = load('validation')
        classifier = dataset == 'pendigits'
        modeltype = ExtraTreesClassifier if classifier else ExtraTreesRegressor
        def model():
            return modeltype(n_estimators=80, min_samples_leaf=2, max_features=.8, n_jobs=2, random_state=42)
        start = time.time()
        teacher = model().fit(x, y)
        # Class confidence or standard deviation across correlated tree predictions.
        if classifier:
            probabilities = teacher.predict_proba(u)
            confidence = probabilities.max(axis=1)
            pseudo = teacher.classes_[probabilities.argmax(axis=1)]
        else:
            preds = np.stack([tree.predict(u) for tree in teacher.estimators_])
            confidence = -preds.std(axis=0)
            pseudo = preds.mean(axis=0)
        k = min(1200, max(1, len(u) // 5))
        selected = np.argsort(-confidence, kind='stable')[:k]
        u_select, target = u[selected], pseudo[selected]
        nn = NearestNeighbors(n_neighbors=2, algorithm='brute', n_jobs=2).fit(u_select)
        neighbor = nn.kneighbors(u_select, return_distance=False)[:, 1]
        mixed = .85 * u_select + .15 * u_select[neighbor]
        mixed_pred = teacher.predict(mixed)
        if classifier:
            valid = mixed_pred == target
        else:
            valid = np.abs(mixed_pred - target) <= .1 * max(float(np.std(y)), 1e-6)
        candidates = {'pseudo_only': (u_select, target),
                      'pseudo_consistent_mix': (np.concatenate([u_select, mixed[valid]]),
                                                np.concatenate([target, target[valid]]))}
        for name, (pseudo_x, pseudo_y) in candidates.items():
            student = model().fit(np.concatenate([x, pseudo_x]), np.concatenate([y, pseudo_y]),
                                  sample_weight=np.r_[np.ones(len(x)), np.full(len(pseudo_x), .2)])
            pred = student.predict(val)
            if dataset == 'nhanes':
                levels = np.array(manifest['target_values'], dtype=int)
                rounded = levels[np.abs(pred[:, None] - levels).argmin(axis=1)]
                score = cohen_kappa_score(truth, rounded, labels=levels, weights='quadratic')
                metric = 'qwk'
            elif classifier:
                score, metric = f1_score(truth, pred, average='macro', zero_division=0), 'macro_f1'
            else:
                score, metric = mean_squared_error(truth, pred), 'mse'
            row = {'dataset': dataset, 'label_percent': pct, 'method': name, 'metric': metric,
                   'validation': float(score), 'labeled_count': len(y), 'pseudo_count': len(pseudo_x),
                   'mix_accepted': int(valid.sum()), 'seconds_cumulative': time.time()-start,
                   'seed': 42, 'test_used': False}
            rows.append(row)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(json.dumps(row), flush=True)
