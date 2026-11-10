"""Validation-only lightweight tree baselines on the existing labeled splits."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, f1_score, mean_squared_error

root = Path(__file__).resolve().parents[1]
data = root / 'eksperimen/hasil/data-5seed'
output = root / 'usulan/hasil/baseline_cpu.csv'
rows = []
for dataset in ('nhanes', 'pendigits', 'superconductivity'):
    for pct in (1, 10, 30):
        folder = data / f'label_{pct:02d}' / dataset
        manifest = json.loads((folder / 'manifest.json').read_text())
        with np.load(folder / 'train_labeled.npz', allow_pickle=False) as f:
            train_x, train_y = f['X'], f['y']
        with np.load(folder / 'validation.npz', allow_pickle=False) as f:
            val_x, val_y = f['X'], f['y']
        is_class = dataset == 'pendigits'
        models = {
            'extra_trees': (ExtraTreesClassifier if is_class else ExtraTreesRegressor)(
                n_estimators=80, min_samples_leaf=2, max_features=.8, n_jobs=2, random_state=42),
            'hist_gradient_boosting': (HistGradientBoostingClassifier if is_class else HistGradientBoostingRegressor)(
                max_iter=120, learning_rate=.08, max_leaf_nodes=15, l2_regularization=1., random_state=42),
        }
        for name, model in models.items():
            start = time.time()
            model.fit(train_x, train_y)
            pred = model.predict(val_x)
            if dataset == 'nhanes':
                levels = np.array(manifest['target_values'], dtype=int)
                labels = levels[np.abs(pred[:, None] - levels).argmin(axis=1)]
                score = cohen_kappa_score(val_y, labels, labels=levels, weights='quadratic')
                metric = 'qwk'
            elif is_class:
                score, metric = f1_score(val_y, pred, average='macro', zero_division=0), 'macro_f1'
            else:
                score, metric = mean_squared_error(val_y, pred), 'mse'
            row = {'dataset': dataset, 'label_percent': pct, 'method': name, 'metric': metric,
                   'validation': score, 'seconds': time.time() - start, 'labeled_count': len(train_y),
                   'seed': 42, 'test_used': False}
            rows.append(row)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(json.dumps(row), flush=True)
