"""Check fold isolation and the one-seed workflow on the real datasets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eksperimen'))
import prepare_data
from workflow_kfold import run_fold
from ringkas_kfold import summarize


class KFoldTests(unittest.TestCase):
    def test_folds_keep_test_and_groups_separate(self):
        config = prepare_data.read_config(ROOT / 'config/datasets.json')
        config['labeled_fraction'] = .01
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for spec in config['datasets']:
                tests, validations = [], []
                for fold in range(1, 6):
                    out = root / str(fold)
                    manifest = prepare_data.prepare_one(spec, config, out, fold=fold)
                    with np.load(out / spec['name'] / 'split_indices.npz') as splits:
                        train = set(splits['train_labeled']) | set(splits['train_unlabeled'])
                        validation, test = set(splits['validation']), set(splits['test'])
                        self.assertFalse(train & validation or train & test or validation & test)
                        self.assertEqual(len(train | validation | test), manifest['rows'])
                        tests.append(test)
                        validations.append(validation)
                    if spec['task_type'] != 'regression':
                        self.assertEqual(len(manifest['class_counts_labeled']), len(manifest['target_values'] or manifest['classes']))
                    if spec['name'] == 'superconductivity':
                        features = manifest['feature_columns']
                        def hashes(part):
                            frame = pd.read_csv(out / spec['name'] / f'{part}.csv')
                            return set(pd.util.hash_pandas_object(frame[features], index=False))
                        train_groups = hashes('train_labeled') | hashes('train_unlabeled')
                        self.assertFalse(train_groups & hashes('validation'))
                        self.assertFalse(train_groups & hashes('test'))
                        self.assertFalse(hashes('validation') & hashes('test'))
                self.assertTrue(all(test == tests[0] for test in tests))
                self.assertEqual(sum(map(len, validations)), len(set.union(*validations)))
                self.assertEqual(len(set.union(*validations)) + len(tests[0]), manifest['rows'])
                if spec['name'] == 'pendigits':
                    self.assertEqual(tests[0], set(range(7494, 10992)))

    def test_one_seed_and_fold_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            def fake_run(command, check):
                calls.append(command)
                get = lambda k: command[command.index(k) + 1]
                out = Path(get('--output-root')) / 'nhanes'
                out.mkdir(parents=True, exist_ok=True)
                if '--tune-only' in command:
                    (out / 'rankup_gaussian_best_params.json').write_text('{}')
                else:
                    (out / 'rankup_gaussian_metrics.json').write_text(json.dumps({
                        'dataset': 'nhanes', 'primary_metric': 'qwk', 'seed': 42,
                        'qwk': .3, 'train_qwk': .4, 'validation_qwk': .2}))
                    for suffix in ('run_config.json', 'history.csv', 'predictions.npz', 'model.pt'):
                        (out / f'rankup_gaussian_{suffix}').write_text('data')
            for fold in range(1, 6):
                out = root/'kfold'/'label_01'/f'fold_{fold:02d}'
                result = run_fold('nhanes', 'gaussian', root/'data', out, 42, 2, 1, 1, fake_run)
                self.assertEqual(result['seed'], 42)
            self.assertEqual(len(calls), 10)
            calls.clear()
            run_fold('nhanes', 'gaussian', root/'data', root/'kfold'/'label_01'/'fold_01',
                     42, 2, 1, 1, fake_run)
            self.assertEqual(len(calls), 1)
            self.assertIn('--tune-only', calls[0])
            output = root/'summary.csv'
            summarize(root, output)
            import csv
            with output.open() as f:
                rows = list(csv.DictReader(f))
            row = next(r for r in rows if r['label_percent'] == '1' and r['dataset'] == 'nhanes' and r['method'] == 'gaussian')
            self.assertEqual(row['status'], 'complete')
            self.assertEqual(row['n_folds'], '5')
            self.assertAlmostEqual(float(row['validation_mean']), .2)
            self.assertAlmostEqual(float(row['test_std']), 0.)
            (root/'kfold'/'label_01'/'fold_03'/'seed_42'/'nhanes'/'rankup_gaussian_model.pt').write_text('changed')
            summarize(root, output)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            row = next(r for r in rows if r['label_percent'] == '1' and r['dataset'] == 'nhanes' and r['method'] == 'gaussian')
            self.assertEqual(row['status'], 'partial')
            self.assertEqual(row['missing_folds'], '3')
            self.assertEqual(row['test_mean'], '')


if __name__ == '__main__':
    unittest.main()
