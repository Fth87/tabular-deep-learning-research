"""Integration check for the actual prepared datasets and result layout."""
import csv
import json
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import prepare_data
from eksperimen.ringkas_hasil import summarize


class SplitTests(unittest.TestCase):
    def test_all_fractions_share_validation_and_test(self):
        config = prepare_data.read_config(ROOT / 'config/datasets.json')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for percent in (1, 10, 30):
                config['labeled_fraction'] = percent / 100
                for spec in config['datasets']:
                    manifest = prepare_data.prepare_one(spec, config, root / str(percent))
                    if spec['task_type'] != 'regression':
                        self.assertEqual(len(manifest['class_counts_labeled']), len(manifest['target_values'] or manifest['classes']))
                    else:
                        self.assertEqual(manifest['primary_metric'], 'mse')
                        self.assertEqual(manifest['target_values'], [])
                    with np.load(root / str(percent) / spec['name'] / 'train_unlabeled.npz') as unlabeled:
                        self.assertNotIn('y', unlabeled.files)
                    if spec['name'] == 'pendigits':
                        actual_test = pd.read_csv(root / str(percent) / spec['name'] / 'test.csv')
                        official_test = pd.read_csv(ROOT / spec['test_csv'])
                        pd.testing.assert_frame_equal(actual_test, official_test)
                    if spec['name'] == 'superconductivity':
                        features = manifest['feature_columns']
                        def feature_hash(split):
                            frame = pd.read_csv(root / str(percent) / spec['name'] / f'{split}.csv')
                            return set(pd.util.hash_pandas_object(frame[features], index=False))
                        train = feature_hash('train_labeled') | feature_hash('train_unlabeled')
                        val, test = feature_hash('validation'), feature_hash('test')
                        self.assertTrue(train.isdisjoint(val))
                        self.assertTrue(train.isdisjoint(test))
                        self.assertTrue(val.isdisjoint(test))
            for spec in config['datasets']:
                for split in ('validation', 'test'):
                    with np.load(root / '1' / spec['name'] / f'{split}.npz') as reference:
                        for percent in (10, 30):
                            with np.load(root / str(percent) / spec['name'] / f'{split}.npz') as other:
                                np.testing.assert_array_equal(reference['X'], other['X'])
                                np.testing.assert_array_equal(reference['y'], other['y'])

    def test_summary_distinguishes_fraction_and_missing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'label_01' / 'seed_42' / 'nhanes'
            dataset.mkdir(parents=True)
            (dataset / 'rankup_scarf_metrics.json').write_text(json.dumps(
                {'dataset': 'nhanes', 'primary_metric': 'qwk', 'qwk': 0.5, 'train_qwk': 0.6, 'validation_qwk': 0.4, 'seed': 42}))
            summarize(root, root / 'summary.csv')
            with (root / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 54)
            partial = [row for row in rows if row['status'] == 'partial']
            self.assertEqual([(r['label_percent'], r['method'], r['seed_42']) for r in partial], [('1', 'scarf', '0.5')])
            self.assertEqual(partial[0]['mean'], '')
            self.assertEqual(partial[0]['n_seeds'], '1')


if __name__ == '__main__':
    unittest.main()
