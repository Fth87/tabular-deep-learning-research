"""Check the one-tuning, five-seed workflow with a fake subprocess runner."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eksperimen.experiments.workflow import run_experiment, SEEDS
from eksperimen.ringkas_hasil import summarize


class WorkflowTests(unittest.TestCase):
    def test_interrupted_training_resumes_at_missing_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            interrupted = True
            def fake_run(command, check):
                nonlocal interrupted
                get = lambda key: command[command.index(key) + 1]
                out = Path(get('--output-root')) / 'nhanes'
                out.mkdir(parents=True, exist_ok=True)
                if '--tune-only' in command:
                    (out / 'rankup_gaussian_best_params.json').write_text('{}')
                    return
                seed = int(get('--seed'))
                calls.append(seed)
                if seed == 456 and interrupted:
                    interrupted = False
                    raise RuntimeError('simulated interruption')
                (out / 'rankup_gaussian_metrics.json').write_text(json.dumps({
                    'dataset': 'nhanes', 'primary_metric': 'qwk', 'seed': seed,
                    'qwk': .2, 'train_qwk': .3, 'validation_qwk': .4}))
                for name in ('run_config.json', 'history.csv', 'predictions.npz', 'model.pt'):
                    (out / f'rankup_gaussian_{name}').write_text('test')
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                run_experiment('nhanes', 'gaussian', root/'data', root, 42, 2, 1, 1, fake_run)
            self.assertEqual(calls, [42, 123, 456])
            calls.clear()
            run_experiment('nhanes', 'gaussian', root/'data', root, 42, 2, 1, 1, fake_run)
            self.assertEqual(calls, [456, 789, 101112])

    def test_tune_once_then_reuse_parameters_for_five_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            def fake_run(command, check):
                calls.append(command)
                self.assertTrue(check)
                get = lambda key: command[command.index(key) + 1]
                out = Path(get('--output-root')) / 'nhanes'
                out.mkdir(parents=True, exist_ok=True)
                if '--tune-only' in command:
                    (out / 'rankup_gaussian_best_params.json').write_text('{}')
                else:
                    params = Path(get('--params-file'))
                    self.assertEqual(params, root / 'tuning/nhanes/rankup_gaussian_best_params.json')
                    seed = int(get('--seed'))
                    (out / 'rankup_gaussian_metrics.json').write_text(json.dumps({
                        'dataset': 'nhanes', 'primary_metric': 'qwk', 'qwk': SEEDS.index(seed) / 10,
                        'train_qwk': 0.6, 'validation_qwk': 0.4,
                        'seed': seed}))
                    for name in ('run_config.json', 'history.csv', 'predictions.npz', 'model.pt'):
                        (out / f'rankup_gaussian_{name}').write_text('test')
            summary = run_experiment('nhanes', 'gaussian', root / 'data', root, 42, 2, 1, 1, fake_run)
            self.assertEqual(len(calls), 6)
            self.assertEqual(sum('--tune-only' in call for call in calls), 1)
            self.assertEqual([int(call[call.index('--seed')+1]) for call in calls[1:]], list(SEEDS))
            self.assertAlmostEqual(summary['mean'], .2)
            self.assertEqual(summary['n_seeds'], 5)
            self.assertTrue((root/'summary/nhanes/rankup_gaussian_summary.json').exists())
            calls.clear()
            run_experiment('nhanes', 'gaussian', root / 'data', root, 42, 2, 1, 1, fake_run)
            self.assertEqual(len(calls), 1)
            self.assertIn('--tune-only', calls[0])
            calls.clear()
            (root/'seed_456/nhanes/rankup_gaussian_model.pt').write_text('corrupted')
            run_experiment('nhanes', 'gaussian', root / 'data', root, 42, 2, 1, 1, fake_run)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][calls[1].index('--seed') + 1], '456')
            with self.assertRaisesRegex(ValueError, 'run-id baru'):
                run_experiment('nhanes', 'gaussian', root / 'data', root, 42, 3, 1, 1, fake_run)

    def test_summary_requires_all_five_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i, seed in enumerate(SEEDS):
                out = root/'label_01'/f'seed_{seed}'/'nhanes'
                out.mkdir(parents=True)
                (out/'rankup_gaussian_metrics.json').write_text(json.dumps({
                    'dataset': 'nhanes', 'primary_metric': 'qwk', 'seed': seed, 'qwk': i/10,
                    'train_qwk': 0.6, 'validation_qwk': 0.4}))
            summarize(root, root/'summary.csv')
            with (root/'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            match = next(r for r in rows if r['dataset']=='nhanes' and r['method']=='gaussian' and r['label_percent']=='1')
            self.assertEqual(match['status'], 'complete')
            self.assertEqual(match['n_seeds'], '5')
            self.assertAlmostEqual(float(match['mean']), .2)
            self.assertGreater(float(match['std']), 0)
            (root/'label_01'/f'seed_{SEEDS[-1]}'/'nhanes'/'rankup_gaussian_metrics.json').unlink()
            summarize(root, root/'summary.csv')
            with (root/'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            match = next(r for r in rows if r['dataset']=='nhanes' and r['method']=='gaussian' and r['label_percent']=='1')
            self.assertEqual(match['status'], 'partial')
            self.assertEqual(match['mean'], '')


if __name__ == '__main__':
    unittest.main()
