"""Run with a Python environment containing CPU PyTorch: python -m unittest discover -s eksperimen/tests."""
import ast
import importlib
import json
import tempfile
from unittest import mock
import sys
from types import SimpleNamespace
from pathlib import Path
import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = F = None

ROOT = Path(__file__).resolve().parents[1] / 'experiments'


def get_functions(path, *names):
    tree = ast.parse(path.read_text())
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(wanted) != len(names):
        raise AssertionError(f'{path}: expected {names}, found {[f.name for f in wanted]}')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@unittest.skipUnless(torch is not None, "requires PyTorch; run in the training environment")
class RankUpTests(unittest.TestCase):
    def test_study_resumes_remaining_trials_and_sampler_state(self):
        import optuna
        sys.path.insert(0, str(ROOT.parent))
        from experiments.resume import load_study, save_study, remaining_trials
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'study.pkl'
            def create():
                return optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
            def objective(trial):
                return trial.suggest_float('x', -1, 1)
            uninterrupted = create()
            uninterrupted.optimize(objective, n_trials=4)
            interrupted = create()
            interrupted.optimize(objective, n_trials=2, callbacks=[lambda s, _: save_study(s, path)])
            resumed = load_study(create(), path)
            self.assertEqual(remaining_trials(resumed, 4), 2)
            resumed.optimize(objective, n_trials=remaining_trials(resumed, 4))
            self.assertEqual([t.params for t in resumed.trials], [t.params for t in uninterrupted.trials])
            self.assertEqual(remaining_trials(resumed, 4), 0)

    def test_scarf_draws_independent_feature_donors(self):
        for family, aug in [('rankup', 'augment_method'), ('rankup_classification', 'augment'), ('regression', 'augment_method')]:
            path = ROOT / family / 'scarf.py'
            fns = get_functions(path, 'feature_mask', aug)
            source = torch.arange(20, dtype=torch.float32).unsqueeze(1).repeat(1, 6)
            x = torch.full((200, 6), -1.0)
            torch.manual_seed(42)
            result = fns[aug](x, 1.0, source, None, None, None, None, None)
            self.assertTrue(((result != result[:, :1]).any(dim=1)).any(), str(path))
            self.assertTrue(((result >= 0) & (result < 20)).all())

    def test_unlabeled_loss_averages_all_pairs(self):
        for path in sorted([*(ROOT / 'rankup').glob('*.py'), *(ROOT / 'regression').glob('*.py')]):
            if path.name == '__init__.py':
                continue
            fns = get_functions(path, 'pairwise_logits', 'pseudo_rank_loss')
            weak = torch.tensor([[5., 0.], [0., 5.]])
            strong = torch.zeros_like(weak)
            loss, rate = fns['pseudo_rank_loss'](weak, strong, 0.95)
            self.assertAlmostEqual(rate, 0.5)
            self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.)).item() * rate, places=6, msg=str(path))

    def test_regression_and_ordinal_train_step_on_cpu(self):
        import numpy as np
        if torch.cuda.is_available():
            self.skipTest('CPU test')
        sys.path.insert(0, str(ROOT.parent))
        rng = np.random.default_rng(42)
        x_lab = rng.normal(size=(8, 4)).astype(np.float32)
        x_unlab = rng.normal(size=(8, 4)).astype(np.float32)
        x_val = rng.normal(size=(10, 4)).astype(np.float32)
        args = SimpleNamespace(seed=42, learning_rate=.002, batch_size=8, max_steps=1,
            donor_confidence=.8, confidence_cutoff=.95, rank_weight=.2,
            unlabeled_weight=.2, patience=2, ordinal_levels=np.arange(5))
        for family, method in [('rankup', 'gaussian'), *[('regression', m) for m in
                ('gaussian', 'scarf', 'range_limited', 'tabcutmix', 'tabcutmix_plus', 'zero_out')]]:
            with self.subTest(family=family, method=method):
                module = importlib.import_module(f'experiments.{family}.{method}')
                y_lab = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.float32)
                y_val = np.tile(np.arange(5, dtype=np.float32), 2)
                if family == 'regression':
                    y_lab = y_lab * .27 + .123
                    y_val = y_val * .27 + .123
                score, epoch, state, history = module.fit_model(
                    args, x_lab, y_lab, x_unlab, x_val, y_val, .2, .6, 1)
                self.assertTrue(np.isfinite(score))
                self.assertEqual(epoch, 1)
                self.assertTrue(state)
                self.assertIn('validation_qwk' if family == 'rankup' else 'validation_mse', history[0])

    def test_tuning_parameters_reused_by_final_training(self):
        import numpy as np
        if torch.cuda.is_available():
            self.skipTest('CPU-only integration test')
        sys.path.insert(0, str(ROOT.parent))
        real_torch = torch
        class CudaProxy:
            def is_available(self): return True
            def get_device_name(self, device): return 'CPU synthetic test'
            def __getattr__(self, name): return getattr(real_torch.cuda, name)
        class TorchProxy:
            cuda = CudaProxy()
            def device(self, *args, **kwargs): return real_torch.device('cpu')
            def __getattr__(self, name): return getattr(real_torch, name)
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for dataset, family, task, metric in (
                ('nhanes', 'rankup', 'ordinal_regression', 'qwk'),
                ('superconductivity', 'regression', 'regression', 'mse'),
                ('pendigits', 'rankup_classification', 'classification', 'macro_f1')):
                with self.subTest(dataset=dataset):
                    module = importlib.import_module(f'experiments.{family}.gaussian')
                    folder = root / 'data' / dataset
                    folder.mkdir(parents=True)
                    manifest = {'task_type': task, 'seed': 42, 'labeled_fraction_of_train': .1,
                                'target_values': [0, 1, 2, 3, 4],
                                'classes': ['0.0', '1.0', '2.0', '3.0', '4.0'], 'labeled_class_ratio': 1.0}
                    (folder / 'manifest.json').write_text(json.dumps(manifest))
                    y = np.tile(np.arange(5, dtype=np.float32), 2)
                    if task == 'regression': y = y * .27 + .123
                    if task == 'classification': y = y.astype(str)
                    for name in ('train_labeled', 'validation'):
                        np.savez(folder / f'{name}.npz', X=rng.normal(size=(10,4)).astype('float32'), y=y)
                    np.savez(folder / 'train_unlabeled.npz', X=rng.normal(size=(10,4)).astype('float32'))
                    args = SimpleNamespace(dataset=dataset, data_root=root/'data', output_root=root/'tuning',
                        seed=42, params_file=None, tune_only=True, n_trials=1, optuna_epochs=1,
                        epochs=1, learning_rate=.002, batch_size=10, max_steps=1, donor_confidence=.8,
                        confidence_cutoff=.95, rank_weight=.2, unlabeled_weight=.2, patience=2)
                    with mock.patch.object(module, 'torch', TorchProxy()):
                        module.run(args)
                        prefix = 'rankup_cls' if task == 'classification' else 'rankup'
                        params = root/'tuning'/dataset/f'{prefix}_gaussian_best_params.json'
                        self.assertTrue(params.is_file())
                        best = json.loads(params.read_text())
                        self.assertEqual(best['trial_counts']['complete'], 1)
                        self.assertTrue(best['search_space'])
                        tune_config = json.loads((root/'tuning'/dataset/f'{prefix}_gaussian_run_config.json').read_text())
                        self.assertEqual(tune_config['arguments']['n_trials'], 1)
                        self.assertEqual(tune_config['stage'], 'tuning')
                        # The test split is intentionally created only after tuning.
                        np.savez(folder/'test.npz', X=rng.normal(size=(10,4)).astype('float32'), y=y)
                        args.params_file = params
                        args.tune_only = False
                        args.seed = 123
                        args.output_root = root/'seed_123'
                        with mock.patch.object(module.optuna, 'create_study', side_effect=AssertionError('Retuned')):
                            module.run(args)
                        metrics = json.loads((root/'seed_123'/dataset/f'{prefix}_gaussian_metrics.json').read_text())
                        self.assertEqual(metrics['seed'], 123)
                        self.assertEqual(metrics['primary_metric'], metric)
                        self.assertTrue(np.isfinite(metrics[f'train_{metric}']))
                        self.assertTrue(np.isfinite(metrics[f'validation_{metric}']))
                        seed_dir = root/'seed_123'/dataset
                        config = json.loads((seed_dir/f'{prefix}_gaussian_run_config.json').read_text())
                        self.assertEqual(config['stage'], 'training')
                        self.assertEqual(config['arguments']['seed'], 123)
                        checkpoint = torch.load(seed_dir/f'{prefix}_gaussian_model.pt', map_location='cpu')
                        self.assertTrue(checkpoint['state_dict'])
                        self.assertEqual(checkpoint['seed'], 123)
                        if task == 'classification':
                            self.assertEqual(checkpoint['classes'], manifest['classes'])
                        else:
                            self.assertIn('target_scaler', checkpoint)
                        self.assertEqual(metrics['weak_strength'], best['weak_strength'])
                        self.assertEqual(metrics['strong_strength'], best['strong_strength'])

    def test_tabm_forward_and_backward(self):
        sys.path.insert(0, str(ROOT.parent))
        ordinal = importlib.import_module('experiments.rankup.gaussian')
        classification = importlib.import_module('experiments.rankup_classification.gaussian')
        x = torch.randn(4, 5)
        for model in (ordinal.TabMRankUp(5), classification.TabMClassifier(5, 3)):
            output = model(x)
            tensors = output if isinstance(output, tuple) else (output,)
            self.assertTrue(all(t.shape[0] == 4 and t.ndim == 3 for t in tensors))
            sum(t.square().mean() for t in tensors).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_tabcutmix_plus_respects_zero_and_full_strength(self):
        sys.path.insert(0, str(ROOT.parent))
        for family in ('rankup', 'rankup_classification'):
            module = importlib.import_module(f'experiments.{family}.tabcutmix_plus')
            x = torch.tensor([[1., 1., 1., 1.], [2., 2., 2., 2.]])
            groups = [torch.tensor([0, 1]), torch.tensor([2, 3])]
            args = (x, x, torch.tensor([0, 0]), torch.ones(2), 0.8, None, groups)
            weak = module.augment(args[0], 0.0, *args[1:])
            strong = module.augment(args[0], 1.0, *args[1:])
            self.assertTrue(torch.equal(weak, x))
            self.assertTrue(torch.equal(strong, x.flip(0)))

    def test_labeled_loader_restarts_after_each_pass(self):
        for family in ('rankup', 'rankup_classification', 'regression'):
            for path in sorted((ROOT / family).glob('*.py')):
                if path.name == '__init__.py':
                    continue
                fns = get_functions(path, 'repeat_loader')
                class Loader:
                    def __init__(self): self.calls = 0
                    def __iter__(self):
                        self.calls += 1
                        yield self.calls
                loader = Loader()
                batches = fns['repeat_loader'](loader)
                self.assertEqual([next(batches), next(batches)], [1, 2], str(path))


if __name__ == '__main__':
    unittest.main()
