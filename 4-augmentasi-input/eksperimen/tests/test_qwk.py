"""Check ordinal score conversion and QWK without local PyTorch."""
import ast
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'rankup'


def load_scoring(path):
    tree = ast.parse(path.read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'ordinal_predictions', 'score_qwk'}]
    namespace = {'np': np, 'cohen_kappa_score': cohen_kappa_score}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class QWKTests(unittest.TestCase):
    def test_ordinal_levels_and_qwk_for_each_augmentation(self):
        levels = np.array([0, 1, 2, 3, 4])
        targets = np.array([0, 1, 2, 3, 4])
        scores = np.array([-3., 1.2, 1.9, 2.8, 9.])
        expected = np.array([0, 1, 2, 3, 4])
        for path in sorted(ROOT.glob('*.py')):
            if path.name == '__init__.py':
                continue
            with self.subTest(path=path.name):
                funcs = load_scoring(path)
                np.testing.assert_array_equal(funcs['ordinal_predictions'](scores, levels), expected)
                self.assertAlmostEqual(funcs['score_qwk'](targets, scores, levels), 1.0)
                inaccurate = np.array([4., 3., 2., 1., 0.])
                self.assertAlmostEqual(funcs['score_qwk'](targets, inaccurate, levels),
                    cohen_kappa_score(targets, inaccurate, labels=levels, weights='quadratic'))
                source = path.read_text()
                self.assertIn('direction="maximize"', source)
                self.assertIn('if val_qwk > best_qwk:', source)
                self.assertIn('"primary_metric": "qwk"', source)


if __name__ == '__main__':
    unittest.main()
