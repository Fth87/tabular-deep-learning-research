import unittest
import numpy as np
import torch
from augmentasi_adaptif import AdaptiveAug, operator_probabilities


class TestAdaptive(unittest.TestCase):
    def test_probabilities_and_output(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(12, 4)).astype(np.float32)
        y = np.arange(12, dtype=np.float32)
        p = operator_probabilities(x[:4], x[4:])
        self.assertAlmostEqual(sum(p[k] for k in ('random', 'guided', 'group')), 1.)
        self.assertTrue(all(0 <= p[k] <= 1 for k in ('random', 'guided', 'group')))
        aug = AdaptiveAug(x[:4], y[:4], x[4:], False)
        batch = torch.tensor(x[:4]); pool = torch.tensor(x)
        weak = aug(batch, .1, pool, None, None, .8, None, None)
        self.assertTrue(torch.equal(weak, batch))
        strong = aug(batch, .3, pool, None, None, .8, None, None)
        self.assertEqual(strong.shape, batch.shape)
        self.assertTrue(torch.isfinite(strong).all())
        self.assertEqual(sum(aug.counts.values()), len(batch))


if __name__ == '__main__':
    unittest.main()
