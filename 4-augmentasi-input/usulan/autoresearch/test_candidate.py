import unittest
import numpy as np
import torch
from candidate import Candidate


class CandidateTests(unittest.TestCase):
    def test_weak_and_discrete_values(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(20, 4)).astype(np.float32)
        x[:, 0] = np.arange(20) % 2
        y = np.arange(20, dtype=np.float32)
        aug = Candidate(x[:10], y[:10], x[10:], False)
        a = torch.from_numpy(x)
        weak = aug(a, .1, a, None, None, .8, None, None)
        self.assertTrue(torch.equal(weak, a))
        strong = aug(a, .3, a, None, None, .8, None, None)
        self.assertTrue(torch.isfinite(strong).all())
        self.assertTrue(set(strong[:, 0].tolist()) <= {0., 1.})
        self.assertTrue(((strong != a).sum(dim=1) <= 1).all())
        continuous_delta = strong[:, 1:] - a[:, 1:]
        self.assertTrue(((continuous_delta.square().sum(dim=1) / a.shape[1]).sqrt() <= .250001).all())

    def test_classification_uses_base_operator(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(20, 4)).astype(np.float32)
        y = np.arange(20) % 2
        aug = Candidate(x[:10], y[:10], x[10:], True)
        a = torch.from_numpy(x)
        torch.manual_seed(7)
        baseline = aug.base(a, .3, a, None, None, .8, None, None)
        torch.manual_seed(7)
        result = aug(a, .3, a, None, None, .8, None, None)
        self.assertTrue(torch.equal(baseline, result))


if __name__ == '__main__':
    unittest.main()
