import unittest
import numpy as np
import torch
from augmentasi_terarah import feature_weights, FeatureGuidedAug


class TestGuided(unittest.TestCase):
    def test_weights_and_no_donor(self):
        x = np.array([[0., 1., 1.], [1., 1., 1.], [2., 1., 1.], [3., 1., 1.]], dtype=np.float32)
        for y, classification in ((np.arange(4), False), (np.array([0, 0, 1, 1]), True)):
            weights = feature_weights(x, y, classification)
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue((weights > 0).all())
            self.assertLess(weights[0], weights[1])
            a = torch.tensor(x[:1])
            self.assertTrue(torch.equal(FeatureGuidedAug(weights)(a, .3, a, None, None, .8, None, None), a))

    def test_weak_view_and_mask_count(self):
        x = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.], [2., 2., 2., 2.]])
        aug = FeatureGuidedAug(np.ones(4, dtype=np.float32))
        self.assertTrue(torch.equal(aug(x, .1, x, None, None, .8, None, None), x))
        out = aug(x, .5, x, None, None, .8, None, None)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(((out != x).sum(dim=1) <= 2).all())


if __name__ == '__main__':
    unittest.main()
