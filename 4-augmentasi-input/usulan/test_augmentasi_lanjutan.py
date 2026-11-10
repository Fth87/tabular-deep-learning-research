import unittest
import torch
from augmentasi_lanjutan import AugmentasiLanjutan


class TestAugmentasi(unittest.TestCase):
    def test_zero_strength_and_singleton(self):
        torch.manual_seed(1)
        x = torch.tensor([[0., 0.], [2., 3.], [4., 5.]])
        for kind in ('correlated_noise', 'gated_cutmix', 'hybrid'):
            aug = AugmentasiLanjutan(kind)
            out = aug(x, 0, x, None, None, .8, None, None)
            self.assertTrue(torch.equal(out, x))
            out = aug(x[:1], .3, x, None, None, .8, None, None)
            self.assertEqual(out.shape, (1, 2))
            self.assertTrue(torch.isfinite(out).all())

    def test_gating_and_self_donor(self):
        x = torch.tensor([[0., 0.], [2., 2.]])
        aug = AugmentasiLanjutan('gated_cutmix')
        label, conf = torch.tensor([0, 1]), torch.ones(2)
        self.assertTrue(torch.equal(aug(x, 1, x, label, conf, .8, None, None), x))
        label = torch.tensor([0, 0])
        self.assertTrue(torch.equal(aug(x, 1, x, label, conf, .8, None, None), x.flip(0)))


if __name__ == '__main__':
    unittest.main()
