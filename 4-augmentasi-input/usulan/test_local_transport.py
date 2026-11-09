import math
import unittest
import torch
from local_transport import LocalTransport


class LocalTransportTests(unittest.TestCase):
    def test_bounds_and_no_confident_donor(self):
        x = torch.tensor([[0., 0.], [10., 10.], [0.1, 0.1]])
        labels = torch.tensor([0, 0, 1])
        confidence = torch.tensor([.9, .9, .9])
        aug = LocalTransport(radius=.5)
        out = aug(x, .3, None, labels, confidence, .8, None, None)
        self.assertTrue(torch.equal(out[2], x[2]))
        self.assertGreater((out[0] - x[0]).abs().sum().item(), 0)
        self.assertLessEqual((torch.linalg.vector_norm(out - x, dim=1) / math.sqrt(2)).max().item(), .150001)
        self.assertAlmostEqual(aug.diagnostics()['eligible_rate'], 2 / 3)

    def test_unreliable_samples_and_singleton_are_unchanged(self):
        x = torch.tensor([[0., 0.], [1., 1.]])
        aug = LocalTransport()
        out = aug(x, .3, None, torch.tensor([0, 0]), torch.tensor([.5, .9]), .8, None, None)
        self.assertTrue(torch.equal(out, x))
        self.assertTrue(torch.equal(aug(x[:1], .3, None, None, None, .8, None, None), x[:1]))

    def test_regression_without_pseudo_labels(self):
        x = torch.tensor([[0., 0.], [1., 1.]])
        out = LocalTransport(radius=.5)(x, .3, None, None, None, .8, None, None)
        self.assertTrue(torch.all(out[0] > x[0]))
        self.assertTrue(torch.all(out[1] < x[1]))


if __name__ == '__main__':
    unittest.main()
