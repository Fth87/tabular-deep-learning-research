import unittest
import torch
from conditional_aug import ConditionalAug, soft_rank_loss


class ConditionalTests(unittest.TestCase):
    def test_weak_and_singleton_fallback(self):
        for grouped in (False, True):
            aug = ConditionalAug(grouped=grouped)
            x = torch.tensor([[1., 2., 3.]])
            self.assertTrue(torch.equal(aug(x, .1, x, None, None, .8, None, None), x))
            self.assertTrue(torch.equal(aug(x, .3, x, None, None, .8, None, None), x))

    def test_valid_output_and_unmodified_features(self):
        torch.manual_seed(7)
        x = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.], [2., 2., 2., 2.]])
        for grouped in (False, True):
            out = ConditionalAug(grouped=grouped)(x, .5, x, None, None, .8, None, None)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(((out != x).sum(dim=1) <= 2).all())

    def test_soft_pairwise_weights_and_gradients(self):
        class Module:
            @staticmethod
            def pairwise_logits(logits):
                return logits[:, None, :] - logits[None, :, :]
        logits = torch.tensor([[2., -2.], [-2., 2.]], requires_grad=True)
        loss, weight = soft_rank_loss(Module, logits.detach(), logits, .95)
        self.assertTrue(0 < weight < 1)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
