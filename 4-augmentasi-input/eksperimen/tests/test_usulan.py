"""Self-check for the block-Gibbs operator; needs PyTorch, run with modal_usulan.py --action check."""
import unittest

import torch

from experiments.usulan import block_gibbs, gibbs_chain


class BlockGibbsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Two tight clusters: the kept features identify the cluster of the row.
        self.low = torch.randn(500, 6) * 0.01 - 3
        self.high = torch.randn(500, 6) * 0.01 + 3
        self.train = torch.cat([self.low, self.high])

    def test_kept_cells_unchanged_and_block_comes_from_one_real_row(self):
        x = self.train[:32]
        out = block_gibbs(x, 0.5, self.train, k=5)
        self.assertEqual(((out != x).sum(1) > 0).sum().item(), 32)
        self.assertTrue(((out != x).sum(1) <= 3).all())
        for row, original in zip(out, x):
            changed = row != original
            donors = (self.train[:, changed] == row[changed]).all(1)
            self.assertGreater(donors.sum().item(), 0)

    def test_never_returns_the_row_itself_and_conditions_on_kept_features(self):
        x = self.low[:32]
        out = block_gibbs(x, 0.5, self.train, k=5)
        self.assertFalse((out == x).all(1).any())
        # A marginal draw would land in the other cluster half the time.
        self.assertTrue((out < 0).all())

    def test_full_block_is_a_marginal_draw(self):
        x = self.low[:64]
        out = block_gibbs(x, 1.0, self.train, k=1000)
        self.assertGreater((out[:, 0] > 0).float().mean().item(), 0.2)

    def test_chain_stays_in_column_support_and_moves_farther(self):
        x = self.train[:200]
        one = block_gibbs(x, 0.5, self.train, k=50)
        three = gibbs_chain(x, 0.5, self.train, k=50, steps=3)
        for column in range(x.shape[1]):
            self.assertTrue(torch.isin(three[:, column], self.train[:, column]).all())
        self.assertGreater(((three - x) ** 2).sum(), ((one - x) ** 2).sum())


if __name__ == '__main__':
    unittest.main()
