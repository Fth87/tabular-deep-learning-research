"""Self-check operator latent dan hook. Jalankan: lihat README.md bagian "Menjalankan"."""
import math
import unittest

import torch
from torch import nn

from latent import LATENT_OPS, angular, head_shift, latent_model, perturb


class Base(nn.Module):
    """Tiruan TabM: model.backbone lalu head; cukup untuk menguji hook."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Identity()

    def forward(self, x):
        return self.model.backbone(x)


class LatentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.h = torch.relu(torch.randn(32, 4, 64))  # output TabM habis ReLU: semua ≥ 0
        self.labels = torch.arange(32) % 3
        self.conf = torch.ones(32)

    def test_every_op_keeps_shape_and_changes_values(self):
        weight = torch.randn(4, 64, 3)
        for op in LATENT_OPS:
            strength = {'angular': (0.3, 0.5), 'advrot': (0.1, 0.2), 'headrot': (0.1, 0.2)}.get(op, 0.5)
            out = perturb(op, self.h, strength, self.labels, self.conf, head=weight)
            self.assertEqual(out.shape, self.h.shape, op)
            self.assertTrue(torch.isfinite(out).all(), op)
            self.assertFalse(torch.equal(out, self.h), op)

    def test_angular_hits_target_cosine_and_keeps_norm(self):
        out = angular(self.h, 0.4, 0.4)
        cos = torch.nn.functional.cosine_similarity(out, self.h, dim=-1)
        self.assertTrue(torch.allclose(cos, torch.full_like(cos, 0.4), atol=1e-4))
        self.assertTrue(torch.allclose(out.norm(dim=-1), self.h.norm(dim=-1), atol=1e-4))

    def test_angular_leaves_zero_vector(self):
        h = self.h.clone()
        h[0] = 0
        self.assertTrue(torch.equal(angular(h, 0.3, 0.5)[0], h[0]))

    def test_advrot_hits_angle_keeps_norm_and_pushes_rank_scores_to_median(self):
        weight = torch.randn(4, 64, 3)
        a = weight[..., 2] - weight[..., 1]
        out = perturb('advrot', self.h, (0.1, 0.1), head=weight, rank=True)
        cos = torch.nn.functional.cosine_similarity(out, self.h, dim=-1)
        self.assertTrue(torch.allclose(cos, torch.full_like(cos, math.cos(0.1 * math.pi)), atol=1e-4))
        self.assertTrue(torch.allclose(out.norm(dim=-1), self.h.norm(dim=-1), atol=1e-3))
        # Sudut kecil (orde-1): setiap skor bergerak ke arah median batch.
        r = (self.h * a).sum(-1)
        moved = (perturb('advrot', self.h, (0.01, 0.01), head=weight, rank=True) * a).sum(-1) - r
        away = r - r.median(dim=0).values
        self.assertTrue((moved * away)[away != 0].lt(0).all())

    def test_advrot_moves_head_more_than_random_rotation_at_same_angle(self):
        h = torch.relu(torch.randn(32, 4, 512))  # dimensi asli TabM: arah acak hampir tegak lurus head
        weight = torch.randn(4, 512, 3)
        for rank in (True, False):
            w = weight if rank else torch.randn(4, 512, 10)
            adv = head_shift(h, perturb('advrot', h, (0.2, 0.2), head=w, rank=rank), w, rank)
            rand = head_shift(h, angular(h, math.cos(0.2 * math.pi), math.cos(0.2 * math.pi)), w, rank)
            self.assertGreater(adv, 3 * rand, rank)

    def test_classification_advrot_moves_toward_second_class(self):
        weight = torch.randn(4, 64, 10)
        logits = lambda x: torch.einsum('bkd,kdo->bko', x, weight)
        before = logits(self.h).topk(2, dim=-1)
        after = logits(perturb('advrot', self.h, (0.05, 0.05), head=weight, rank=False))
        margin = lambda z: z.gather(-1, before.indices[..., :1]) - z.gather(-1, before.indices[..., 1:])
        self.assertTrue((margin(after) < margin(logits(self.h))).all())

    def test_tabcutmix_takes_units_only_from_same_pseudo_label(self):
        h = self.labels.float()[:, None, None].expand(32, 4, 64)  # isi = labelnya sendiri
        out = perturb('tabcutmix', h, 1.0, self.labels, self.conf)
        self.assertTrue(torch.equal(out, h))

    def test_only_marked_tensors_in_training_are_perturbed(self):
        marked, cosines = {}, {'weak': [], 'strong': []}
        config = {'latent_op': 'gaussian', 'latent_weak': 0.0, 'latent_strong': 0.5}
        model = latent_model(Base, config, marked, cosines)()
        x = self.h.clone()
        marked.update(weak=x.clone(), strong=x.clone(), weak_context=(None, None, None),
                      strong_context=(None, None, None))
        self.assertTrue(torch.equal(model(x), x))                  # data berlabel: tidak diganggu
        self.assertTrue(torch.equal(model(marked['weak']), x))     # latent_weak = 0: identitas
        self.assertFalse(torch.equal(model(marked['strong']), x))  # strong diganggu
        self.assertEqual((len(cosines['weak']), len(cosines['strong'])), (0, 1))
        model.eval()
        self.assertTrue(torch.equal(model(marked['strong']), x))   # evaluasi: tidak diganggu

    def test_pre_relu_hook_perturbs_before_relu(self):
        class Blocks(Base):  # tiruan backbone TabM: blocks[-1] = (Linear, ReLU)
            def __init__(self):
                super().__init__()
                self.model.backbone = nn.Module()
                self.model.backbone.blocks = nn.ModuleList([nn.Sequential(nn.Identity(), nn.ReLU())])
                self.model.backbone.forward = lambda x: self.model.backbone.blocks[0](x)

        marked, cosines = {}, {'weak': [], 'strong': []}
        config = {'latent_op': 'angular', 'latent_strong': (0.3, 0.5), 'latent_at': 'pre_relu'}
        model = latent_model(Blocks, config, marked, cosines)()
        marked.update(strong=self.h.clone(), strong_context=(None, None, None))
        out = model(marked['strong'])
        self.assertTrue((out >= 0).all())                 # ReLU tetap dijalankan setelah gangguan
        self.assertFalse(torch.equal(out, self.h))
        self.assertAlmostEqual(float(cosines['strong'][0]), 0.4, delta=0.1)  # cos diukur sebelum ReLU


if __name__ == '__main__':
    unittest.main()
