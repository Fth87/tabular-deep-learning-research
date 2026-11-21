"""Self-check adaptor tabular SSR-RCUS. Butuh torch, tabm, dan paper_repo, jadi dijalankan di Modal:
modal run modal_rcus.py (action check). Semua test di CPU dengan data split asli NHANES 10%."""
import tempfile
import unittest

import numpy as np
import torch

import tabular_rcus as tr
from semilearn.algorithms.rcus.rankup_net import RankUp_Net


def batch(alg):
    lb, ulb = next(iter(alg.loader_dict['train_lb'])), next(iter(alg.loader_dict['train_ulb']))
    return {'x_lb': lb['x_lb'], 'y_lb': lb['y_lb'], 'idx_ulb': ulb['idx_ulb'],
            'x_ulb_w': ulb['x_ulb_w'], 'x_ulb_s': ulb['x_ulb_s']}


class AdaptorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.work = tempfile.mkdtemp()

    def build(self, variant, dataset='nhanes'):
        return tr.build(tr.write_config(dataset, 10, variant, 42, tr.DATA, self.work, num_workers=0))

    def test_algorithm_code_is_official(self):
        # "yang lain jangan dirubah": adaptor regresi/ordinal hanya mengganti data.
        own = lambda cls: [n for n in vars(cls) if not n.startswith('__')]
        self.assertEqual(own(tr.TabularData), ['set_dataset'])
        self.assertEqual(own(tr.TabularRCUS) + own(tr.TabularRankUp), [])
        self.assertIs(tr.TabularRCUS.train_step, tr.RCUS.train_step)
        self.assertIs(tr.TabularRankUp.train_step, tr.RankUp.train_step)
        self.assertIs(tr.TabularRCUS.arc_manifold_mixup, tr.RCUS.arc_manifold_mixup)
        # klasifikasi: hanya bagian khusus regresi yang diganti, sisanya tetap resmi
        self.assertEqual(own(tr.TabularCls), ['set_scaler', 'evaluate', 'weak_targets'])
        self.assertIs(tr.TabularRCUSCls.set_optimizer, tr.RCUS.set_optimizer)

    def test_tabm_net_matches_semilearn_interface(self):
        net = tr.TabMNet(7).eval()
        x = torch.randn(5, 7)
        out = net(x)
        self.assertEqual(out['logits'].shape, (5,))
        self.assertEqual(out['feat'].shape, (5, net.num_features))
        self.assertTrue(torch.allclose(net(net(x, only_feat=True), only_fc=True), out['logits']))
        arc = RankUp_Net(net)(x, use_arc=True, targets=torch.rand(5))
        self.assertEqual(arc['logits_arc'].shape, (5, 2))
        self.assertEqual(arc['logits_mat'].shape, (25, 2))

    def test_splits_eval_is_validation_and_ulb_includes_labeled(self):
        alg = self.build('rcus')
        root = tr.DATA / 'label_10' / 'nhanes'
        sizes = {name: len(tr.load(root, name)[0]) for name in ('train_labeled', 'train_unlabeled', 'validation', 'test')}
        d = alg.dataset_dict
        self.assertEqual(len(d['train_lb']), sizes['train_labeled'])
        self.assertEqual(len(d['train_ulb']), sizes['train_unlabeled'] + sizes['train_labeled'])
        self.assertEqual(len(d['eval']), sizes['validation'])
        self.assertEqual(len(d['test']), sizes['test'])
        self.assertTrue(np.isnan(d['train_ulb'].targets).all())          # tidak ada label unlabeled yang dipakai
        self.assertTrue(((d['train_lb'].targets >= 0) & (d['train_lb'].targets <= 1)).all())  # min-max resmi

    def test_e1_and_rankup_control_views_are_identical_and_loss_is_finite(self):
        for variant, cls in (('rcus', tr.TabularRCUS), ('rankup', tr.TabularRankUp)):
            alg = self.build(variant)
            self.assertIs(type(alg), cls)
            b = batch(alg)
            self.assertTrue(torch.equal(b['x_ulb_w'], b['x_ulb_s']))
            self.assertTrue(torch.isfinite(alg.train_step(**b)[0]['loss']), variant)

    def test_e3_input_operator_changes_only_strong_view(self):
        alg = self.build('rcus_input_gaussian')
        b = batch(alg)
        self.assertGreater((b['x_ulb_s'] != b['x_ulb_w']).float().mean(), 0.99)  # noise di semua fitur, weak asli
        self.assertTrue(torch.isfinite(alg.train_step(**b)[0]['loss']))

    def test_e2_latent_operator_hits_weak_and_strong_only(self):
        calls, original = [], tr.perturb
        tr.perturb = lambda op, h, strength: calls.append((op, list(strength), h.shape[0])) or original(op, h, strength)
        try:
            for variant, expected in (('rcus_angular', [[0.3, 0.5]]), ('rcus_angular_weak', [[0.7, 0.95], [0.3, 0.5]])):
                calls.clear()
                alg = self.build(variant)
                b = batch(alg)
                self.assertTrue(torch.isfinite(alg.train_step(**b)[0]['loss']))
                self.assertEqual([c[1] for c in calls], expected, variant)     # urutan resmi: weak lalu strong
                self.assertTrue(all(c[2] == len(b['x_ulb_s']) for c in calls))  # tidak pernah batch berlabel
                self.assertEqual(alg.model.backbone.tabm._forward_hooks, {})    # hook dilepas setelah langkah
        finally:
            tr.perturb = original


    def test_pendigits_uses_class_ids_and_classification_algorithms(self):
        for variant, cls in (('rcus', tr.TabularRCUSCls), ('rankup', tr.TabularRankUpCls)):
            alg = self.build(variant, 'pendigits')
            self.assertIs(type(alg), cls)
            y = alg.dataset_dict['train_lb'].targets
            self.assertTrue(np.array_equal(y, y.round()) and y.min() >= 0 and y.max() <= 9)  # tanpa min-max
            self.assertEqual(alg.model.backbone.regressor.out_features, 10)
            b = batch(alg)
            self.assertTrue(torch.isfinite(alg.train_step(**b)[0]['loss']), variant)

    def test_class_mixup_targets_and_mask(self):
        alg = self.build('rcus', 'pendigits')
        feats = torch.randn(6, alg.model.backbone.num_features)
        labels = torch.full((6,), 3)
        self.assertEqual(alg.class_mixup(feats, labels, torch.zeros(6)).item(), 0.0)  # tak ada pasangan yakin: 0
        # label sama di kedua sisi: target campuran = kelas itu, jadi loss = CE biasa pada fitur campuran
        np.random.seed(0)
        loss = alg.class_mixup(feats, labels)
        np.random.seed(0)
        lam = np.random.beta(alg.alpha, alg.alpha)
        lam = max(lam, 1 - lam)
        mixed = lam * feats.unsqueeze(0) + (1 - lam) * feats.unsqueeze(1)
        logits = alg.model.backbone.regressor(mixed).flatten(0, 1)
        expected = torch.nn.functional.cross_entropy(logits, torch.full((36,), 3))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-5))

    def test_latent_operator_on_classification(self):
        alg = self.build('rcus_angular', 'pendigits')
        self.assertIsInstance(alg, tr.TabularRCUSCls)
        self.assertTrue(torch.isfinite(alg.train_step(**batch(alg))[0]['loss']))


if __name__ == '__main__':
    unittest.main()
