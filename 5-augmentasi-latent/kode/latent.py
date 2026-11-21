"""Augmentasi weak/strong di ruang latent TabM untuk RankUp.

Alur satu langkah training (loop RankUp lama, tidak diubah):

    x_u ──augment(x_u, 'weak')──► x_weak ──► backbone ──► h ──[perturb weak]──► head ──► pseudo-label
    x_u ──augment(x_u, 'strong')─► x_strong ─► backbone ─► h ─[perturb strong]─► head ──► loss konsistensi

`augment` (augmentasi input, opsional) dan hook di output backbone (augmentasi latent, opsional)
diatur oleh satu config. Data berlabel, validation, dan test tidak pernah diganggu.
"""
from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rankup.tabcutmix import same_rank_donors  # aturan donor yang sama dengan TabCutMix input
from experiments.usulan import FAMILIES, load, scores

LATENT_OPS = ('gaussian', 'zero_out', 'scarf', 'tabcutmix', 'dropout', 'fnoise', 'angular', 'advrot', 'headrot')


def rotate(h: torch.Tensor, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """h' = ‖h‖ (c·v + √(1−c²)·û), v = h/‖h‖, û = bagian u yang tegak lurus v (dinormalkan). Norma tetap."""
    norm = h.norm(dim=-1, keepdim=True)
    v = h / norm.clamp_min(1e-12)
    u = u - (u * v).sum(-1, keepdim=True) * v
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rotated = norm * (c * v + (1 - c ** 2).clamp_min(0).sqrt() * u)
    return torch.where(norm > 1e-6, rotated, h)  # vektor nol tidak punya arah, dibiarkan


def angular(h: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Putar h ke arah acak sehingga cos(h, h') = c ~ U(low, high) tepat, norma tetap (ide pembimbing)."""
    u = torch.randn_like(h)  # urutan RNG sama dengan versi lama: hasil run angular lama tetap bisa diulang
    return rotate(h, u, torch.empty_like(h[..., :1]).uniform_(low, high))


def head_view(weight: torch.Tensor, rank: bool) -> torch.Tensor:
    """Bobot head (k, d, out) yang benar-benar dilihat loss. RankUp: output [regresi, ARC0, ARC1] dan
    peluang i > j hanya bergantung pada r_i − r_j, dengan r = h·a, a = W[..., 2] − W[..., 1]."""
    weight = weight.detach()
    return (weight[..., 2] - weight[..., 1])[..., None] if rank else weight


def head_direction(h: torch.Tensor, weight: torch.Tensor, rank: bool, adversarial: bool) -> torch.Tensor:
    """Arah di ruang h yang paling mengubah output head linear (closed form, tanpa backward tambahan).

    rank: skor r = h·a didorong ke median batch, sehingga urutan antar sampel paling sulit dibedakan.
    klasifikasi: dari bobot kelas teratas ke kelas kedua (batas keputusan terdekat).
    adversarial=False: tanda arah diacak (kontrol 'headrot': sadar-head tapi tidak adversarial).
    """
    w = head_view(weight, rank)                                       # (k, d, o)
    if rank:
        a = w[..., 0]                                                 # (k, d)
        r = (h * a).sum(-1)                                           # (B, k)
        sign = -torch.sign(r - r.median(dim=0).values)
        direction = a.expand_as(h)
    else:
        top = torch.einsum('bkd,kdo->bko', h, w).topk(2, dim=-1).indices
        wt, member = w.transpose(1, 2), torch.arange(w.shape[0], device=h.device)  # (k, o, d)
        direction = wt[member, top[..., 1]] - wt[member, top[..., 0]]  # (B, k, d)
        sign = torch.ones(h.shape[:2], device=h.device)
    if not adversarial:
        sign = torch.randint(0, 2, h.shape[:2], device=h.device) * 2.0 - 1
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)        # sampel tepat di median
    return sign[..., None] * direction


def head_shift(h: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, rank: bool) -> torch.Tensor:
    """‖Δ output head‖ / ‖output head‖ tanpa bias: seberapa besar gangguan yang benar-benar dilihat head."""
    w = head_view(weight, rank)
    before = torch.einsum('bkd,kdo->bko', h, w)
    after = torch.einsum('bkd,kdo->bko', out, w)
    return (after - before).norm(dim=-1).mean() / before.norm(dim=-1).mean().clamp_min(1e-12)


def perturb(op: str, h: torch.Tensor, strength, pseudo_labels=None, confidence=None,
            donor_confidence: float = 0.8, head=None, rank: bool = True) -> torch.Tensor:
    """h: (batch, submodel, dim). strength: angka; (low, high) cosine untuk 'angular';
    (low, high) angular distance d ∈ [0, 1] (θ = π·d) untuk 'advrot'/'headrot', yang juga butuh head (k, d, out)."""
    chosen = torch.rand_like(h) < strength if op in ('zero_out', 'scarf', 'tabcutmix') else None
    if op == 'gaussian':  # skala noise ikut sebaran tiap unit di batch, jadi strength tanpa satuan
        return h + strength * h.std(dim=0, keepdim=True, unbiased=False) * torch.randn_like(h)
    if op == 'zero_out':
        return h.masked_fill(chosen, 0.0)
    if op == 'scarf':  # unit terpilih diambil dari baris acak lain di batch
        return torch.where(chosen, h.gather(0, torch.randint(len(h), h.shape, device=h.device)), h)
    if op == 'tabcutmix':  # unit terpilih diambil dari baris dengan pseudo-label sama
        donors, has_donor = same_rank_donors(pseudo_labels, confidence, donor_confidence)
        return torch.where(chosen & has_donor[:, None, None], h[donors], h)
    if op == 'dropout':  # UniMatch / CCT
        return F.dropout(h, p=strength, training=True)
    if op == 'fnoise':  # CCT F-Noise: z + z·U(−s, s)
        return h + h * (2 * torch.rand_like(h) - 1) * strength
    if op == 'angular':
        return angular(h, *strength)
    if op in ('advrot', 'headrot'):
        d = torch.empty_like(h[..., :1]).uniform_(*strength)
        return rotate(h, head_direction(h, head, rank, op == 'advrot'), torch.cos(math.pi * d))
    raise ValueError(f'operator latent tidak dikenal: {op!r}')


def latent_model(base: type, config: dict, marked: dict, cosines: dict) -> type:
    """Subclass model RankUp: output backbone diganggu hanya untuk tensor weak/strong yang ditandai.

    config['latent_at'] = 'pre_relu': gangguan dipasang di output Linear blok terakhir, sebelum ReLU dan
    dropout, sehingga hasilnya tetap ≥ 0 seperti yang biasa dilihat head. Default: output backbone.
    """
    rank = base.__name__ != 'TabMClassifier'  # TabMRankUp: head [regresi, ARC0, ARC1]
    if config.get('latent_at') == 'pre_relu' and config.get('latent_op') in ('advrot', 'headrot'):
        raise ValueError('advrot/headrot menghitung arah dari head, jadi harus di output backbone')

    class Latent(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._role = None
            backbone = self.model.backbone
            target = backbone.blocks[-1][0] if config.get('latent_at') == 'pre_relu' else backbone
            target.register_forward_hook(self._hook)

        def _hook(self, _module, _inputs, h):
            strength = config.get(f'latent_{self._role}') if self._role else None
            if not strength:
                return None
            _, pseudo_labels, confidence = marked[f'{self._role}_context']
            head = getattr(self.model, 'output', None)  # LinearEnsemble TabM, bobot (k, d, out)
            weight = head.weight if head is not None else None
            out = perturb(config['latent_op'], h, strength, pseudo_labels, confidence, head=weight, rank=rank)
            cosines[self._role].append(F.cosine_similarity(out.detach(), h.detach(), dim=-1).mean())
            if weight is not None and config.get('latent_at') != 'pre_relu':
                cosines[f'shift_{self._role}'].append(head_shift(h.detach(), out.detach(), weight, rank))
            return out

        def forward(self, x):
            self._role = next((role for role in ('weak', 'strong') if marked.get(role) is x), None) \
                if self.training else None
            try:
                return super().forward(x)
            finally:
                self._role = None
    return Latent


def run_one(data_root, dataset: str, label_percent: int, config: dict, seed: int, epochs: int = 120) -> dict:
    """Satu run RankUp dengan protokol E2 (tanpa Optuna). Contoh config ada di GRID (modal_latent.py)."""
    started = time.time()
    root = Path(data_root) / f'label_{label_percent:02d}' / dataset
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    task = manifest['task_type']
    # Loop host: modul operator input, atau modul TabCutMix (satu-satunya yang menghitung pseudo-label di semua tugas).
    module = importlib.import_module(f'experiments.{FAMILIES[task]}.{config.get("input_op") or "tabcutmix"}')
    model_name = 'TabMClassifier' if task == 'classification' else 'TabMRankUp'
    base_model, base_augment = getattr(module, model_name), module.augment
    marked, cosines = {}, {'weak': [], 'strong': [], 'shift_weak': [], 'shift_strong': []}

    def augment(x, role, *rest):
        # fit_model memanggil augment(x, weak_strength, ...); di sini weak_strength = 'weak'.
        strength = config.get(f'input_{role}', 0.0)
        out = base_augment(x, strength, *rest) if strength > 0 else x.clone()
        marked[role], marked[f'{role}_context'] = out, rest[:3]  # (train_features, pseudo_labels, confidence)
        return out

    setattr(module, model_name, latent_model(base_model, config, marked, cosines))
    module.augment = augment
    try:
        result = _fit_and_score(module, base_model, root, manifest, task, config, seed, epochs)
    finally:
        setattr(module, model_name, base_model)
        module.augment = base_augment
    cos = {role: float(torch.stack(values).mean()) for role, values in cosines.items() if values}
    return {'dataset': dataset, 'label_percent': label_percent, 'config': config, 'seed': seed, 'task': task,
            **result, 'cos': cos, 'seconds': time.time() - started}


def _fit_and_score(module, base_model, root, manifest, task, config, seed, epochs) -> dict:
    """Sama dengan usulan.run_one (E2): hiperparameter, early stopping di validation, test dibuka sekali."""
    from sklearn.preprocessing import StandardScaler

    x_lab, y_lab = load(root, 'train_labeled')
    x_unlab, _ = load(root, 'train_unlabeled')
    x_val, y_val = load(root, 'validation')
    x_test, y_test = load(root, 'test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = SimpleNamespace(batch_size=32, max_steps=100, patience=15, rank_weight=0.2, confidence_cutoff=0.95,
                           unlabeled_weight=config.get('unlabeled_weight', 0.2), donor_confidence=0.8,
                           learning_rate=2e-3, seed=seed)
    levels = None
    if task == 'classification':
        ids = {str(label): index for index, label in enumerate(manifest['classes'])}
        y_lab, y_val, y_test = (np.asarray([ids[v] for v in np.asarray(y).astype(str)], dtype=np.int64)
                                for y in (y_lab, y_val, y_test))
        _, best_epoch, state, history = module.fit_model(
            args, x_lab, y_lab, x_unlab, x_val, y_val, len(ids), 'macro_f1', 'weak', 'strong', epochs, device)
        model = base_model(x_lab.shape[1], len(ids)).to(device)
        model.load_state_dict(state)
        predict = lambda x: module.predict(model, x, device)
    else:
        y_lab, y_val, y_test = (np.asarray(v, dtype=np.float32).reshape(-1) for v in (y_lab, y_val, y_test))
        if task == 'ordinal_regression':
            levels = args.ordinal_levels = np.asarray(manifest['target_values'], dtype=np.int64)
        _, best_epoch, state, history = module.fit_model(
            args, x_lab, y_lab, x_unlab, x_val, y_val, 'weak', 'strong', epochs)
        scaler = StandardScaler().fit(y_lab.reshape(-1, 1))
        model = base_model(x_lab.shape[1]).to(device)
        model.load_state_dict(state)
        predict = lambda x: module.predict(model, x, scaler, device)
    return {'validation': scores(task, y_val, predict(x_val), levels),
            'test': scores(task, y_test, predict(x_test), levels),
            'best_epoch': int(best_epoch), 'epochs_run': len(history),
            'accept_rate_last5': float(np.mean([row['pseudo_accept_rate'] for row in history[-5:]])),
            'device': str(device)}
