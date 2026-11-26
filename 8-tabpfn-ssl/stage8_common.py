"""Helper yang disalin oleh buat_notebook.py dari tahap 5."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)


def load(root, name):
    with np.load(root / f'{name}.npz', allow_pickle=False) as split:
        return split['X'].astype(np.float32), (split['y'] if 'y' in split else None)


def scores(task, y, pred, levels=None):
    if task == 'classification':
        return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
                'accuracy': float(accuracy_score(y, pred))}
    result = {'mse': float(mean_squared_error(y, pred)), 'mae': float(mean_absolute_error(y, pred)),
              'r2': float(r2_score(y, pred)), 'spearman': float(spearmanr(y, pred).statistic)}
    if task == 'ordinal_regression':
        rounded = levels[np.abs(pred.reshape(-1, 1) - levels).argmin(axis=1)]
        result.update(qwk=float(cohen_kappa_score(y, rounded, labels=levels, weights='quadratic')),
                      ordinal_mae=float(mean_absolute_error(y, rounded)),
                      accuracy=float(accuracy_score(y, rounded)))
    return result


def same_rank_donors(pseudo_labels: torch.Tensor, confidence: torch.Tensor, cutoff: float):
    n = len(pseudo_labels)
    eligible = (pseudo_labels[:, None].eq(pseudo_labels[None, :])
                & confidence[None, :].ge(cutoff)
                & ~torch.eye(n, dtype=torch.bool, device=pseudo_labels.device))
    has_donor = eligible.any(dim=1)
    priorities = torch.rand((n, n), device=pseudo_labels.device)
    donors = priorities.masked_fill(~eligible, -1.0).argmax(dim=1)
    return torch.where(has_donor, donors, torch.arange(n, device=pseudo_labels.device)), has_donor


def rotate(h: torch.Tensor, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """h' = ‖h‖ (c·v + √(1−c²)·û), v = h/‖h‖, û = bagian u yang tegak lurus v (dinormalkan). Norma tetap."""
    norm = h.norm(dim=-1, keepdim=True)
    v = h / norm.clamp_min(1e-12)
    u = u - (u * v).sum(-1, keepdim=True) * v
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rotated = norm * (c * v + (1 - c ** 2).clamp_min(0).sqrt() * u)
    return torch.where(norm > 1e-6, rotated, h)


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
