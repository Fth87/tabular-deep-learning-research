"""Bounded local interpolation for numerical tabular SSL."""
import math
import torch


class LocalTransport:
    def __init__(self, radius=0.5, confidence=0.8):
        if radius <= 0 or not 0 <= confidence <= 1:
            raise ValueError('Invalid radius or confidence')
        self.radius = radius
        self.confidence = confidence
        self.count = 0
        self.eligible = None
        self.clipped = None
        self.displacement_sum = None
        self.max_displacement = None

    def __call__(self, x, strength, train_features, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        if not 0 <= strength <= 1:
            raise ValueError('strength must be between 0 and 1')
        if (pseudo_labels is None) != (confidence is None):
            raise ValueError('labels and confidence must both be provided or both be absent')
        if len(x) < 2:
            return x.clone()
        distances = torch.cdist(x, x) / math.sqrt(x.shape[1])
        distances.fill_diagonal_(float('inf'))
        if pseudo_labels is not None:
            reliable = confidence >= self.confidence
            valid = (pseudo_labels[:, None] == pseudo_labels[None, :]) & reliable[:, None] & reliable[None, :]
            distances = distances.masked_fill(~valid, float('inf'))
        nearest_dist, indices = distances.min(dim=1)
        usable = torch.isfinite(nearest_dist)
        scale = torch.clamp(self.radius / nearest_dist.clamp_min(1e-12), max=1.0)
        scale = torch.where(usable, scale, torch.zeros_like(scale))
        delta = strength * scale[:, None] * (x[indices] - x)
        delta_rms = torch.linalg.vector_norm(delta, dim=1) / math.sqrt(x.shape[1])
        self.count += len(x)
        eligible = usable.sum().detach()
        clipped = (usable & (nearest_dist > self.radius)).sum().detach()
        total = delta_rms.sum().detach()
        maximum = delta_rms.max().detach()
        self.eligible = eligible if self.eligible is None else self.eligible + eligible
        self.clipped = clipped if self.clipped is None else self.clipped + clipped
        self.displacement_sum = total if self.displacement_sum is None else self.displacement_sum + total
        self.max_displacement = maximum if self.max_displacement is None else torch.maximum(self.max_displacement, maximum)
        return x + delta

    def diagnostics(self):
        if not self.count:
            return {'n_views': 0}
        return {'n_views': self.count, 'eligible_rate': float(self.eligible.cpu()) / self.count,
                'clipped_rate': float(self.clipped.cpu()) / self.count,
                'mean_delta_rms': float(self.displacement_sum.cpu()) / self.count,
                'max_delta_rms': float(self.max_displacement.cpu())}
