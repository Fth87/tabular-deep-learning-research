"""Round 3: limit large continuous replacements on ranking tasks only."""
import numpy as np
import torch
from augmentasi_terarah import FeatureGuidedAug, feature_weights


class Candidate:
    def __init__(self, x_labeled, y_labeled, x_unlabeled, classification):
        self.base = FeatureGuidedAug(feature_weights(x_labeled, y_labeled, classification))
        self.classification = classification
        train = np.concatenate([x_labeled, x_unlabeled])
        self.discrete = np.array([np.unique(train[:, i]).size <= 8 for i in range(train.shape[1])])
        self.count = 0
        self.clipped = 0
        self.sum_rms = 0.0

    def __call__(self, x, strength, train, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        out = self.base(x, strength, train, pseudo_labels, confidence,
                        donor_confidence, quantile_edges, feature_groups)
        if self.classification:
            self.count += len(x)
            self.sum_rms += float(((out - x).square().mean(dim=1).sqrt().sum()).item())
            return out
        discrete = torch.as_tensor(self.discrete, device=x.device).unsqueeze(0)
        delta = out - x
        continuous = torch.where(discrete, torch.zeros_like(delta), delta)
        rms = continuous.square().mean(dim=1, keepdim=True).sqrt()
        scale = (.25 / rms.clamp_min(1e-12)).clamp(max=1.)
        mixed = torch.where(discrete, out, x + continuous * scale)
        self.count += len(x)
        self.clipped += int((rms > .25).sum().item())
        self.sum_rms += float(((mixed - x).square().mean(dim=1).sqrt().sum()).item())
        return mixed

    def diagnostics(self):
        return {'base': self.base.diagnostics(), 'discrete_features': int(self.discrete.sum()),
                'clipped_rate_all_views': self.clipped / self.count if self.count else None,
                'mean_delta_rms_all_views': self.sum_rms / self.count if self.count else None}
