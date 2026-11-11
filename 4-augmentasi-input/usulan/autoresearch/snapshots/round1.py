"""Round 1: conditional guided donor with discrete-safe shrinkage."""
import numpy as np
import torch
from augmentasi_terarah import FeatureGuidedAug, feature_weights


class Candidate:
    def __init__(self, x_labeled, y_labeled, x_unlabeled, classification):
        self.base = FeatureGuidedAug(feature_weights(x_labeled, y_labeled, classification))
        train = np.concatenate([x_labeled, x_unlabeled])
        self.discrete = np.array([np.unique(train[:, i]).size <= 8 for i in range(train.shape[1])])
        self.sum_rms = 0.0
        self.count = 0

    def __call__(self, x, strength, train, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        out = self.base(x, strength, train, pseudo_labels, confidence,
                        donor_confidence, quantile_edges, feature_groups)
        is_discrete = torch.as_tensor(self.discrete, device=x.device).unsqueeze(0)
        mixed = torch.where(is_discrete, out, x + .5 * (out - x))
        self.count += len(x)
        self.sum_rms += float(((mixed - x).square().mean(dim=1).sqrt().sum()).item())
        return mixed

    def diagnostics(self):
        return {'base': self.base.diagnostics(), 'discrete_features': int(self.discrete.sum()),
                'mean_delta_rms_all_views': self.sum_rms / self.count if self.count else None}
