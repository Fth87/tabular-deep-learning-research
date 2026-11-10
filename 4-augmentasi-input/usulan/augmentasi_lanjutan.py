"""Three exploratory tabular augmentations for the existing SSL pipeline."""
import math
import torch


class AugmentasiLanjutan:
    def __init__(self, kind, confidence=0.8):
        if kind not in ('correlated_noise', 'gated_cutmix', 'hybrid'):
            raise ValueError('Unknown augmentation')
        self.kind = kind
        self.confidence = confidence
        self.count = 0
        self.eligible = 0
        self.delta_sum = 0.0

    def __call__(self, x, strength, train_features, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        if not 0 <= strength <= 1:
            raise ValueError('Invalid strength')
        if (pseudo_labels is None) != (confidence is None):
            raise ValueError('Pseudo-label and confidence must be supplied together')
        n = len(x)
        if n == 0:
            return x.clone()
        idx1 = torch.randint(len(train_features), (n,), device=x.device)
        idx2 = torch.randint(len(train_features), (n,), device=x.device)
        # Differences preserve the empirical covariance structure in expectation.
        noise_view = x + strength * (train_features[idx1] - train_features[idx2]) / math.sqrt(2)
        donor_view = x
        usable = torch.zeros(n, dtype=torch.bool, device=x.device)
        if n > 1 and self.kind != 'correlated_noise':
            distances = torch.cdist(x, x)
            distances.fill_diagonal_(float('inf'))
            if pseudo_labels is not None:
                reliable = confidence >= self.confidence
                valid = (pseudo_labels[:, None] == pseudo_labels[None, :]) & reliable[:, None] & reliable[None, :]
                distances = distances.masked_fill(~valid, float('inf'))
            distance, donor = distances.min(dim=1)
            usable = torch.isfinite(distance)
            mask = (torch.rand_like(x) < strength) & usable[:, None]
            donor_view = torch.where(mask, x[donor], x)
        if self.kind == 'correlated_noise':
            result = noise_view
        elif self.kind == 'gated_cutmix':
            result = donor_view
        else:
            choice = torch.rand(n, device=x.device) < 0.5
            result = torch.where(choice[:, None], donor_view, noise_view)
        delta = torch.linalg.vector_norm(result - x, dim=1) / math.sqrt(x.shape[1])
        self.count += n
        self.eligible += int(usable.sum().item())
        self.delta_sum += float(delta.sum().item())
        return result

    def diagnostics(self):
        return {'n_views': self.count,
                'eligible_rate': self.eligible / self.count if self.count and self.kind != 'correlated_noise' else None,
                'mean_delta_rms': self.delta_sum / self.count if self.count else None}
