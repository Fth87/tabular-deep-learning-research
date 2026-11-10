"""Labeled-feature-guided conditional donor augmentation for RankUp experiments."""
import numpy as np
import torch


def feature_weights(x, y, classification):
    """Return positive sampling weights; shrink weak small-sample associations."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0)
    total = (centered ** 2).sum(axis=0)
    if classification:
        between = np.zeros(x.shape[1])
        for label in np.unique(y):
            group = x[y == label]
            between += len(group) * (group.mean(axis=0) - x.mean(axis=0)) ** 2
        association = np.sqrt(np.clip(np.divide(between, total, out=np.zeros_like(total), where=total > 0), 0, 1))
    else:
        centered_y = np.asarray(y, dtype=np.float64) - np.mean(y)
        denom = np.sqrt(total * np.sum(centered_y ** 2))
        association = np.abs(np.divide(centered.T @ centered_y, denom, out=np.zeros_like(total), where=denom > 0))
    shrink = len(x) / (len(x) + 100.0)
    return (1.0 - .75 * shrink * np.clip(association, 0., 1.)).astype(np.float32)


class FeatureGuidedAug:
    def __init__(self, weights, pool=2048, k=5):
        self.weights, self.pool, self.k = weights, pool, k
        self.n = self.changed = 0
        self.delta_sum = 0.0

    def __call__(self, x, strength, train, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        if strength <= .1 or not len(x):
            return x.clone()
        n, d = x.shape
        count = min(d - 1, max(1, round(strength * d)))
        weights = torch.as_tensor(self.weights, dtype=x.dtype, device=x.device).expand(n, -1)
        indices = torch.multinomial(weights, count, replacement=False)
        mask = torch.zeros_like(x, dtype=torch.bool).scatter_(1, indices, True)
        candidates = train[torch.randint(len(train), (min(self.pool, len(train)),), device=x.device)]
        kept = (~mask).to(x.dtype)
        distances = ((x.square() * kept).sum(dim=1, keepdim=True)
                     + kept @ candidates.square().T - 2 * (x * kept) @ candidates.T).clamp_min_(0)
        distances.masked_fill_(torch.cdist(x, candidates) <= 1e-6, float('inf'))
        values, neighbours = distances.topk(min(self.k, len(candidates)), largest=False)
        finite = torch.isfinite(values)
        donor_weights = finite.to(x.dtype)
        donor_weights[:, 0] += (~finite.any(dim=1)).to(x.dtype)
        pick = torch.multinomial(donor_weights, 1)
        donor = candidates[neighbours.gather(1, pick).squeeze(1)]
        out = torch.where(mask & finite.any(dim=1, keepdim=True), donor, x)
        self.n += n
        self.changed += int((out != x).any(dim=1).sum().item())
        self.delta_sum += float(torch.linalg.vector_norm(out - x, dim=1).sum().item())
        return out

    def diagnostics(self):
        return {'n_views': self.n, 'changed_rate': self.changed / self.n if self.n else None,
                'mean_delta_l2': self.delta_sum / self.n if self.n else None,
                'feature_weights': self.weights.tolist()}
