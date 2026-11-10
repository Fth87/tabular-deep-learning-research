"""Conditional donor augmentation with random or correlated feature blocks."""
import torch
import torch.nn.functional as F


class ConditionalAug:
    def __init__(self, grouped=False, pool=2048, k=5):
        self.grouped, self.pool, self.k = grouped, pool, k
        self.correlations = None
        self.n = 0
        self.changed = 0
        self.distances = 0.0

    def __call__(self, x, strength, train, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        # An unmodified weak view avoids corrupting the teacher's pseudo-label.
        if strength <= .1 or not len(x):
            return x.clone()
        n, d = x.shape
        count = min(d - 1, max(1, round(strength * d)))
        if self.grouped:
            if self.correlations is None:
                sample = train[:min(4096, len(train))]
                centered = sample - sample.mean(dim=0)
                unit = F.normalize(centered, dim=0)
                self.correlations = (unit.T @ unit).abs()
            pivots = torch.randint(d, (n,), device=x.device)
            order = self.correlations[pivots].argsort(dim=1, descending=True)
        else:
            order = torch.rand((n, d), device=x.device).argsort(dim=1)
        mask = torch.zeros_like(x, dtype=torch.bool).scatter_(1, order[:, :count], True)
        candidates = train[torch.randint(len(train), (min(self.pool, len(train)),), device=x.device)]
        kept = (~mask).to(x.dtype)
        distances = ((x.square() * kept).sum(dim=1, keepdim=True)
                     + kept @ candidates.square().T - 2 * (x * kept) @ candidates.T).clamp_min_(0)
        full_distance = torch.cdist(x, candidates)
        distances.masked_fill_(full_distance <= 1e-6, float('inf'))
        values, neighbours = distances.topk(min(self.k, len(candidates)), largest=False)
        finite = torch.isfinite(values)
        # Draw uniformly only from finite donors, falling back to identity when none exist.
        weights = finite.to(x.dtype)
        weights[:, 0] += (~finite.any(dim=1)).to(x.dtype)
        choice = torch.multinomial(weights, 1).squeeze(1)
        donors = candidates[neighbours.gather(1, choice[:, None]).squeeze(1)]
        out = torch.where(mask & finite.any(dim=1, keepdim=True), donors, x)
        self.n += n
        self.changed += int((out != x).any(dim=1).sum().item())
        self.distances += float(torch.linalg.vector_norm(out - x, dim=1).sum().item())
        return out

    def diagnostics(self):
        return {'n_views': self.n, 'changed_rate': self.changed / self.n if self.n else None,
                'mean_delta_l2': self.distances / self.n if self.n else None}


def soft_rank_loss(module, weak, strong, cutoff):
    """Soft weighting inspired by SoftMatch, applied to RankUp pairwise logits."""
    probabilities = F.softmax(module.pairwise_logits(weak), dim=-1)
    if probabilities.ndim == 4:
        probabilities = probabilities.mean(dim=0)
    confidence, targets = probabilities.max(dim=-1)
    logits = module.pairwise_logits(strong)
    if logits.ndim == 4:
        targets = targets.unsqueeze(0).expand(logits.shape[0], -1, -1)
        confidence = confidence.unsqueeze(0).expand(logits.shape[0], -1, -1)
    losses = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), reduction='none')
    # Linear ramp is an explicit heuristic, not the SoftMatch weighting rule.
    weights = ((confidence.reshape(-1) - .5) / (cutoff - .5)).clamp(0, 1)
    return (losses * weights).mean(), float(weights.mean().detach().cpu())
