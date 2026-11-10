"""Experimental data-adaptive mixture of three RankUp augmentation operators."""
import numpy as np
import torch
from augmentasi_terarah import FeatureGuidedAug, feature_weights
from conditional_aug import ConditionalAug


def operator_probabilities(x_labeled, x_unlabeled):
    """Heuristic: use correlated blocks more when train features are redundant."""
    x = np.concatenate([x_labeled, x_unlabeled])
    corr = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)), nan=0., posinf=0., neginf=0.)
    offdiag = corr[np.triu_indices_from(corr, 1)]
    mean_corr = float(offdiag.mean()) if len(offdiag) else 0.
    n = len(x_labeled)
    group = float(np.clip((mean_corr - .25) / .15, 0., 1.)) * n / (n + 300.)
    guided = (1. - group) * n / (n + 100.)
    return {'random': 1. - group - guided, 'guided': guided, 'group': group,
            'train_mean_abs_correlation': mean_corr, 'n_labeled': n}


class AdaptiveAug:
    def __init__(self, x_labeled, y_labeled, x_unlabeled, classification):
        self.probabilities = operator_probabilities(x_labeled, x_unlabeled)
        self.random = ConditionalAug(grouped=False)
        self.guided = FeatureGuidedAug(feature_weights(x_labeled, y_labeled, classification))
        self.group = ConditionalAug(grouped=True)
        self.counts = {'random': 0, 'guided': 0, 'group': 0}

    def __call__(self, x, strength, train, pseudo_labels, confidence,
                 donor_confidence, quantile_edges, feature_groups):
        if strength <= .1 or not len(x):
            return x.clone()
        p = self.probabilities
        choice = torch.rand(len(x), device=x.device)
        selection = {'random': choice < p['random'],
                     'guided': (choice >= p['random']) & (choice < p['random'] + p['guided']),
                     'group': choice >= p['random'] + p['guided']}
        result = x.clone()
        for name, chosen in selection.items():
            if chosen.any():
                pl = None if pseudo_labels is None else pseudo_labels[chosen]
                conf = None if confidence is None else confidence[chosen]
                result[chosen] = getattr(self, name)(x[chosen], strength, train, pl, conf,
                                                     donor_confidence, quantile_edges, feature_groups)
                self.counts[name] += int(chosen.sum().item())
        return result

    def diagnostics(self):
        return {'probabilities': self.probabilities, 'operator_counts': self.counts,
                'operators': {name: getattr(self, name).diagnostics() for name in self.counts}}
