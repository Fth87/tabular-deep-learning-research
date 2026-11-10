"""Promote a bounded number of confident tree-teacher predictions for training."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


def promote(x_l, y_l, x_u, classification, seed=42, ratio=.5, maximum=1200):
    if not len(x_u) or ratio <= 0 or maximum <= 0:
        return x_l, y_l, {'promoted': 0}
    kind = ExtraTreesClassifier if classification else ExtraTreesRegressor
    teacher = kind(n_estimators=80, min_samples_leaf=2, max_features=.8, n_jobs=2, random_state=seed)
    teacher.fit(x_l, y_l)
    if classification:
        probabilities = teacher.predict_proba(x_u)
        confidence = probabilities.max(axis=1)
        pseudo = teacher.classes_[probabilities.argmax(axis=1)]
    else:
        predictions = np.stack([tree.predict(x_u) for tree in teacher.estimators_])
        confidence = -predictions.std(axis=0)
        pseudo = predictions.mean(axis=0)
    count = min(len(x_u), maximum, max(1, int(len(x_l) * ratio)))
    chosen = np.argsort(-confidence, kind='stable')[:count]
    return (np.concatenate([x_l, x_u[chosen]]).astype(np.float32),
            np.concatenate([y_l, pseudo[chosen]]).astype(y_l.dtype),
            {'promoted': count, 'ratio_to_labeled': ratio, 'maximum': maximum,
             'confidence': 'max_probability' if classification else 'negative_tree_std',
             'promoted_samples_remain_in_unlabeled_pool': True})
