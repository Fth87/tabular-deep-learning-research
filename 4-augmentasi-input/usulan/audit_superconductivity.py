"""Describe repeated feature vectors in the train pool, never validation/test targets.

This research audit reads all train-pool labels, including labels hidden from SSL
training. It therefore is exploratory and must not be treated as blind evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

root = Path(__file__).resolve().parents[1]
config = json.loads((root / 'config/datasets.json').read_text())
spec = next(x for x in config['datasets'] if x['name'] == 'superconductivity')
frame = pd.read_csv(root / spec['csv'])
features = frame.drop(columns=spec['target'])
groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
indices = np.arange(len(frame))
test_share = config['test_fraction']
trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=config['seed']).split(indices, groups=groups))
val_share = config['validation_fraction'] / (config['train_fraction'] + config['validation_fraction'])
train_relative, validation_relative = next(GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=config['seed']).split(trainval, groups=groups[trainval]))
train = trainval[train_relative]
validation = trainval[validation_relative]
training = frame.iloc[train].copy()
training['feature_hash'] = groups[train]
target_groups = training.groupby('feature_hash')[spec['target']]
sizes = target_groups.size()
spreads = target_groups.max() - target_groups.min()
result = {'split_seed': config['seed'], 'train_rows': len(train), 'train_feature_groups': len(sizes),
          'train_rows_in_repeated_feature_groups': int(sizes[sizes > 1].sum()),
          'train_repeated_groups_with_different_target': int(((sizes > 1) & (spreads > 0)).sum()),
          'median_target_range_within_different_target_groups': float(spreads[spreads > 0].median()),
          'max_target_range': float(spreads.max()),
          'exact_feature_hash_overlap_train_validation': len(set(groups[train]) & set(groups[validation])),
          'exact_feature_hash_overlap_train_test': len(set(groups[train]) & set(groups[test])),
          'hidden_train_labels_used_for_this_descriptive_audit': True,
          'validation_test_targets_used': False}
output = root / 'usulan/hasil/audit_superconductivity_train.json'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
