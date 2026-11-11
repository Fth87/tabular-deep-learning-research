"""Paired seed check of the frozen round-3 candidate against feature-guided."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pilot
from augmentasi_terarah import FeatureGuidedAug, feature_weights
from candidate import Candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=pilot.FAMILIES, required=True)
    parser.add_argument('--method', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    candidate_hash = hashlib.sha256(Path(__file__).with_name('candidate.py').read_bytes()).hexdigest()
    if candidate_hash != '064b9759c5a7bd42497230157f9c22ca9617f5ea2bf1c7564888fab720690592':
        raise ValueError('Candidate changed since round 3; use the frozen snapshot')
    data = Path('/data') / 'label_01' / args.dataset
    with np.load(data / 'train_labeled.npz', allow_pickle=False) as f:
        xl, yl = f['X'], f['y']
    with np.load(data / 'train_unlabeled.npz', allow_pickle=False) as f:
        xu = f['X']
    is_class = args.dataset == 'pendigits'
    weights = feature_weights(xl, yl, is_class)
    pilot.LocalTransport = (lambda **kwargs: Candidate(xl, yl, xu, is_class)) if args.method == 'candidate' else (lambda **kwargs: FeatureGuidedAug(weights))
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        result = pilot.run(args.dataset, 1, 'local_transport', Path('/data'), temp,
                           epochs=120, max_steps=100, seed=43)
        src = temp / 'label_01' / args.dataset
        out = args.output_root / 'label_01' / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{args.method}_history.csv').write_bytes((src / 'local_transport_history.csv').read_bytes())
        result['method'] = args.method
        result['candidate_sha256'] = candidate_hash
        result['pilot_config'].pop('radius', None)
        result['pilot_config'].update({'effective_weak_strength': 0., 'strong_strength': .3,
                                       'donor_pool': 2048, 'donor_k': 5})
        (out / f'{args.method}_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print('COMPLETED', args.dataset, args.method, result['validation'], flush=True)


if __name__ == '__main__':
    main()
