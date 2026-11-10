"""Feature-guided augmentation pilot using the existing RankUp loops."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import pilot
from augmentasi_terarah import feature_weights, FeatureGuidedAug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=pilot.FAMILIES, required=True)
    parser.add_argument('--percent', type=int, choices=(1, 10, 30), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    train_file = Path('/data') / f'label_{args.percent:02d}' / args.dataset / 'train_labeled.npz'
    with np.load(train_file, allow_pickle=False) as source:
        weights = feature_weights(source['X'], source['y'], args.dataset == 'pendigits')
    pilot.LocalTransport = lambda **kwargs: FeatureGuidedAug(weights)
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        result = pilot.run(args.dataset, args.percent, 'local_transport', Path('/data'), temp,
                           epochs=120, max_steps=100)
        src = temp / f'label_{args.percent:02d}' / args.dataset
        out = args.output_root / f'label_{args.percent:02d}' / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / 'feature_guided_history.csv').write_bytes((src / 'local_transport_history.csv').read_bytes())
        result['method'] = 'feature_guided'
        result['pilot_config'].pop('radius', None)
        result['pilot_config'].update({'effective_weak_strength': 0., 'strong_strength': .3,
                                       'donor_pool': 2048, 'donor_k': 5})
        (out / 'feature_guided_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print('COMPLETED', args.dataset, args.percent, result['validation'], flush=True)


if __name__ == '__main__':
    main()
