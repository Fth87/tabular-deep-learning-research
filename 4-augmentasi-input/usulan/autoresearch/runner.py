"""Run candidate with fixed RankUp pilot, retaining input and code checksums."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pilot
from candidate import Candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=pilot.FAMILIES, required=True)
    parser.add_argument('--percent', type=int, choices=(1, 10, 30), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    data = Path('/data') / f'label_{args.percent:02d}' / args.dataset
    with np.load(data / 'train_labeled.npz', allow_pickle=False) as f:
        xl, yl = f['X'], f['y']
    with np.load(data / 'train_unlabeled.npz', allow_pickle=False) as f:
        xu = f['X']
    pilot.LocalTransport = lambda **kwargs: Candidate(xl, yl, xu, args.dataset == 'pendigits')
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        result = pilot.run(args.dataset, args.percent, 'local_transport', Path('/data'), temp,
                           epochs=120, max_steps=100)
        src = temp / f'label_{args.percent:02d}' / args.dataset
        out = args.output_root / f'label_{args.percent:02d}' / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / 'candidate_history.csv').write_bytes((src / 'local_transport_history.csv').read_bytes())
        result['method'] = 'candidate'
        result['candidate_sha256'] = hashlib.sha256(Path(__file__).with_name('candidate.py').read_bytes()).hexdigest()
        result['pilot_config'].pop('radius', None)
        result['pilot_config'].update({'effective_weak_strength': 0., 'strong_strength': .3,
                                       'donor_pool': 2048, 'donor_k': 5})
        (out / 'candidate_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print('COMPLETED', args.dataset, args.percent, result['validation'], result['candidate_sha256'], flush=True)


if __name__ == '__main__':
    main()
