"""Run one fixed-budget candidate, using the same pilot and split as before."""
import argparse
import json
import tempfile
from pathlib import Path

import pilot
from augmentasi_lanjutan import AugmentasiLanjutan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--percent', type=int, required=True)
    parser.add_argument('--method', choices=('correlated_noise', 'gated_cutmix', 'hybrid'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    pilot.LocalTransport = lambda **kwargs: AugmentasiLanjutan(args.method)
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        record = pilot.run(args.dataset, args.percent, 'local_transport', Path('/data'), temporary)
        src = temporary / f'label_{args.percent:02d}' / args.dataset
        out = args.output_root / f'label_{args.percent:02d}' / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{args.method}_history.csv').write_bytes((src / 'local_transport_history.csv').read_bytes())
        record['method'] = args.method
        record['pilot_config'].pop('radius', None)
        (out / f'{args.method}_metrics.json').write_text(json.dumps(record, indent=2) + '\n')
    print('COMPLETED', args.dataset, args.percent, args.method, record['validation'], flush=True)


if __name__ == '__main__':
    main()
