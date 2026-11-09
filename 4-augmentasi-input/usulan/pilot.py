"""Low-cost, validation-only pilot using existing RankUp training functions."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from local_transport import LocalTransport
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAMILIES = {'nhanes': 'rankup', 'pendigits': 'rankup_classification', 'superconductivity': 'regression'}
METRICS = {'nhanes': 'qwk', 'pendigits': 'macro_f1', 'superconductivity': 'mse'}
METHODS = ('supervised', 'identity', 'gaussian', 'tabcutmix', 'tabcutmix_plus',
           'range_limited', 'scarf', 'zero_out', 'local_transport')


def identity(x, *args):
    return x


def run(dataset, percent, method, data_root, output_root, epochs=30, max_steps=40, seed=42):
    if method not in METHODS:
        raise ValueError('Unknown pilot method')
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot must run on a Modal GPU')
    root = Path(data_root) / f'label_{percent:02d}' / dataset
    manifest = json.loads((root / 'manifest.json').read_text())
    def load(name):
        with np.load(root / f'{name}.npz', allow_pickle=False) as split:
            return split['X'].astype(np.float32), split['y'] if 'y' in split else None
    x_l, y_l = load('train_labeled')
    x_u, _ = load('train_unlabeled')
    x_v, y_v = load('validation')
    device = torch.device('cuda')
    variant = 'gaussian' if method in ('supervised', 'identity', 'local_transport') else method
    module = importlib.import_module(f'experiments.{FAMILIES[dataset]}.{variant}')
    transport = LocalTransport(radius=.5, confidence=.8) if method == 'local_transport' else None
    if transport:
        module.augment = transport
    elif method in ('identity', 'supervised'):
        module.augment = identity
    args = SimpleNamespace(seed=seed, learning_rate=.002, batch_size=32, max_steps=max_steps,
                           patience=15, rank_weight=.2, unlabeled_weight=0.0 if method == 'supervised' else .2,
                           confidence_cutoff=.95, donor_confidence=.8,
                           ordinal_levels=np.array(manifest.get('target_values', []), dtype=np.int64))
    if dataset == 'pendigits':
        classes = [str(c) for c in manifest['classes']]
        mapping = {c: i for i, c in enumerate(classes)}
        y_l = np.array([mapping[str(y)] for y in y_l], dtype=np.int64)
        y_v = np.array([mapping[str(y)] for y in y_v], dtype=np.int64)
        score, epoch, _, history = module.fit_model(args, x_l, y_l, x_u, x_v, y_v,
                                                     len(classes), METRICS[dataset], .1, .3, epochs, device)
    else:
        y_l = np.asarray(y_l, dtype=np.float32)
        y_v = np.asarray(y_v, dtype=np.float32)
        score, epoch, _, history = module.fit_model(args, x_l, y_l, x_u, x_v, y_v, .1, .3, epochs)
    input_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ('manifest.json', 'train_labeled.npz', 'train_unlabeled.npz', 'validation.npz')}
    record = {'dataset': dataset, 'label_percent': percent, 'method': method, 'seed': seed,
              'metric': METRICS[dataset], 'validation': float(score), 'best_epoch': epoch,
              'epochs_run': len(history), 'pilot_config': {'epochs': epochs, 'max_steps': max_steps,
              'weak_strength': .1, 'strong_strength': .3, 'radius': .5, 'donor_confidence': .8,
              'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
              'unlabeled_weight': args.unlabeled_weight, 'confidence_cutoff': args.confidence_cutoff},
              'input_sha256': input_hashes,
              'mean_pseudo_accept_rate': float(np.mean([h['pseudo_accept_rate'] for h in history])),
              'best_epoch_pseudo_accept_rate': float(history[epoch-1]['pseudo_accept_rate']),
              'augmentation_diagnostics': transport.diagnostics() if transport else None,
              'test_used': False, 'device': torch.cuda.get_device_name(0)}
    out = Path(output_root) / f'label_{percent:02d}' / dataset
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / f'{method}_history.csv', index=False)
    (out / f'{method}_metrics.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=FAMILIES, required=True)
    p.add_argument('--percent', type=int, choices=(1,10,30), required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/data'))
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=40)
    args = p.parse_args()
    run(args.dataset, args.percent, args.method, args.data_root, args.output_root, args.epochs, args.max_steps)


if __name__ == '__main__':
    main()
