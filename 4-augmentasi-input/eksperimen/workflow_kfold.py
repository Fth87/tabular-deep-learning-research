"""Tune and train one seed for one fold, with resumable checkpoints."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from experiments.workflow import FAMILIES, METRICS
from experiments.resume import (commit_progress, digest, mark_seed_complete, run_signature,
                                seed_complete, seed_files, verify_signature)


def run_fold(dataset, method, data_root, output_root, seed, trials, optuna_epochs, epochs,
             run_command=subprocess.run):
    prefix = 'rankup_cls' if dataset == 'pendigits' else 'rankup'
    metric = METRICS[dataset]
    common = [sys.executable, '-m', f'experiments.{FAMILIES[dataset]}.{method}', '--dataset', dataset,
              '--data-root', str(data_root), '--n-trials', str(trials),
              '--optuna-epochs', str(optuna_epochs), '--epochs', str(epochs)]
    tuning_root = output_root / 'tuning'
    directory = tuning_root / dataset
    params = directory / f'{prefix}_{method}_best_params.json'
    signature_file = directory / f'{prefix}_{method}_resume.json'
    existing = [directory / f'{prefix}_{method}_{suffix}' for suffix in
                ('study.pkl', 'optuna_trials.csv', 'best_params.json', 'run_config.json')]
    if not signature_file.exists() and any(path.exists() for path in existing):
        raise ValueError('Hasil lama tidak memiliki metadata resume; gunakan run-id baru')
    signature = run_signature(dataset, method, data_root, seed, trials, optuna_epochs, epochs)
    signature['kfold_workflow_sha256'] = digest(Path(__file__))
    verify_signature(signature_file, signature)
    run_command([*common, '--output-root', str(tuning_root), '--seed', str(seed), '--tune-only'], check=True)
    commit_progress()
    seed_root = output_root / f'seed_{seed}'
    directory = seed_root / dataset
    files = [*seed_files(directory, prefix, method), params]
    marker = directory / f'{prefix}_{method}_complete.json'
    complete = seed_complete(marker, files)
    if not complete:
        run_command([*common, '--output-root', str(seed_root), '--seed', str(seed),
                     '--params-file', str(params)], check=True)
    else:
        print(f'Resuming: seed {seed} already complete', flush=True)
    record = json.loads((directory / f'{prefix}_{method}_metrics.json').read_text())
    if record.get('dataset') != dataset or record.get('primary_metric') != metric or record.get('seed') != seed:
        raise ValueError('Unexpected dataset, metric or seed')
    for key in (metric, f'train_{metric}', f'validation_{metric}'):
        if not math.isfinite(float(record[key])):
            raise ValueError(f'Non-finite metric {key}')
    if not complete:
        mark_seed_complete(marker, files)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=FAMILIES, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tuning-seed', type=int, default=42)
    parser.add_argument('--n-trials', type=int, default=40)
    parser.add_argument('--optuna-epochs', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    run_fold(args.dataset, args.method, args.data_root, args.output_root, args.tuning_seed,
             args.n_trials, args.optuna_epochs, args.epochs)


if __name__ == '__main__':
    main()
