"""Tune once, evaluate five training seeds, and summarize one experiment."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from .resume import (commit_progress, mark_seed_complete, run_signature,
                     seed_complete, seed_files, verify_signature)

SEEDS = (42, 123, 456, 789, 101112)
FAMILIES = {'nhanes': 'rankup', 'pendigits': 'rankup_classification', 'superconductivity': 'regression'}
METRICS = {'nhanes': 'qwk', 'pendigits': 'macro_f1', 'superconductivity': 'mse'}


def run_experiment(dataset, method, data_root, output_root, tuning_seed, trials,
                   optuna_epochs, epochs, run_command=subprocess.run):
    family = FAMILIES[dataset]
    prefix = 'rankup_cls' if dataset == 'pendigits' else 'rankup'
    metric = METRICS[dataset]
    common = [sys.executable, '-m', f'experiments.{family}.{method}', '--dataset', dataset,
              '--data-root', str(data_root), '--n-trials', str(trials),
              '--optuna-epochs', str(optuna_epochs), '--epochs', str(epochs)]
    tuning_root = output_root / 'tuning'
    params_file = tuning_root / dataset / f'{prefix}_{method}_best_params.json'
    signature_file = tuning_root / dataset / f'{prefix}_{method}_resume.json'
    existing = [signature_file.parent / f'{prefix}_{method}_{suffix}' for suffix in
                ('study.pkl', 'optuna_trials.csv', 'best_params.json', 'run_config.json')]
    if not signature_file.exists() and any(path.exists() for path in existing):
        raise ValueError('Hasil lama tidak memiliki metadata resume; gunakan run-id baru')
    signature = run_signature(dataset, method, data_root, tuning_seed, trials, optuna_epochs, epochs)
    verify_signature(signature_file, signature)
    run_command([*common, '--output-root', str(tuning_root), '--seed', str(tuning_seed),
                 '--tune-only'], check=True)
    commit_progress()
    values = []
    split_values = {"train": [], "validation": [], "test": []}
    for seed in SEEDS:
        seed_root = output_root / f'seed_{seed}'
        directory = seed_root / dataset
        files = [*seed_files(directory, prefix, method), params_file]
        marker = directory / f'{prefix}_{method}_complete.json'
        complete = seed_complete(marker, files)
        if complete:
            print(f'Resuming: seed {seed} already complete', flush=True)
        else:
            run_command([*common, '--output-root', str(seed_root), '--seed', str(seed),
                         '--params-file', str(params_file)], check=True)
        path = directory / f'{prefix}_{method}_metrics.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record.get('dataset') != dataset or record.get('primary_metric') != metric
                or record.get('seed') != seed or not math.isfinite(float(record[metric]))):
            raise ValueError(f'Unexpected result for seed {seed}: {path}')
        values.append(float(record[metric]))
        for split, key in (('train', f'train_{metric}'), ('validation', f'validation_{metric}'), ('test', metric)):
            value = float(record[key])
            if not math.isfinite(value):
                raise ValueError(f'Non-finite {split} metric for seed {seed}: {path}')
            split_values[split].append(value)
        if not complete:
            mark_seed_complete(marker, files)
    split_metrics = {split: {'values': scores, 'mean': statistics.mean(scores),
                              'std': statistics.stdev(scores)}
                     for split, scores in split_values.items()}
    summary = {'dataset': dataset, 'method': method, 'metric': metric, 'seeds': list(SEEDS),
               'values': values, 'mean': statistics.mean(values), 'std': statistics.stdev(values),
               'n_seeds': len(values), 'tuning_seed': tuning_seed, 'split_metrics': split_metrics}
    path = output_root / 'summary' / dataset / f'{prefix}_{method}_summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    commit_progress()
    print(json.dumps(summary, indent=2))
    return summary


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
    run_experiment(args.dataset, args.method, args.data_root, args.output_root,
                   args.tuning_seed, args.n_trials, args.optuna_epochs, args.epochs)


if __name__ == '__main__':
    main()
