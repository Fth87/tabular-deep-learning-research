"""Summarize five-fold results without treating partial runs as complete."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from ringkas_hasil import AUGMENTATIONS
from experiments.resume import seed_complete, seed_files


def summarize(run_dir: Path, output: Path, seed=42):
    rows = []
    for percentage in (1, 10, 30):
        for dataset, metric in (('nhanes', 'qwk'), ('pendigits', 'macro_f1'), ('superconductivity', 'mse')):
            prefix = 'rankup_cls' if dataset == 'pendigits' else 'rankup'
            for method in AUGMENTATIONS:
                scores = {split: {} for split in ('train', 'validation', 'test')}
                missing = []
                for fold in range(1, 6):
                    base = run_dir / 'kfold' / f'label_{percentage:02d}' / f'fold_{fold:02d}'
                    directory = base / f'seed_{seed}' / dataset
                    path = directory / f'{prefix}_{method}_metrics.json'
                    marker = directory / f'{prefix}_{method}_complete.json'
                    params = base / 'tuning' / dataset / f'{prefix}_{method}_best_params.json'
                    if not seed_complete(marker, [*seed_files(directory, prefix, method), params]):
                        missing.append(fold)
                        continue
                    result = json.loads(path.read_text(encoding='utf-8'))
                    if (result.get('dataset') != dataset or result.get('primary_metric') != metric
                            or result.get('seed') != seed):
                        raise ValueError(f'Unexpected dataset, metric or seed in {path}')
                    for split in scores:
                        key = metric if split == 'test' else f'{split}_{metric}'
                        value = float(result[key])
                        if not math.isfinite(value):
                            raise ValueError(f'Non-finite {split} metric in {path}')
                        scores[split][fold] = value
                complete = not missing
                count = len(scores['test'])
                row = {'label_percent': percentage, 'dataset': dataset, 'method': method,
                       'metric': metric, 'seed': seed, 'n_folds': count,
                       'status': 'complete' if complete else 'missing' if not count else 'partial',
                       'missing_folds': ','.join(map(str, missing))}
                for split, values in scores.items():
                    row[f'{split}_mean'] = statistics.mean(values.values()) if complete else ''
                    row[f'{split}_std'] = statistics.stdev(values.values()) if complete else ''
                    row.update({f'{split}_fold_{fold}': values.get(fold, '') for fold in range(1, 6)})
                rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(row['status'] == 'complete' for row in rows)}/{len(rows)} kombinasi lengkap: {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('eksperimen/hasil/ringkasan-kfold.csv'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    summarize(args.run_dir, args.output, args.seed)
