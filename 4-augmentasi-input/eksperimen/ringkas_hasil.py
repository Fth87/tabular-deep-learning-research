"""Summarize downloaded five-seed results without hiding incomplete runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

AUGMENTATIONS = ('tabcutmix', 'scarf', 'range_limited', 'gaussian', 'zero_out', 'tabcutmix_plus')
SEEDS = (42, 123, 456, 789, 101112)


def summarize(run_dir: Path, output: Path):
    rows = []
    for percentage in (1, 10, 30):
        for dataset in ('nhanes', 'pendigits', 'superconductivity'):
            metric = {'nhanes': 'qwk', 'pendigits': 'macro_f1', 'superconductivity': 'mse'}[dataset]
            prefix = 'rankup_cls' if dataset == 'pendigits' else 'rankup'
            for method in AUGMENTATIONS:
                values, missing = {}, []
                split_values = {"train": {}, "validation": {}}
                for seed in SEEDS:
                    path = run_dir / f'label_{percentage:02d}' / f'seed_{seed}' / dataset / f'{prefix}_{method}_metrics.json'
                    if not path.is_file():
                        missing.append(seed)
                        continue
                    result = json.loads(path.read_text(encoding='utf-8'))
                    if (result.get('dataset') != dataset or result.get('primary_metric') != metric
                            or result.get('seed') != seed or not math.isfinite(float(result[metric]))):
                        raise ValueError(f'Unexpected dataset, metric or seed in {path}')
                    values[seed] = float(result[metric])
                    for split in split_values:
                        value = float(result[f'{split}_{metric}'])
                        if not math.isfinite(value):
                            raise ValueError(f'Non-finite {split} metric in {path}')
                        split_values[split][seed] = value
                complete = not missing
                row = {'label_percent': percentage, 'dataset': dataset, 'method': method,
                       'metric': metric, 'n_seeds': len(values),
                       'mean': statistics.mean(values.values()) if complete else '',
                       'std': statistics.stdev(values.values()) if complete else '',
                       'status': 'complete' if complete else 'missing' if not values else 'partial',
                       'missing_seeds': ','.join(map(str, missing))}
                for split, scores in split_values.items():
                    row[f'{split}_mean'] = statistics.mean(scores.values()) if complete else ''
                    row[f'{split}_std'] = statistics.stdev(scores.values()) if complete else ''
                row.update({f'seed_{seed}': values.get(seed, '') for seed in SEEDS})
                rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(row['status'] == 'complete' for row in rows)}/{len(rows)} eksperimen lengkap: {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path, help='Folder hasil untuk satu run ID')
    parser.add_argument('--output', type=Path, default=Path('eksperimen/hasil/ringkasan.csv'))
    args = parser.parse_args()
    summarize(args.run_dir, args.output)
