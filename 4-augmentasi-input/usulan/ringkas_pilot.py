"""Summarize validation-only pilot records downloaded from the Modal volume."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in args.directory.rglob('*_metrics.json')]
    if not records:
        raise SystemExit('No pilot metrics found')
    rows = []
    for record in records:
        if record['test_used']:
            raise ValueError('Unexpected test use in validation-only pilot')
        diagnostics = record.get('augmentation_diagnostics') or {}
        rows.append({k: record[k] for k in ('dataset', 'label_percent', 'method', 'metric',
                     'validation', 'best_epoch', 'epochs_run', 'seed', 'device')} | {
                     'pseudo_accept_rate': record['best_epoch_pseudo_accept_rate'],
                     'eligible_rate': diagnostics.get('eligible_rate'),
                     'input_hashes': json.dumps(record['input_sha256'], sort_keys=True)})
    frame = pd.DataFrame(rows)
    if frame.duplicated(['dataset', 'label_percent', 'method']).any():
        raise ValueError('Duplicate experimental condition')
    hashes = frame.groupby(['dataset', 'label_percent'])['input_hashes'].nunique()
    if (hashes != 1).any():
        raise ValueError('Different splits within a comparison')
    utility = frame['validation'].where(frame['metric'] != 'mse', -frame['validation'])
    frame['validation_rank'] = utility.groupby([frame['dataset'], frame['label_percent']]).rank(
        ascending=False, method='min')
    frame = frame.drop(columns='input_hashes').sort_values(['dataset', 'label_percent', 'validation_rank'])
    destination = args.directory.parent / (args.directory.name + '-ringkasan.csv')
    frame.to_csv(destination, index=False)
    print(f'{len(frame)} records saved to {destination}')
    for (dataset, percent), group in frame.groupby(['dataset', 'label_percent']):
        print(f'\n{dataset} {percent}% ({group.metric.iloc[0]}):')
        for _, row in group.iterrows():
            print(f"  {int(row.validation_rank)} {row.method:16} {row.validation:.6f} epoch={row.best_epoch}")


if __name__ == '__main__':
    main()
