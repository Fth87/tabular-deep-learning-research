"""Generate descriptive EDA from the three experiment datasets."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'eda' / 'hasil'


def analyze(name, target, task="classification"):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'config' / 'datasets.json').read_text(encoding='utf-8'))
    spec = next(spec for spec in config['datasets'] if spec['name'] == name)
    df = pd.read_csv(ROOT / spec['csv'])
    if spec.get('test_csv'):
        df = pd.concat([df, pd.read_csv(ROOT / spec['test_csv'])], ignore_index=True)
    counts = df[target].value_counts().sort_index()
    features = df.drop(columns=target)
    numeric = features.select_dtypes(include='number')
    summary = {
        'dataset': name, 'rows': len(df), 'features': len(features.columns),
        'numeric_features': len(numeric.columns), 'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }
    if task == 'regression':
        summary['target_statistics'] = {k: float(v) for k, v in df[target].describe().items()}
    else:
        summary['target_counts'] = {str(k): int(v) for k, v in counts.items()}
        summary['target_proportion'] = {str(k): round(float(v / len(df)), 4) for k, v in counts.items()}
    (OUTPUT / f'{name}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    df.describe(include='all').T.to_csv(OUTPUT / f'{name}_descriptive.csv')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if task == 'regression':
        ax.hist(df[target], bins=30, color='#287b86', edgecolor='white')
    else:
        counts.plot.bar(ax=ax, color='#287b86', rot=0)
    ax.set(title=f'Distribusi target: {name}', xlabel='Nilai target' if task == 'regression' else 'Level' if name == 'nhanes' else 'Kelas', ylabel='Jumlah')
    fig.tight_layout()
    fig.savefig(OUTPUT / f'{name}_target.png', dpi=150)
    plt.close(fig)
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='vlag', center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title(f'Korelasi Pearson fitur: {name}')
    fig.tight_layout()
    fig.savefig(OUTPUT / f'{name}_correlation.png', dpi=130)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    for dataset, target, task in [('nhanes', 'y', 'ordinal'), ('pendigits', 'digit', 'classification'), ('superconductivity', 'critical_temp', 'regression')]:
        print(json.dumps(analyze(dataset, target, task), indent=2))
