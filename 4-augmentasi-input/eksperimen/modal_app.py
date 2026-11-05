"""Train RankUp on regression, ordinal and classification datasets."""
import re
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App('experiment-ta-rankup')
data = modal.Volume.from_name('experiment-ta-data', create_if_missing=True)
results = modal.Volume.from_name('experiment-ta-results', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.10')
    .pip_install('numpy<2', 'pandas', 'scikit-learn==1.3.2', 'scipy',
                 'joblib', 'torch==2.1.2', 'tabm', 'optuna==3.6.1')
    .pip_install('modal>=1,<2')
    .add_local_dir(ROOT / 'dataset', remote_path='/opt/ta/dataset')
    .add_local_dir(ROOT / 'config', remote_path='/opt/ta/config')
    .add_local_dir(ROOT / 'eksperimen' / 'experiments', remote_path='/opt/ta/experiments')
    .add_local_file(ROOT / 'prepare_data.py', remote_path='/opt/ta/prepare_data.py')
    .add_local_file(ROOT / 'eksperimen' / 'workflow_kfold.py', remote_path='/opt/ta/workflow_kfold.py')
    .add_local_dir(ROOT / 'eksperimen' / 'tests', remote_path='/opt/ta/tests')
)

@app.function(image=image, timeout=1200)
def check():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '/opt/ta/tests',
                    '-p', 'test_rankup.py', '-v'], check=True, cwd='/opt/ta')


@app.function(image=image, volumes={'/data': data}, timeout=1800)
def prepare(dataset: str, label_percent: int, protocol: str, fold: int):
    import subprocess
    import sys
    root = f'/data/label_{label_percent:02d}'
    command = [sys.executable, '/opt/ta/prepare_data.py', '--dataset', dataset,
               '--labeled-fraction', str(label_percent / 100)]
    if protocol == 'kfold':
        root = f'/data/kfold/label_{label_percent:02d}/fold_{fold:02d}'
        command += ['--fold', str(fold)]
    subprocess.run([*command, '--output', root], check=True)
    data.commit()

@app.function(image=image, gpu='T4', timeout=43200,
              volumes={'/data': data, '/results': results})
def train(dataset: str, method: str, run_id: str, label_percent: int, seed: int, trials: int,
          optuna_epochs: int, epochs: int, protocol: str, fold: int):
    import os
    import subprocess
    import sys
    data.reload()
    results.reload()
    os.environ['EXPERIMENT_TA_RESULTS_VOLUME'] = 'experiment-ta-results'
    data_root = f'/data/label_{label_percent:02d}'
    output_root = f'/results/{run_id}/label_{label_percent:02d}'
    workflow = ['-m', 'experiments.workflow']
    if protocol == 'kfold':
        data_root = f'/data/kfold/label_{label_percent:02d}/fold_{fold:02d}'
        output_root = f'/results/{run_id}/kfold/label_{label_percent:02d}/fold_{fold:02d}'
        workflow = ['/opt/ta/workflow_kfold.py']
    subprocess.run([sys.executable, *workflow,
                    '--dataset', dataset, '--method', method,
                    '--data-root', data_root, '--output-root', output_root,
                    '--tuning-seed', str(seed), '--n-trials', str(trials),
                    '--optuna-epochs', str(optuna_epochs), '--epochs', str(epochs)],
                   cwd='/opt/ta', check=True)
    results.commit()

@app.local_entrypoint()
def main(action: str = 'prepare', dataset: str = 'all', method: str = 'all',
         run_id: str = '', label_percent: str = 'all', seed: int = 42, trials: int = 40,
         optuna_epochs: int = 40, epochs: int = 120, max_parallel_jobs: int = 4,
         protocol: str = 'single', fold: str = 'all'):
    if action == 'check':
        check.remote()
        return
    datasets = ('nhanes', 'pendigits', 'superconductivity') if dataset == 'all' else (dataset,)
    methods = ('tabcutmix', 'scarf', 'range_limited', 'gaussian', 'zero_out', 'tabcutmix_plus')
    if any(d not in ('nhanes', 'pendigits', 'superconductivity') for d in datasets):
        raise ValueError('dataset must be all, nhanes, pendigits, or superconductivity')
    percentages = (1, 10, 30) if label_percent == 'all' else (int(label_percent),)
    if any(p not in (1, 10, 30) for p in percentages):
        raise ValueError('label-percent must be all, 1, 10, or 30')
    if protocol not in ('single', 'kfold'):
        raise ValueError('protocol must be single or kfold')
    if protocol == 'single' and fold != 'all':
        raise ValueError('fold requires --protocol kfold')
    folds = (0,) if protocol == 'single' else tuple(range(1, 6)) if fold == 'all' else (int(fold),)
    if protocol == 'kfold' and any(f not in range(1, 6) for f in folds):
        raise ValueError('fold must be all or 1..5')
    if action == 'prepare':
        for percentage in percentages:
            for d in datasets:
                for f in folds:
                    prepare.remote(d, percentage, protocol, f)
        return
    if action != 'train':
        raise ValueError('action must be check, prepare or train')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('provide a unique --run-id containing letters, numbers, _ or -')
    if method != 'all' and method not in methods:
        raise ValueError(f'method must be all or one of {methods}')
    if min(trials, optuna_epochs, epochs, max_parallel_jobs) < 1:
        raise ValueError('training counts and parallelism must be positive')
    selected = methods if method == 'all' else (method,)
    jobs = [(d, m, p, f) for p in percentages for d in datasets for m in selected for f in folds]
    print(f"Submitting {len(jobs)} jobs, maximum {max_parallel_jobs} GPU containers", flush=True)
    worker = train.with_options(max_containers=max_parallel_jobs)
    worker.spawn_map([d for d, _, _, _ in jobs], [m for _, m, _, _ in jobs],
                     [run_id] * len(jobs), [p for _, _, p, _ in jobs],
                     [seed] * len(jobs), [trials] * len(jobs),
                     [optuna_epochs] * len(jobs), [epochs] * len(jobs),
                     [protocol] * len(jobs), [f for _, _, _, f in jobs])
    print(f"Queued all {len(jobs)} jobs in Modal", flush=True)
