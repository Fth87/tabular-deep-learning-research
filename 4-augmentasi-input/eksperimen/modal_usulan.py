"""Run the proposed augmentation on Modal; results are written locally, volumes are untouched."""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App('experiment-ta-usulan')
image = (
    modal.Image.debian_slim(python_version='3.10')
    .pip_install('numpy<2', 'pandas', 'scikit-learn==1.3.2', 'scipy', 'joblib',
                 'torch==2.1.2', 'tabm', 'optuna==3.6.1')
    .add_local_dir(ROOT / 'hasil' / 'data-5seed', remote_path='/opt/ta/data',
                   ignore=['**/*.csv', '**/*.joblib'])
    .add_local_dir(ROOT / 'hasil' / 'data-semua', remote_path='/opt/ta/data2',
                   ignore=['**/*.csv', '**/*.joblib'])
    .add_local_dir(ROOT / 'experiments', remote_path='/opt/ta/experiments')
    .add_local_dir(ROOT / 'tests', remote_path='/opt/ta/tests')
)

GRID = {
    'none': {'op': 'none'},
    'scarf_w0_s50': {'op': 'scarf', 'weak': 0.0, 'strong': 0.5},
    'gibbs_k5_s30': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'strong': 0.3},
    'gibbs_k5_s50': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'strong': 0.5},
    'gibbs_k20_s30': {'op': 'gibbs', 'k': 20, 'weak': 0.0, 'strong': 0.3},
    'gibbs_k20_s50': {'op': 'gibbs', 'k': 20, 'weak': 0.0, 'strong': 0.5},
    'gibbs_k5_w10_s50': {'op': 'gibbs', 'k': 5, 'weak': 0.1, 'strong': 0.5},
    # rho = target relative displacement; strength per dataset is calibrated from it without labels.
    'cal05': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05},
    'cal05_lab': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'lab_rho': 0.05},
    'cal15_lab': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.15, 'lab_rho': 0.05},
    'cal05_lr': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'lr': 5e-4},
    'cal05_lr_uw1': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'lr': 5e-4, 'uw': 1.0},
    'cal05_uw1_c80': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'uw': 1.0, 'cutoff': 0.8},
    'cal05_lr_uw1_c80': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'lr': 5e-4, 'uw': 1.0, 'cutoff': 0.8},
    'cal05_sat': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'sat': True},
    'cal05_sat_uw1': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'rho': 0.05, 'sat': True, 'uw': 1.0},
    # More diversity while keeping P(x): more Gibbs steps, wider neighbourhoods, or steps calibrated to rho.
    'gibbs_t3': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'strong': 0.5, 'steps': 3},
    'gibbs_k50': {'op': 'gibbs', 'k': 50, 'weak': 0.0, 'strong': 0.5},
    'gibbs_rho20': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'strong': 0.5, 'rho_steps': 0.2},
    'gibbs_k5_s50_lab': {'op': 'gibbs', 'k': 5, 'weak': 0.0, 'strong': 0.5, 'lab_rho': 0.05},
}
DATASETS = ('nhanes', 'pendigits', 'superconductivity')
# Same setting for every operator: weak = identity, strong = 0.5, no tuning.
GRID.update({f'def_{m}': {'op': m, 'weak': 0.0, 'strong': 0.5}
             for m in ('tabcutmix', 'scarf', 'range_limited', 'gaussian', 'zero_out', 'tabcutmix_plus')})
EXTRA = ('lev', 'esl', 'student_performance', 'melbourne_airbnb', 'diabetes_pima', 'cmc', 'satimage',
         'bank_marketing', 'adult_income', 'grid_stability')


def data_root(dataset, percent, base='/opt/ta'):
    return f'{base}/data' if dataset in DATASETS else f'{base}/data2'


METHODS = ('tabcutmix', 'scarf', 'range_limited', 'gaussian', 'zero_out', 'tabcutmix_plus')


def config_for(name, dataset, percent, protocol):
    """tuned_<method> reuses the strengths your 40-trial Optuna run picked for that condition."""
    if name.startswith('tuned_'):
        method = name[len('tuned_'):]
        prefix = 'rankup_cls' if dataset == 'pendigits' else 'rankup'
        best = json.loads((ROOT / 'hasil' / 'ta-5seed-40trial-20261105' / f'label_{percent:02d}' / 'tuning'
                           / dataset / f'{prefix}_{method}_best_params.json').read_text(encoding='utf-8'))
        config = {'op': method, 'weak': best['weak_strength'], 'strong': best['strong_strength']}
    else:
        config = dict(GRID[name])
    return {**config, 'protocol': protocol}


def _run(job):
    import sys
    sys.path.insert(0, '/opt/ta')
    from experiments.usulan import run_one
    dataset, percent, name, config, seed, epochs = job
    return {'name': name, **run_one(data_root(dataset, percent), dataset, percent, config, seed, epochs)}


@app.function(image=image, gpu='T4', timeout=3600, scaledown_window=30)
def run_t4(job):
    return _run(job)


# shortcut: 4 cores is the benchmarked CPU shape, re-measure before changing it.
@app.function(image=image, cpu=4.0, memory=8192, timeout=3600, scaledown_window=30)
def run_cpu(job):
    return _run(job)


@app.function(image=image, cpu=2.0, memory=4096, timeout=600)
def check():
    import subprocess
    import sys
    subprocess.run([sys.executable, '/opt/ta/tests/test_usulan.py', '-v'], check=True, cwd='/opt/ta',
                   env={'PYTHONPATH': '/opt/ta'})


@app.local_entrypoint()
def main(action: str = 'check', names: str = 'all', seeds: str = '42', percents: str = '1,10,30',
         datasets: str = 'all', epochs: int = 120, device: str = 't4', containers: int = 2,
         out: str = 'usulan-screening', protocol: str = 'old'):
    if action == 'check':
        check.remote()
        return
    selected = list(GRID) if names == 'all' else [f'tuned_{m}' for m in METHODS] if names == 'tuned' else names.split(',')
    if action == 'benchmark':
        job = ('superconductivity', 30, 'gibbs_k5_s50', config_for('gibbs_k5_s50', 'superconductivity', 30, protocol), 42, 2)
        for fn, label, price in ((run_cpu, 'cpu4', 0.253), (run_t4, 't4', 0.59)):
            result = fn.remote(job)
            print(label, result['device'], f"{result['seconds']:.1f}s", f"${price}/h", flush=True)
        return
    path = ROOT / 'hasil' / out / 'results.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done.add((row['dataset'], row['label_percent'], row['name'], row['seed']))
    wanted = (DATASETS if datasets == 'all' else DATASETS + EXTRA if datasets == 'semua'
              else tuple(datasets.split(',')))
    local = {'/opt/ta/data': ROOT / 'hasil' / 'data-5seed', '/opt/ta/data2': ROOT / 'hasil' / 'data-semua'}
    exists = lambda d, p: (local[data_root(d, p)] / f'label_{p:02d}' / d / 'manifest.json').exists()
    jobs = [(d, int(p), n, config_for(n, d, int(p), protocol), int(s), epochs) for p in percents.split(',') for d in wanted
            for n in selected for s in seeds.split(',')
            if (d, int(p), n, int(s)) not in done and exists(d, int(p))]
    print(f'{len(jobs)} jobs on {device}, {containers} containers', flush=True)
    worker = (run_t4 if device == 't4' else run_cpu).with_options(max_containers=containers)
    with path.open('a', encoding='utf-8') as sink:
        for result in worker.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                print('FAILED', repr(result), flush=True)
                continue
            sink.write(json.dumps(result) + '\n')
            sink.flush()
            print('done', result['dataset'], result['label_percent'], result['name'], result['seed'],
                  f"{result['seconds']:.0f}s", flush=True)
    print('ALL FINISHED', flush=True)
