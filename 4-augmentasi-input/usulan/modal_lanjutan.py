"""Cheap exploratory comparison: three methods, one seed, T4, no Optuna."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('ta-augmentasi-lanjutan')
data = modal.Volume.from_name('experiment-ta-data')
results = modal.Volume.from_name('experiment-ta-results')
image = (modal.Image.debian_slim(python_version='3.10')
         .pip_install('numpy<2', 'pandas', 'scikit-learn==1.3.2', 'scipy', 'joblib',
                      'torch==2.1.2', 'tabm', 'optuna==3.6.1')
         .pip_install('modal>=1,<2')
         .add_local_dir(ROOT / 'eksperimen' / 'experiments', remote_path='/opt/ta/experiments')
         .add_local_dir(ROOT / 'usulan', remote_path='/opt/ta/usulan',
                        ignore=['hasil/**', '__pycache__/**']))


@app.function(image=image, timeout=180)
def check():
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '/opt/ta/usulan',
                    '-p', 'test_augmentasi_lanjutan.py', '-v'], cwd='/opt/ta/usulan', check=True)


@app.function(image=image, gpu='T4', timeout=300, volumes={'/data': data, '/results': results})
def pilot(dataset: str, percent: int, method: str, run_id: str):
    import subprocess, sys
    data.reload()
    subprocess.run([sys.executable, '/opt/ta/usulan/pilot_lanjutan.py', '--dataset', dataset,
                    '--percent', str(percent), '--method', method,
                    '--output-root', f'/results/{run_id}'], cwd='/opt/ta', check=True)
    results.commit()


@app.local_entrypoint()
def main(action: str = 'check', run_id: str = 'pilot-lanjutan-20261110',
         parallel: int = 3, dataset: str = 'all', percent: str = 'all', method: str = 'all'):
    if action == 'check':
        check.remote()
        return
    if action != 'run' or not 1 <= parallel <= 3 or not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid arguments')
    datasets = ('nhanes', 'pendigits', 'superconductivity')
    methods = ('correlated_noise', 'gated_cutmix', 'hybrid')
    if dataset != 'all' and dataset not in datasets or method != 'all' and method not in methods:
        raise ValueError('Invalid dataset or method')
    percentages = (1, 10, 30) if percent == 'all' else (int(percent),)
    if any(p not in (1, 10, 30) for p in percentages):
        raise ValueError('Invalid percentage')
    jobs = [(d, p, m) for d in (datasets if dataset == 'all' else (dataset,))
            for p in percentages for m in (methods if method == 'all' else (method,))]
    pilot.with_options(max_containers=parallel).spawn_map(
        [d for d, _, _ in jobs], [p for _, p, _ in jobs], [m for _, _, m in jobs], [run_id] * len(jobs))
    print(f'Queued {len(jobs)} jobs with up to {parallel} T4 GPUs', flush=True)
