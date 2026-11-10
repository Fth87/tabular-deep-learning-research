"""Validation-only tree-teacher RankUp pilots on at most three T4 GPUs."""
from pathlib import Path
import modal

root = Path(__file__).resolve().parents[1]
app = modal.App('ta-tree-rankup')
data = modal.Volume.from_name('experiment-ta-data')
results = modal.Volume.from_name('experiment-ta-results')
image = (modal.Image.debian_slim(python_version='3.10')
         .pip_install('numpy<2', 'pandas', 'scikit-learn==1.3.2', 'scipy', 'joblib',
                      'torch==2.1.2', 'tabm', 'optuna==3.6.1')
         .pip_install('modal>=1,<2')
         .add_local_dir(root / 'eksperimen/experiments', remote_path='/opt/ta/experiments')
         .add_local_dir(root / 'usulan', remote_path='/opt/ta/usulan',
                        ignore=['hasil/**', '__pycache__/**']))


@app.function(image=image, timeout=180)
def check():
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '/opt/ta/usulan',
                    '-p', 'test_pseudo_teacher.py', '-v'], cwd='/opt/ta/usulan', check=True)


@app.function(image=image, gpu='T4', timeout=600, volumes={'/data': data, '/results': results})
def run(dataset: str, percent: int, method: str, run_id: str):
    import subprocess, sys
    data.reload()
    subprocess.run([sys.executable, '/opt/ta/usulan/pilot_teacher.py', '--dataset', dataset,
                    '--percent', str(percent),
                    '--output-root', f'/results/{run_id}'], cwd='/opt/ta', check=True)
    results.commit()


@app.local_entrypoint()
def main(action: str = 'check', run_id: str = 'tree-rankup-20261110', parallel: int = 3,
         dataset: str = 'all', percent: str = 'all', method: str = 'all'):
    if action == 'check':
        check.remote()
        return
    if action != 'run' or not 1 <= parallel <= 3 or not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid arguments')
    datasets = ('nhanes', 'pendigits', 'superconductivity')
    methods = ('tree_rankup',)
    if dataset != 'all' and dataset not in datasets or method != 'all' and method not in methods:
        raise ValueError('Invalid dataset or method')
    percents = (1, 10, 30) if percent == 'all' else (int(percent),)
    if any(p not in (1, 10, 30) for p in percents):
        raise ValueError('Invalid percentage')
    jobs = [(d, p, m) for d in (datasets if dataset == 'all' else (dataset,)) for p in percents
            for m in (methods if method == 'all' else (method,))]
    run.with_options(max_containers=parallel).spawn_map(
        [d for d, _, _ in jobs], [p for _, p, _ in jobs], [m for _, _, m in jobs], [run_id] * len(jobs))
    print(f'Queued {len(jobs)} validation-only jobs; at most {parallel} T4s', flush=True)
