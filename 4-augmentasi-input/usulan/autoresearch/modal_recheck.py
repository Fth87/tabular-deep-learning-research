"""Six bounded jobs for a paired seed-43 validation check, label fraction 1%."""
from pathlib import Path
import modal

path = Path(__file__).resolve()
root = path.parents[2] if len(path.parents) > 2 else Path('/')
app = modal.App('ta-autoresearch-recheck')
data = modal.Volume.from_name('experiment-ta-data')
results = modal.Volume.from_name('experiment-ta-results')
image = (modal.Image.debian_slim(python_version='3.10')
         .pip_install('numpy<2', 'pandas', 'scikit-learn==1.3.2', 'scipy', 'joblib',
                      'torch==2.1.2', 'tabm', 'optuna==3.6.1')
         .pip_install('modal>=1,<2')
         .add_local_dir(root / 'eksperimen/experiments', remote_path='/opt/ta/experiments')
         .add_local_dir(root / 'usulan', remote_path='/opt/ta/usulan',
                        ignore=['hasil/**', '__pycache__/**']))


@app.function(image=image, gpu='T4', timeout=600, volumes={'/data': data, '/results': results})
def run(dataset: str, method: str, run_id: str):
    import subprocess, sys
    data.reload()
    subprocess.run([sys.executable, '/opt/ta/usulan/autoresearch/runner_recheck.py', '--dataset', dataset,
                    '--method', method, '--output-root', f'/results/{run_id}'], cwd='/opt/ta', check=True)
    results.commit()


@app.local_entrypoint()
def main(run_id: str = 'autoresearch-recheck-seed43-20261111', parallel: int = 2):
    if not 1 <= parallel <= 2 or not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid arguments')
    jobs = [(d, m) for d in ('nhanes', 'pendigits', 'superconductivity') for m in ('baseline', 'candidate')]
    run.with_options(max_containers=parallel).spawn_map([d for d, _ in jobs], [m for _, m in jobs], [run_id] * len(jobs))
    print(f'Queued {len(jobs)} paired jobs, max {parallel} T4 GPUs', flush=True)
