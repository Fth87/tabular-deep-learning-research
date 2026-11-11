"""Run the Task 3 deep SSL baselines (VIME, SSLAE, STUNT, GFTab) on the Experiment TA splits.

They are classifiers, so only NHANES (ordinal as ordered classes) and Pen Digits are run.
Results come back to eksperimen/hasil/pembanding/; no Modal volume is written.
Usage: MODAL_PROFILE=lagi uv run modal run eksperimen/modal_pembanding.py
"""
import io
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
TASK3 = ROOT.parent.parent / 'task 3' / 'experiments'
app = modal.App('experiment-ta-pembanding')


def with_code(image, name):
    return (image.add_local_dir(ROOT / 'hasil' / 'data-5seed', remote_path='/opt/data',
                                ignore=['**/superconductivity/**', '**/*.joblib'])
            .add_local_file(TASK3 / '__init__.py', remote_path='/opt/task3/experiments/__init__.py')
            .add_local_dir(TASK3 / name, remote_path=f'/opt/task3/experiments/{name}'))


# Same pinned images as task 3/modal_app.py.
IMAGES = {
    'sslae': modal.Image.debian_slim(python_version='3.10').pip_install(
        'numpy<2', 'pandas', 'scikit-learn==1.3.2', 'joblib', 'torch==2.1.2', 'pytorch-lightning==2.0.6',
        'rich', 'torchmetrics==1.8.2'),
    'vime': modal.Image.debian_slim(python_version='3.10').pip_install(
        'numpy<2', 'scikit-learn==1.3.2', 'tensorflow[and-cuda]==2.15.1'),
    'stunt': modal.Image.debian_slim(python_version='3.10').pip_install(
        'numpy<2', 'scikit-learn==1.3.2', 'torch==2.1.2'),
    'gftab': modal.Image.debian_slim(python_version='3.10').pip_install(
        'numpy<2', 'pandas', 'scikit-learn==1.3.2', 'torch==2.1.2', 'xgboost==2.1.4', 'einops==0.7.0',
        'scipy', 'matplotlib'),
}


def _run(method, dataset, percent):
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', f'experiments.{method}.run', '--dataset', dataset,
                    '--data-root', f'/opt/data/label_{percent:02d}', '--output-root', '/tmp/out'],
                   check=True, cwd='/opt/task3')
    folder = Path('/tmp/out') / dataset
    return {'metrics': json.loads((folder / f'{method}_metrics.json').read_text()),
            'predictions': (folder / f'{method}_predictions.npz').read_bytes()}


# T4 is cheapest; L4 is the fallback when Modal has no T4 capacity.
OPTIONS = dict(gpu=['T4', 'L4'], timeout=7200, scaledown_window=30)


@app.function(image=with_code(IMAGES['sslae'], 'sslae'), **OPTIONS)
def run_sslae(dataset, percent):
    return _run('sslae', dataset, percent)


@app.function(image=with_code(IMAGES['vime'], 'vime'), **OPTIONS)
def run_vime(dataset, percent):
    return _run('vime', dataset, percent)


@app.function(image=with_code(IMAGES['stunt'], 'stunt'), **OPTIONS)
def run_stunt(dataset, percent):
    return _run('stunt', dataset, percent)


@app.function(image=with_code(IMAGES['gftab'], 'gftab'), **OPTIONS)
def run_gftab(dataset, percent):
    return _run('gftab', dataset, percent)


FUNCTIONS = {'sslae': run_sslae, 'vime': run_vime, 'stunt': run_stunt, 'gftab': run_gftab}


@app.local_entrypoint()
def main(methods: str = 'vime,sslae,stunt,gftab', datasets: str = 'nhanes,pendigits', percents: str = '1,10,30'):
    out = ROOT / 'hasil' / 'pembanding'
    out.mkdir(parents=True, exist_ok=True)
    calls = []
    for method in methods.split(','):
        for dataset in datasets.split(','):
            for percent in map(int, percents.split(',')):
                if (out / f'{method}_{dataset}_{percent:02d}.json').exists():
                    continue
                calls.append(((method, dataset, percent), FUNCTIONS[method].spawn(dataset, percent)))
    print(f'{len(calls)} jobs', flush=True)
    for (method, dataset, percent), call in calls:
        try:
            result = call.get()
        except Exception as error:  # keep the other baselines when one fails
            print('FAILED', method, dataset, percent, repr(error)[:300], flush=True)
            continue
        stem = out / f'{method}_{dataset}_{percent:02d}'
        stem.with_suffix('.npz').write_bytes(result['predictions'])
        stem.with_suffix('.json').write_text(json.dumps(result['metrics'], indent=2))
        print('done', method, dataset, percent, flush=True)
    print('ALL FINISHED', flush=True)
