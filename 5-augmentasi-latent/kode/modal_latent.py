"""Menjalankan GRID di Modal (T4). Hasil ditambahkan ke ../hasil/results.jsonl; run yang sudah ada dilewati.

Dari folder kode/ (ganti <profil> dengan profil Modal-mu, atau hapus MODAL_PROFILE untuk profil bawaan):
  MODAL_PROFILE=<profil> modal run modal_latent.py                         # test + smoke run 2 epoch
  MODAL_PROFILE=<profil> modal run modal_latent.py --action run --names fase1
  MODAL_PROFILE=<profil> modal run modal_latent.py --action run --names angular,angular_weak --seeds 42
"""
import json
from pathlib import Path

import modal

from grid import DATASETS, GRID, GROUPS, PERCENTS, SEEDS

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / 'experiments'  # loop RankUp (disalin dari eksperimen sebelumnya, tidak diubah)
DATA = HERE.parent / 'data'          # split data per persentase label dan dataset
RESULTS = HERE.parent / 'hasil' / 'results.jsonl'

app = modal.App('experiment-latent')
image = (
    modal.Image.debian_slim(python_version='3.10')
    .pip_install('numpy<2', 'pandas', 'scikit-learn==1.3.2', 'scipy', 'joblib',
                 'torch==2.1.2', 'tabm', 'optuna==3.6.1')
    .add_local_dir(DATA, remote_path='/opt/ta/data')
    .add_local_dir(EXPERIMENTS, remote_path='/opt/ta/experiments', ignore=['**/__pycache__'])
    .add_local_file(HERE / 'latent.py', '/opt/ta/latent.py')
    .add_local_python_source('grid')  # modul ini juga di-import di container
    .add_local_file(HERE / 'test_latent.py', '/opt/ta/test_latent.py')
)


@app.function(image=image, gpu='T4', timeout=3600, scaledown_window=30)
def run_job(job):
    import sys
    sys.path.insert(0, '/opt/ta')
    from latent import run_one
    dataset, percent, name, seed, epochs = job
    return {'name': name, **run_one('/opt/ta/data', dataset, percent, GRID[name], seed, epochs)}


@app.function(image=image, gpu='T4', timeout=1800)
def check():
    import subprocess
    import sys
    subprocess.run([sys.executable, 'test_latent.py', '-v'], check=True, cwd='/opt/ta', env={'PYTHONPATH': '/opt/ta'})
    sys.path.insert(0, '/opt/ta')
    from latent import run_one
    for dataset, name in (('nhanes', 'both_tabcutmix'), ('pendigits', 'latent_scarf_weak'),
                          ('superconductivity', 'angular_weak'), ('nhanes', 'advrot_10'),
                          ('pendigits', 'advrot_10')):
        result = run_one('/opt/ta/data', dataset, 10, GRID[name], 42, epochs=2)
        print('smoke', dataset, name, result['test'], result['cos'], f"{result['seconds']:.0f}s", flush=True)


def expand(names: str) -> list:
    """'fase1', 'all', atau daftar nama/grup dipisah koma."""
    if names == 'all':
        return list(GRID)
    picked = [n for part in names.split(',') for n in GROUPS.get(part, [part])]
    unknown = [n for n in picked if n not in GRID]
    if unknown:
        raise SystemExit(f'nama tidak ada di GRID: {unknown}')
    return picked


@app.local_entrypoint()
def main(action: str = 'check', names: str = 'fase1', seeds: str = '', percents: str = '', datasets: str = '',
         epochs: int = 120, containers: int = 8):
    if action == 'check':
        check.remote()
        return
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done.add((row['dataset'], row['label_percent'], row['name'], row['seed']))
    pick = lambda text, default, cast=str: [cast(v) for v in text.split(',')] if text else list(default)
    jobs = [(d, p, n, s, epochs) for p in pick(percents, PERCENTS, int) for d in pick(datasets, DATASETS)
            for n in expand(names) for s in pick(seeds, SEEDS, int) if (d, p, n, s) not in done]
    print(f'{len(jobs)} run baru, {len(done)} sudah ada, {containers} container', flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open('a', encoding='utf-8') as sink:
        for result in run_job.with_options(max_containers=containers).map(
                jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                print('GAGAL', repr(result), flush=True)
                continue
            sink.write(json.dumps(result) + '\n')
            sink.flush()
            print('selesai', result['dataset'], result['label_percent'], result['name'], result['seed'],
                  f"{result['seconds']:.0f}s", flush=True)
    print('SEMUA SELESAI', flush=True)
