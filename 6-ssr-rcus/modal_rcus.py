"""Menjalankan baseline SSR-RCUS tabular di Modal (T4). Hasil ditambahkan ke hasil/results.jsonl; run yang sudah ada dilewati.

Dari folder 6-ssr-rcus/ (jalankan ./siapkan_paper_repo.sh sekali dulu):
  MODAL_PROFILE=lombaaaa modal run modal_rcus.py                                   # unit test + smoke 200 langkah
  MODAL_PROFILE=lombaaaa modal run modal_rcus.py --action run --names rcus --seeds 42 --percents 1
  MODAL_PROFILE=lombaaaa modal run modal_rcus.py --action run --names all          # 7 varian × 3 dataset × 3 label × 3 seed
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / '5-augmentasi-latent'
RESULTS = HERE / 'hasil' / 'results.jsonl'
REMOTE = '/opt/ta/6-ssr-rcus'  # struktur folder sama dengan lokal, jadi path relatif di tabular_rcus.py tetap berlaku
SEEDS, PERCENTS = (42, 123, 456), (1, 10, 30)

if modal.is_local() and not (HERE / 'paper_repo' / 'train.py').exists():  # di container, path-nya beda
    raise SystemExit('paper_repo/ belum ada: jalankan ./siapkan_paper_repo.sh')

app = modal.App('experiment-ssr-rcus')
image = (
    modal.Image.debian_slim(python_version='3.10')
    .pip_install_from_requirements(str(HERE / 'paper_repo' / 'requirements.txt'))  # dependensi resmi
    .pip_install('tabm==0.0.3', 'optuna==3.6.1', 'numpy<2')  # TabM dan modul operator proyek
    .add_local_dir(HERE / 'paper_repo', f'{REMOTE}/paper_repo', ignore=['.git', '.git/**', '**/__pycache__'])
    .add_local_file(HERE / 'tabular_rcus.py', f'{REMOTE}/tabular_rcus.py')
    .add_local_file(HERE / 'test_rcus.py', f'{REMOTE}/test_rcus.py')
    .add_local_file(HERE / 'varian.py', f'{REMOTE}/varian.py')
    .add_local_dir(PROJECT / 'kode' / 'experiments', '/opt/ta/5-augmentasi-latent/kode/experiments',
                   ignore=['**/__pycache__'])
    .add_local_file(PROJECT / 'kode' / 'latent.py', '/opt/ta/5-augmentasi-latent/kode/latent.py')
    .add_local_file(PROJECT / 'kode' / 'grid.py', '/opt/ta/5-augmentasi-latent/kode/grid.py')
    .add_local_dir(PROJECT / 'data', '/opt/ta/5-augmentasi-latent/data')
)


@app.function(image=image, gpu='T4', cpu=4, timeout=3600, scaledown_window=30)
def run_job(job):
    import sys
    sys.path.insert(0, REMOTE)
    from tabular_rcus import run_one
    dataset, percent, name, seed, override = job
    return run_one(dataset, percent, name, seed, **override)


@app.function(image=image, gpu='T4', cpu=4, timeout=1800)
def check():
    import subprocess
    import sys
    subprocess.run([sys.executable, 'test_rcus.py', '-v'], check=True, cwd=REMOTE)
    sys.path.insert(0, REMOTE)
    from tabular_rcus import run_one
    for dataset, name in (('nhanes', 'rcus'), ('superconductivity', 'rcus_angular'), ('pendigits', 'rcus_input_scarf')):
        result = run_one(dataset, 10, name, 42, epoch=2, num_train_iter=200, milestones=[200])
        print('smoke', dataset, name, result['validation'], result['test'], f"{result['seconds']:.0f}s", flush=True)


@app.local_entrypoint()
def main(action: str = 'check', names: str = 'all', seeds: str = '', percents: str = '', datasets: str = '',
         containers: int = 8):
    from varian import DATASETS, VARIANTS  # tanpa torch

    if action == 'check':
        check.remote()
        return
    picked = list(VARIANTS) if names == 'all' else names.split(',')
    unknown = [n for n in picked if n not in VARIANTS]
    if unknown:
        raise SystemExit(f'varian tidak dikenal: {unknown}; pilihan: {list(VARIANTS)}')
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done.add((row['dataset'], row['label_percent'], row['name'], row['seed']))
    pick = lambda text, default, cast=str: [cast(v) for v in text.split(',')] if text else list(default)
    jobs = [(d, p, n, s, {}) for p in pick(percents, PERCENTS, int) for d in pick(datasets, DATASETS)
            for n in picked for s in pick(seeds, SEEDS, int) if (d, p, n, s) not in done]
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
