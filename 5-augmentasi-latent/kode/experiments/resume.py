"""Persist completed trials and validate the inputs used when resuming a run."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def commit_progress():
    volume = os.environ.get('EXPERIMENT_TA_RESULTS_VOLUME')
    if volume:
        import modal
        modal.Volume.from_name(volume).commit()


def load_study(study, path):
    if path.exists():
        with path.open('rb') as stream:
            study = pickle.load(stream)
    return study


def save_study(study, path):
    atomic_write(path, pickle.dumps(study, protocol=pickle.HIGHEST_PROTOCOL))
    commit_progress()


def remaining_trials(study, target):
    finished = sum(t.state.name in ('COMPLETE', 'PRUNED') for t in study.trials)
    return max(0, target - finished)


def run_signature(dataset, method, data_root, tuning_seed, trials, optuna_epochs, epochs):
    from importlib import metadata
    files = {}
    root = Path(__file__).parent
    for path in sorted(root.rglob('*.py')):
        files[f'code/{path.relative_to(root)}'] = digest(path)
    data_path = Path(data_root) / dataset
    for path in sorted(data_path.glob('*')):
        if path.is_file():
            files[f'data/{path.name}'] = digest(path)
    versions = {}
    for name in ('torch', 'tabm', 'optuna', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'modal'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'dataset': dataset, 'method': method, 'tuning_seed': tuning_seed,
            'trials': trials, 'optuna_epochs': optuna_epochs, 'epochs': epochs,
            'python': sys.version, 'versions': versions, 'files': files}


def verify_signature(path, signature):
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError(f'Konfigurasi, kode, atau data berubah; gunakan run-id baru: {path}')
    else:
        atomic_write(path, (json.dumps(signature, indent=2) + '\n').encode())
        commit_progress()


def seed_files(output_dir, prefix, method):
    return [output_dir / f'{prefix}_{method}_{suffix}' for suffix in
            ('run_config.json', 'metrics.json', 'history.csv', 'predictions.npz', 'model.pt')]


def seed_complete(marker, files):
    if not marker.is_file() or any(not path.is_file() for path in files):
        return False
    try:
        return json.loads(marker.read_text()) == {path.name: digest(path) for path in files}
    except (OSError, ValueError):
        return False


def mark_seed_complete(marker, files):
    if any(not path.is_file() for path in files):
        raise ValueError(f'File hasil seed belum lengkap: {marker.parent}')
    atomic_write(marker, json.dumps({path.name: digest(path) for path in files}, indent=2).encode())
    commit_progress()
