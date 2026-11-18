"""Record reproducibility metadata and model checkpoints for each experiment."""
from importlib import metadata
import json
from pathlib import Path


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Cannot encode {type(value).__name__}')


def save_run_config(output_dir, prefix, method, args, manifest):
    versions = {}
    for name in ('torch', 'tabm', 'optuna', 'numpy', 'scipy', 'scikit-learn', 'pandas'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    config = {
        'dataset': args.dataset, 'method': method,
        'stage': 'tuning' if args.tune_only else 'training',
        'arguments': vars(args), 'data_manifest': manifest,
        'optimizer': {'name': 'AdamW', 'weight_decay': 0.0003},
        'library_versions': versions,
    }
    path = output_dir / f'{prefix}_{method}_run_config.json'
    path.write_text(json.dumps(config, indent=2, default=_json_value) + '\n', encoding='utf-8')


def save_checkpoint(output_dir, prefix, method, model, n_features, seed,
                    weak_strength, strong_strength, classes=None, target_scaler=None):
    import torch
    checkpoint = {
        'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        'architecture': type(model).__name__, 'model_repr': repr(model),
        'n_features': int(n_features), 'seed': int(seed),
        'weak_strength': float(weak_strength), 'strong_strength': float(strong_strength),
        'classes': classes,
    }
    if target_scaler is not None:
        checkpoint['target_scaler'] = {'mean': target_scaler.mean_.tolist(),
                                       'scale': target_scaler.scale_.tolist(),
                                       'variance': target_scaler.var_.tolist()}
    path = output_dir / f'{prefix}_{method}_model.pt'
    torch.save(checkpoint, path)
