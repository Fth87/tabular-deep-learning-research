"""Block-Gibbs neighbour augmentation, run through the existing RankUp training loops."""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler

FAMILIES = {'ordinal_regression': 'rankup', 'classification': 'rankup_classification', 'regression': 'regression'}


def block_gibbs(x, strength, train, k=5, pool=2048):
    """Replace a random feature block with the values of a neighbour found on the kept features.

    x_S is drawn from the k-NN estimate of p(x_S | x_{-S}), one block-Gibbs step, so the
    augmented row stays on the data distribution. S is independent of x.
    """
    width = x.shape[1]
    count = min(width, max(1, round(strength * width)))
    order = torch.rand(x.shape, device=x.device).argsort(dim=1)
    replaced = torch.zeros_like(x, dtype=torch.bool).scatter_(1, order[:, :count], True)
    candidates = train[torch.randint(len(train), (min(pool, len(train)),), device=x.device)]
    distance = ((x[:, None] - candidates) ** 2 * ~replaced[:, None]).sum(-1)
    distance[(x[:, None] == candidates).all(-1)] = float('inf')  # a copy of x changes nothing
    neighbours = distance.topk(min(k, len(candidates)), largest=False).indices
    donor = neighbours[torch.arange(len(x)), torch.randint(neighbours.shape[1], (len(x),), device=x.device)]
    return torch.where(replaced, candidates[donor], x)


def gibbs_chain(x, strength, train, k=5, steps=1):
    """Several block-Gibbs steps with fresh blocks: each keeps P(x), so the chain does too, but moves farther."""
    for _ in range(steps):
        x = block_gibbs(x, strength, train, k)
    return x


def calibrate_steps(train, rho, strength, k=5, max_steps=10, rows=1000):
    """Label-free number of steps whose displacement E|x~-x|^2 / E|x-x'|^2 is closest to rho."""
    generator = torch.Generator(device=train.device).manual_seed(0)
    pick = lambda: train[torch.randint(len(train), (rows,), device=train.device, generator=generator)]
    x, other = pick(), pick()
    scale = ((x - other) ** 2).sum(1).mean()
    current, gaps = x, []
    for _ in range(max_steps):
        current = block_gibbs(current, strength, train, k)
        gaps.append(abs(float(((current - x) ** 2).sum(1).mean() / scale) - rho))
    return int(np.argmin(gaps)) + 1


def calibrate(train, rho, k=5, rows=1000):
    """Label-free strength: the block size whose displacement E|x~-x|^2 / E|x-x'|^2 is closest to rho."""
    generator = torch.Generator(device=train.device).manual_seed(0)
    pick = lambda: train[torch.randint(len(train), (rows,), device=train.device, generator=generator)]
    x, other = pick(), pick()
    scale = ((x - other) ** 2).sum(1).mean()
    width = train.shape[1]
    displacement = [((block_gibbs(x, q / width, train, k) - x) ** 2).sum(1).mean() / scale
                    for q in range(1, width + 1)]
    return (int(torch.stack(displacement).sub(rho).abs().argmin()) + 1) / width


def load(root, name):
    with np.load(root / f'{name}.npz', allow_pickle=False) as split:
        return split['X'].astype(np.float32), (split['y'] if 'y' in split else None)


def scores(task, y, pred, levels=None):
    if task == 'classification':
        return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
                'accuracy': float(accuracy_score(y, pred))}
    result = {'mse': float(mean_squared_error(y, pred)), 'mae': float(mean_absolute_error(y, pred)),
              'r2': float(r2_score(y, pred)), 'spearman': float(spearmanr(y, pred).statistic)}
    if task == 'ordinal_regression':
        rounded = levels[np.abs(pred.reshape(-1, 1) - levels).argmin(axis=1)]
        result.update(qwk=float(cohen_kappa_score(y, rounded, labels=levels, weights='quadratic')),
                      ordinal_mae=float(mean_absolute_error(y, rounded)),
                      accuracy=float(accuracy_score(y, rounded)))
    return result


def fit_faithful(module, op_module, config, args, task, x_lab, y_lab, x_unlab, x_val, y_val,
                 epochs, n_classes=None, ema_decay=0.999, uratio=7):
    """RankUp loop as in the official USB code: EMA model for evaluation, uratio unlabeled batches."""
    import copy
    import random
    from itertools import islice
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = torch.from_numpy(np.concatenate([x_lab, x_unlab])).to(device)
    op = config['op']
    if op == 'gibbs':
        aug = lambda x, s, *_: block_gibbs(x, s, train, config['k'])
    elif op == 'none':
        aug = lambda x, *_: x
    else:
        aug = op_module.augment
    edges = op_module.make_quantile_edges(train) if op == 'range_limited' else None
    groups = op_module.make_feature_groups(train) if op == 'tabcutmix_plus' else None
    view = lambda x, s, pl, conf: x if s <= 0 else aug(x, s, train, pl, conf, args.donor_confidence, edges, groups)

    ranking = task != 'classification'
    if ranking:
        scaler = StandardScaler().fit(y_lab.reshape(-1, 1))
        y_scaled = scaler.transform(y_lab.reshape(-1, 1)).astype(np.float32)
        labeled = TensorDataset(torch.from_numpy(x_lab), torch.from_numpy(y_scaled),
                                torch.from_numpy(y_lab.astype(np.float32).reshape(-1, 1)))
        model = module.TabMRankUp(x_lab.shape[1]).to(device)
        bins = torch.as_tensor(np.quantile(y_lab, [0.25, 0.5, 0.75]), dtype=torch.float32, device=device)
    else:
        labeled = TensorDataset(torch.from_numpy(x_lab), torch.from_numpy(y_lab))
        model = module.TabMClassifier(x_lab.shape[1], n_classes).to(device)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
    labeled_loader = DataLoader(labeled, batch_size=args.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(TensorDataset(torch.from_numpy(x_unlab)),
                                  batch_size=args.batch_size * uratio, shuffle=True)
    labeled_batches = module.repeat_loader(labeled_loader)
    best, best_state, best_epoch, stale, history = -float('inf'), None, 0, 0, []
    steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        accepts = []
        for (x_u,) in islice(unlabeled_loader, args.max_steps):
            batch = [value.to(device) for value in next(labeled_batches)]
            x_u = x_u.to(device)
            with torch.no_grad():
                if ranking:
                    raw = model(x_u)[0].mean(dim=1).squeeze(-1)
                    score = raw * float(scaler.scale_[0]) + float(scaler.mean_[0])
                    if task == 'ordinal_regression':
                        pl, conf = score.round().long(), (1 - (score - score.round()).abs()).clamp(0, 1)
                    else:
                        pl, conf = torch.bucketize(score, bins), torch.ones_like(score)
                else:
                    conf, pl = model(x_u).mean(dim=1).softmax(-1).max(-1)
            weak, strong = view(x_u, config.get('weak', 0.0), pl, conf), view(x_u, config.get('strong', 0.0), pl, conf)
            if ranking:
                x_l, y_l_scaled, y_l = batch
                pred_l, rank_l = model(x_l)
                supervised = (F.mse_loss(pred_l, y_l_scaled[:, None, :].expand_as(pred_l))
                              + args.rank_weight * module.supervised_rank_loss(rank_l, y_l))
                with torch.no_grad():
                    rank_weak = model(weak)[1]
                unsupervised, accept = module.pseudo_rank_loss(rank_weak, model(strong)[1], args.confidence_cutoff)
            else:
                x_l, y_l = batch
                logits_l = model(x_l)
                supervised = F.cross_entropy(logits_l.reshape(-1, n_classes), y_l.repeat_interleave(logits_l.shape[1]))
                with torch.no_grad():
                    conf_w, pl_w = model(weak).mean(dim=1).softmax(-1).max(-1)
                logits_s = model(strong)
                each = F.cross_entropy(logits_s.reshape(-1, n_classes), pl_w.repeat_interleave(logits_s.shape[1]),
                                       reduction='none').reshape(len(x_u), -1).mean(dim=1)
                kept = conf_w >= args.confidence_cutoff
                unsupervised, accept = (each * kept).mean(), float(kept.float().mean())
            loss = supervised + args.unlabeled_weight * unsupervised
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            # Warm-up keeps the EMA from sitting on the random init in short schedules.
            decay = min(ema_decay, (1 + steps) / (10 + steps))
            with torch.no_grad():
                for shadow, value in zip(ema.parameters(), model.parameters()):
                    shadow.mul_(decay).add_(value.detach(), alpha=1 - decay)
                for shadow, value in zip(ema.buffers(), model.buffers()):
                    shadow.copy_(value)
            accepts.append(accept)

        if ranking:
            scores_val = module.predict(ema, x_val, scaler, device)
            metric = (module.score_qwk(y_val, scores_val, args.ordinal_levels) if task == 'ordinal_regression'
                      else -float(np.mean((scores_val - y_val) ** 2)))
        else:
            metric = float(f1_score(y_val, module.predict(ema, x_val, device), average='macro', zero_division=0))
        history.append({'epoch': epoch, 'pseudo_accept_rate': float(np.mean(accepts)), 'metric': metric})
        if metric > best:
            best, best_epoch, stale = metric, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    return best, best_epoch, best_state, history


def run_faithful(root, manifest, dataset, label_percent, config, seed, epochs, started):
    task = manifest['task_type']
    family = FAMILIES[task]
    module = importlib.import_module(f'experiments.{family}.scarf')
    op_module = (importlib.import_module(f'experiments.{family}.{config["op"]}')
                 if config['op'] not in ('none', 'gibbs') else None)
    x_lab, y_lab = load(root, 'train_labeled')
    x_unlab, _ = load(root, 'train_unlabeled')
    x_val, y_val = load(root, 'validation')
    x_test, y_test = load(root, 'test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = SimpleNamespace(batch_size=32, max_steps=100, patience=15, rank_weight=0.2, confidence_cutoff=0.95,
                           unlabeled_weight=0.0 if config['op'] == 'none' else 0.2, donor_confidence=0.8,
                           learning_rate=2e-3, seed=seed)
    levels, n_classes = None, None
    if task == 'classification':
        ids = {str(label): index for index, label in enumerate(manifest['classes'])}
        y_lab, y_val, y_test = (np.asarray([ids[v] for v in np.asarray(y).astype(str)], dtype=np.int64)
                                for y in (y_lab, y_val, y_test))
        n_classes = len(ids)
    else:
        y_lab, y_val, y_test = (np.asarray(v, dtype=np.float32).reshape(-1) for v in (y_lab, y_val, y_test))
        if task == 'ordinal_regression':
            levels = args.ordinal_levels = np.asarray(manifest['target_values'], dtype=np.int64)
    _, best_epoch, state, history = fit_faithful(module, op_module, config, args, task, x_lab, y_lab, x_unlab,
                                                 x_val, y_val, epochs, n_classes)
    if task == 'classification':
        model = module.TabMClassifier(x_lab.shape[1], n_classes).to(device)
        model.load_state_dict(state)
        predict = lambda x: module.predict(model, x, device)
    else:
        scaler = StandardScaler().fit(y_lab.reshape(-1, 1))
        model = module.TabMRankUp(x_lab.shape[1]).to(device)
        model.load_state_dict(state)
        predict = lambda x: module.predict(model, x, scaler, device)
    return {'dataset': dataset, 'label_percent': label_percent, 'config': config, 'seed': seed, 'task': task,
            'validation': scores(task, y_val, predict(x_val), levels),
            'test': scores(task, y_test, predict(x_test), levels),
            'best_epoch': int(best_epoch), 'epochs_run': len(history),
            'val_curve': [round(row['metric'], 4) for row in history],
            'accept_rate_last5': float(np.mean([row['pseudo_accept_rate'] for row in history[-5:]])),
            'seconds': time.time() - started, 'device': str(device)}


def run_one(data_root, dataset, label_percent, config, seed, epochs=120):
    """config: op (none|gibbs|an existing method), weak, strong, k for gibbs, protocol (old|faithful)."""
    started = time.time()
    root = Path(data_root) / f'label_{label_percent:02d}' / dataset
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    task = manifest['task_type']
    if config.get('protocol') == 'faithful':
        return run_faithful(root, manifest, dataset, label_percent, config, seed, epochs, started)
    op = config['op']
    # A baseline runs through its own module (its own fit_model and helpers); gibbs/none reuse scarf's loop.
    module = importlib.import_module(f'experiments.{FAMILIES[task]}.{"scarf" if op in ("none", "gibbs") else op}')
    # Containers reuse imported modules, so always start from the module's own functions.
    module.__dict__.setdefault('original_augment', module.augment)
    module.__dict__.setdefault('original_repeat_loader', module.repeat_loader)
    module.repeat_loader = module.original_repeat_loader
    if op == 'gibbs':
        def base(x, strength, train, *_):
            return gibbs_chain(x, strength, train, config['k'], config.get('steps', 1))
    else:
        base = module.original_augment
    ranking = task != 'classification'
    if ranking:
        module.__dict__.setdefault('original_pseudo_rank_loss', module.pseudo_rank_loss)
        module.pseudo_rank_loss = module.original_pseudo_rank_loss
    # Self-adaptive threshold (FreeMatch SAT): tau starts at 1/C and tracks the mean teacher confidence.
    sat = {'tau': 0.5 if ranking else 1 / len(manifest.get('classes', [0, 0])), 'momentum': 0.999}

    def track(mean_confidence):
        sat['tau'] = sat['momentum'] * sat['tau'] + (1 - sat['momentum']) * mean_confidence

    def augment(x, strength, *rest):
        if config.get('sat') and not ranking and strength == config.get('weak', 0.0):
            track(float(rest[2].mean()))  # rest[2] is the teacher confidence, one update per step
        return x if strength <= 0 else base(x, strength, *rest)
    module.augment = augment
    if config.get('sat') and ranking:
        def adaptive_rank_loss(weak, strong, _cutoff):
            with torch.no_grad():
                probabilities = torch.softmax(module.pairwise_logits(weak), dim=-1)
                probabilities = probabilities.mean(dim=0) if probabilities.ndim == 4 else probabilities
                track(float(probabilities.max(dim=-1).values.mean()))
            return module.original_pseudo_rank_loss(weak, strong, sat['tau'])
        module.pseudo_rank_loss = adaptive_rank_loss

    x_lab, y_lab = load(root, 'train_labeled')
    x_unlab, _ = load(root, 'train_unlabeled')
    x_val, y_val = load(root, 'validation')
    x_test, y_test = load(root, 'test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = torch.from_numpy(np.concatenate([x_lab, x_unlab])).to(device)
    config = dict(config)
    if 'rho' in config:
        config['strong'] = calibrate(train, config['rho'], config['k'])
    if 'rho_steps' in config:
        config['steps'] = calibrate_steps(train, config['rho_steps'], config['strong'], config['k'])
    if config.get('lab_rho'):
        # Block-Gibbs keeps P(x), so it is safe on labeled rows too (RankUp weak-augments labeled data).
        config['lab'] = calibrate(train, config['lab_rho'], config['k'])
        def augmented(loader):
            for x, *rest in module.original_repeat_loader(loader):
                yield (block_gibbs(x.to(device), config['lab'], train, config['k']), *rest)
        module.repeat_loader = augmented
    args = SimpleNamespace(
        batch_size=32, max_steps=100, patience=15, rank_weight=0.2, confidence_cutoff=0.95,
        unlabeled_weight=0.0 if op == 'none' else 0.2, donor_confidence=0.8,
        learning_rate=config.get('lr', 2e-3), seed=seed)
    # Protocol overrides apply to every augmentation alike.
    if op != 'none':
        args.unlabeled_weight = config.get('uw', args.unlabeled_weight)
    args.confidence_cutoff = config.get('cutoff', args.confidence_cutoff)
    if config.get('sat') and not ranking:
        class AdaptiveArgs(SimpleNamespace):
            confidence_cutoff = property(lambda self: sat['tau'])
        values = vars(args)
        values.pop('confidence_cutoff')
        args = AdaptiveArgs(**values)
    weak, strong = config.get('weak', 0.0), config.get('strong', 0.0)

    if task == 'classification':
        classes = [str(value) for value in manifest['classes']]
        ids = {label: index for index, label in enumerate(classes)}
        encode = lambda values: np.asarray([ids[label] for label in np.asarray(values).astype(str)], dtype=np.int64)
        y_lab, y_val, y_test = encode(y_lab), encode(y_val), encode(y_test)
        _, best_epoch, state, history = module.fit_model(
            args, x_lab, y_lab, x_unlab, x_val, y_val, len(classes), 'macro_f1',
            weak, strong, epochs, device)
        model = module.TabMClassifier(x_lab.shape[1], len(classes)).to(device)
        model.load_state_dict(state)
        predict = lambda x: module.predict(model, x, device)
        levels = None
    else:
        y_lab, y_val, y_test = (np.asarray(v, dtype=np.float32).reshape(-1) for v in (y_lab, y_val, y_test))
        levels = None
        if task == 'ordinal_regression':
            levels = args.ordinal_levels = np.asarray(manifest['target_values'], dtype=np.int64)
        _, best_epoch, state, history = module.fit_model(
            args, x_lab, y_lab, x_unlab, x_val, y_val, weak, strong, epochs)
        scaler = StandardScaler().fit(y_lab.reshape(-1, 1))
        model = module.TabMRankUp(x_lab.shape[1]).to(device)
        model.load_state_dict(state)
        predict = lambda x: module.predict(model, x, scaler, device)

    return {'dataset': dataset, 'label_percent': label_percent, 'config': config, 'seed': seed,
            'task': task, 'validation': scores(task, y_val, predict(x_val), levels),
            'test': scores(task, y_test, predict(x_test), levels),
            'best_epoch': int(best_epoch), 'epochs_run': len(history), 'final_tau': sat['tau'],
            'accept_rate_last5': float(np.mean([row['pseudo_accept_rate'] for row in history[-5:]])),
            'seconds': time.time() - started, 'device': str(device)}
