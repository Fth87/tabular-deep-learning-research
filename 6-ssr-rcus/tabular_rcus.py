"""SSR-RCUS resmi (paper_repo/, tidak diubah) dijalankan pada tiga jenis dataset proyek dengan backbone TabM.

Regresi (Superconductivity) dan ordinal (NHANES): algoritma resmi apa adanya (train_step, loss, manifold mixup ARC,
hook pseudo-rank, EMA, hiperparameter config resmi). File ini hanya adaptor:

    TabMNet         TabM dengan interface net semilearn (sama seperti WideResNet: logits, feat, only_feat, only_fc)
    TabularDataset  split .npz proyek -> batch semilearn (x_lb, y_lb / x_ulb_w, x_ulb_s)
    TabularData     pengganti set_dataset resmi (eval = validation, test terpisah)
    LatentViews     operator latent proyek (angular, ...) di output TabM untuk view weak/strong
    run_one         satu run: config resmi + penyesuaian tabular, train, lalu skor validation/test

Klasifikasi (Pen Digits): kode resmi hanya untuk regresi, jadi dibuat padanan terdekat yang memakai bagian resmi
sebanyak mungkin (hook masking/pseudo-label, ce_loss, cls_consistency_loss, EMA, sampling λ, bobot loss):
    TabularRankUpCls  RankUp-CLS proyek: CE berlabel + 0,2 × CE view strong dengan pseudo-kelas yakin (≥ 0,95)
    TabularRCUSCls    SSR-RCUS-CLS: manifold mixup semua pasangan fitur strong; target = campuran pseudo-kelas
                      kedua sampel dengan λ yang sama (pasangan dipakai hanya jika keduanya yakin)

Butuh paper_repo/ (siapkan_paper_repo.sh) dan ../5-augmentasi-latent/kode (operator, metrik, data).
"""
from __future__ import annotations

import functools
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
PAPER = HERE / 'paper_repo'
PROJECT = HERE.parent / '5-augmentasi-latent' / 'kode'
DATA = HERE.parent / '5-augmentasi-latent' / 'data'
for path in (PAPER, PROJECT, HERE):
    sys.path.insert(0, str(path))

from semilearn.algorithms.rankup.rankup import RankUp  # noqa: E402  (kode resmi, kontrol di pipeline yang sama)
from semilearn.algorithms.rcus.rcus import RCUS  # noqa: E402  (kode resmi)
from experiments.usulan import FAMILIES, load, scores  # noqa: E402  (loader dan metrik sama dengan 945 run latent)
from latent import perturb  # noqa: E402
from varian import INPUT_OPS, LATENT_OPS, VARIANTS  # noqa: E402

BASE_CONFIG = PAPER / 'config' / 'classic_cv' / 'rcus' / 'rcus_utkface_lb50_s0.yaml'

# Hanya bagian yang harus berubah karena data dan backbone berbeda. Hiperparameter algoritma (alpha 0,5, T 0,5,
# p_cutoff 0,95, rasio loss ARC 0,2/1,0, EMA 0,999, batch 32, uratio 7, criterion l1, include_lb_to_ulb) tetap resmi.
TABULAR = {
    'net': 'tabm',
    # anggaran sama dengan loop TabM proyek: 120 epoch × 100 langkah, evaluasi validation tiap epoch
    'epoch': 120, 'num_train_iter': 12000, 'num_eval_iter': 100, 'num_log_iter': 100,
    # resep optimizer TabM proyek (SGD 0,01 resmi untuk WRN gambar); LR konstan: multistep dengan gamma 1
    'optim': 'AdamW', 'lr': 2e-3, 'weight_decay': 3e-4, 'sch': 'multistep', 'milestones': [12000], 'gamma': 1.0,
    'num_workers': 2, 'use_tensorboard': False, 'resume': False, 'overwrite': True,
}


def read_split(root: Path, name: str, manifest: dict):
    """(X float32, y). y: float32 untuk regresi/ordinal; indeks kelas (urutan manifest) untuk klasifikasi."""
    x, y = load(root, name)
    if y is None:
        return x, None
    if manifest['task_type'] == 'classification':
        ids = {str(label): index for index, label in enumerate(manifest['classes'])}
        return x, np.asarray([ids[str(v)] for v in y], dtype=np.float32)  # float: scaler resmi jalan di semua target
    return x, np.asarray(y, dtype=np.float32)


def read_manifest(root: Path) -> dict:
    return json.loads((root / 'manifest.json').read_text(encoding='utf-8'))


def split_root(args) -> Path:
    return Path(args.data_dir) / f'label_{args.label_percent:02d}' / args.dataset


class TabMNet(nn.Module):
    """TabM default proyek (3 blok × 512, k = 32) dengan interface net semilearn.

    feat = rata-rata embedding k submodel (B, 512); head = Linear(512, n_out) seperti WRN resmi (n_out = 1 untuk
    regresi, jumlah kelas untuk klasifikasi); head ARC dibuat RankUp_Net resmi dari num_features.
    shortcut: TabM asli punya head dan loss per submodel, sedangkan algoritma resmi butuh satu feat dan satu prediksi,
    jadi submodel digabung dengan rata-rata di feat. Ganti hanya kalau algoritmanya juga diubah.
    """

    def __init__(self, n_features: int, n_out: int = 1, pretrained: bool = False, pretrained_path=None):
        super().__init__()
        assert not pretrained, 'TabM tidak punya bobot pretrained'
        from tabm import TabM
        self.tabm = TabM.make(n_num_features=n_features, d_out=None)  # output (B, k, d)
        self.num_features = self.tabm.backbone.get_original_output_shape()[0]
        self.regressor = nn.Linear(self.num_features, n_out)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.regressor(x).squeeze(-1)
        feat = self.tabm(x).mean(dim=1)
        if only_feat:
            return feat
        return {'logits': self.regressor(feat).squeeze(-1), 'feat': feat}


class TabularDataset(Dataset):
    """Satu baris -> dict batch semilearn. Data berlabel/eval: x_lb, y_lb. Unlabeled: x_ulb_w (asli), x_ulb_s."""

    def __init__(self, x: np.ndarray, targets: np.ndarray, is_ulb: bool = False, strong=None):
        self.data, self.targets, self.is_ulb, self.strong = x, targets, is_ulb, strong

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        if not self.is_ulb:
            return {'idx_lb': idx, 'x_lb': x, 'y_lb': self.targets[idx]}  # targets sudah lewat scaler AlgorithmBase
        return {'idx_ulb': idx, 'x_ulb_w': x, 'x_ulb_s': self.strong(x) if self.strong else x.clone()}


def input_augment(task: str, op: str, strength: float, train_features: np.ndarray):
    """Operator input proyek yang sama persis (experiments/<family>/<op>.py), dipakai per baris seperti transform."""
    assert op in INPUT_OPS, f'operator input tidak didukung: {op!r}'
    method = importlib.import_module(f'experiments.{FAMILIES[task]}.{op}').augment  # ada di ketiga keluarga
    pool = torch.from_numpy(train_features)
    return lambda x: method(x[None], strength, pool, None, None, 0.8, None, None)[0]


class TabularData:
    """Pengganti set_dataset resmi: split tabular proyek, eval = validation, test terpisah.
    (Config resmi memakai test sebagai 'eval', jadi checkpoint terbaik dipilih dengan test.)"""

    def set_dataset(self):
        args = self.args
        root = split_root(args)
        manifest = read_manifest(root)
        x_lab, y_lab = read_split(root, 'train_labeled', manifest)
        x_unlab, _ = read_split(root, 'train_unlabeled', manifest)
        x_ulb = np.concatenate([x_unlab, x_lab]) if args.include_lb_to_ulb else x_unlab
        strong = (input_augment(manifest['task_type'], args.input_op, args.input_strong, x_ulb)
                  if getattr(args, 'input_op', None) else None)
        # target unlabeled tidak ada di split; NaN supaya ketahuan kalau ada yang memakainya
        ulb = TabularDataset(x_ulb, np.full(len(x_ulb), np.nan, dtype=np.float32), is_ulb=True, strong=strong)
        args.ulb_dest_len, args.lb_dest_len = len(x_ulb), len(x_lab)
        return {'train_lb': TabularDataset(x_lab, y_lab), 'train_ulb': ulb,
                'eval': TabularDataset(*read_split(root, 'validation', manifest)),
                'test': TabularDataset(*read_split(root, 'test', manifest))}


class TabularRCUS(TabularData, RCUS):
    """SSR-RCUS resmi dengan data tabular."""


class TabularRankUp(TabularData, RankUp):
    """RankUp resmi dengan data tabular: kontrol tanpa mixup SSR-RCUS di pipeline yang sama persis."""


class TabularCls(TabularData):
    """Bagian AlgorithmBase resmi yang khusus regresi, diganti untuk klasifikasi."""

    def set_scaler(self, input_range=(0, 1), output_range=(0, 1)):
        return super().set_scaler((0, 1), (0, 1))  # identitas: label kelas tidak di-min-max

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """Sama dengan evaluate resmi (bobot EMA), tetapi skornya macro-F1. Hook resmi memilih checkpoint dengan
        'eval/mae' terkecil, jadi kunci itu diisi 1 − macro-F1 (checkpoint terbaik = macro-F1 validation tertinggi)."""
        from sklearn.metrics import accuracy_score, f1_score
        self.model.eval()
        self.ema.apply_shadow()
        predictions, labels = [], []
        with torch.no_grad():
            for data in self.loader_dict[eval_dest]:
                x = data['x_lb'].cuda(self.gpu) if self.gpu is not None else data['x_lb']
                predictions.append(self.model(x)[out_key].argmax(dim=-1).cpu())
                labels.append(data['y_lb'].long())
        self.ema.restore()
        self.model.train()
        y_pred, y_true = torch.cat(predictions).numpy(), torch.cat(labels).numpy()
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        result = {f'{eval_dest}/mae': 1.0 - macro_f1, f'{eval_dest}/macro_f1': macro_f1,
                  f'{eval_dest}/accuracy': accuracy_score(y_true, y_pred)}
        if return_logits:
            result[f'{eval_dest}/logits'] = y_pred
        return result

    def weak_targets(self, x_ulb_w):
        """Pseudo-kelas dan mask dari view weak, lewat hook resmi (sama seperti train_step resmi)."""
        self.bn_controller.freeze_bn(self.model)
        with torch.no_grad():
            logits_w = self.model(x_ulb_w)['logits']
        self.bn_controller.unfreeze_bn(self.model)
        mask = self.call_hook('masking', 'MaskingHook', logits_x_ulb=logits_w, softmax_x_ulb=True)
        pseudo = self.call_hook('gen_ulb_targets', 'PseudoLabelingHook', logits=logits_w,
                                use_hard_label=self.use_hard_label, T=self.T, softmax=True)
        return pseudo, mask


class TabularRankUpCls(TabularCls, RankUp):
    """RankUp-CLS (FixMatch dengan pseudo-kelas), bobot unlabeled = arc_loss_ratio × arc_ulb_loss_ratio = 0,2,
    sama dengan RankUp-CLS di eksperimen sebelumnya."""

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        with self.amp_cm():
            sup_loss = self.ce_loss(self.model(x_lb)['logits'], y_lb.long(), reduction='mean')
            pseudo, mask = self.weak_targets(x_ulb_w)
            unsup_loss = self.cls_consistency_loss(self.model(x_ulb_s)['logits'], pseudo, 'ce', mask=mask)
            total_loss = sup_loss + self.arc_loss_ratio * self.arc_ulb_loss_ratio * unsup_loss
        return (self.process_out_dict(loss=total_loss),
                self.process_log_dict(sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                                      total_loss=total_loss.item(), util_ratio=mask.float().mean().item()))


class TabularRCUSCls(TabularCls, RCUS):
    """SSR-RCUS-CLS. Padanan arc_manifold_mixup resmi untuk kelas: semua pasangan (i, j) dicampur
    m_ij = λ·z_j + (1−λ)·z_i (λ ~ Beta(α, α), λ = max(λ, 1−λ), seperti resmi), target = λ·y_j + (1−λ)·y_i.
    Asumsi monotonik regresi diganti asumsi mixup biasa (label berubah linear di antara dua sampel)."""

    def class_mixup(self, feats, targets, mask=None):
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)
        mixed = lam * feats.unsqueeze(0) + (1.0 - lam) * feats.unsqueeze(1)          # [i, j] = λ z_j + (1−λ) z_i
        onehot = F.one_hot(targets.long(), self.model.backbone.regressor.out_features).float()
        soft = lam * onehot.unsqueeze(0) + (1.0 - lam) * onehot.unsqueeze(1)
        logits = self.model.backbone(mixed, only_fc=True).flatten(0, 1)
        pair_mask = None if mask is None else (mask.unsqueeze(0) * mask.unsqueeze(1)).flatten()
        return self.cls_consistency_loss(logits, soft.flatten(0, 1), 'ce', mask=pair_mask)

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        with self.amp_cm():
            out_lb = self.model(x_lb)
            sup_loss = self.ce_loss(out_lb['logits'], y_lb.long(), reduction='mean')
            pseudo, mask = self.weak_targets(x_ulb_w)
            feats_s = self.model(x_ulb_s)['feat']
            mixup_sup = self.class_mixup(out_lb['feat'], y_lb)
            mixup_unsup = self.class_mixup(feats_s, pseudo, mask)
            # bobot resmi SSR-RCUS: 0,2 × mixup berlabel + 1,0 × mixup unlabeled
            total_loss = sup_loss + self.arc_lb_loss_ratio * mixup_sup + self.arc_ulb_loss_ratio * mixup_unsup
        return (self.process_out_dict(loss=total_loss),
                self.process_log_dict(sup_loss=(sup_loss + mixup_sup).item(), unsup_loss=mixup_unsup.item(),
                                      total_loss=total_loss.item(), util_ratio=mask.float().mean().item()))


class LatentViews:
    """Operator latent proyek (latent.perturb) di output TabM, per submodel, sebelum rata-rata. Hanya view weak/strong
    unlabeled yang diganggu; train_step algoritma dipanggil tanpa perubahan."""

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        def hook(_module, inputs, h):
            role = 'weak' if inputs[0] is x_ulb_w else 'strong' if inputs[0] is x_ulb_s else None
            strength = getattr(self.args, f'latent_{role}', None) if role else None
            return perturb(self.args.latent_op, h, strength) if strength else None

        handle = self.model.backbone.tabm.register_forward_hook(hook)
        try:
            return super().train_step(x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s)
        finally:
            handle.remove()


ALGORITHMS = {('rankup', False): TabularRankUp, ('rcus', False): TabularRCUS,
              ('rankup', True): TabularRankUpCls, ('rcus', True): TabularRCUSCls}


def write_config(dataset: str, label_percent: int, variant: str, seed: int, data_root, work_dir, **override) -> Path:
    """Config resmi UTKFace + TABULAR + operator varian, disimpan sebagai YAML seperti config resmi."""
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    config = yaml.load(BASE_CONFIG.read_text(encoding='utf-8'))
    name = f'{variant}_{dataset}_lb{label_percent:02d}_s{seed}'
    extra = {k: list(v) if isinstance(v, tuple) else v for k, v in VARIANTS[variant].items()}
    assert extra.get('latent_op') in (None, *LATENT_OPS), extra
    config.update(TABULAR, **extra, dataset=dataset, label_percent=label_percent, seed=seed, data_dir=str(data_root),
                  save_dir=str(work_dir), save_name=name, load_path=str(Path(work_dir) / name / 'latest_model.pth'))
    config.update(override)
    path = Path(work_dir) / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as sink:
        yaml.dump(config, sink)
    return path


def build(config_path: Path):
    """Sama dengan train.py resmi (main_worker, 1 GPU), kecuali net builder TabM dan kelas data tabular."""
    import torch.backends.cudnn as cudnn
    import train as official
    from semilearn.core.utils import get_logger

    argv, sys.argv = sys.argv, ['train.py', '--c', str(config_path)]
    try:
        args = official.get_config()
    finally:
        sys.argv = argv
    args.distributed, args.gpu = False, (0 if torch.cuda.is_available() else None)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic, cudnn.benchmark = True, True
    logger = get_logger(args.save_name, str(Path(args.save_dir) / args.save_name), 'INFO')
    manifest = read_manifest(split_root(args))
    classification = manifest['task_type'] == 'classification'
    algorithm = ALGORITHMS[args.algorithm, classification]
    if getattr(args, 'latent_op', None):
        assert args.algorithm == 'rcus', 'operator latent hanya dipasang di SSR-RCUS'
        algorithm = type(f'{algorithm.__name__}Latent', (LatentViews, algorithm), {})
    n_features = read_split(split_root(args), 'validation', manifest)[0].shape[1]
    net = functools.partial(TabMNet, n_features, len(manifest['classes']) if classification else 1)
    return algorithm(args, net, None, logger)


def run_one(dataset: str, label_percent: int, variant: str, seed: int, data_root=DATA, work_dir='/tmp/rcus',
            **override) -> dict:
    from semilearn.core.utils import send_model_cuda

    started = time.time()
    alg = build(write_config(dataset, label_percent, variant, seed, data_root, work_dir, **override))
    alg.model = send_model_cuda(alg.args, alg.model)
    alg.ema_model = send_model_cuda(alg.args, alg.ema_model, clip_batch=False)
    alg.train()
    # Hook resmi memuat model_best.pth setelah training, tetapi shadow EMA yang dipakai evaluate() masih bobot EMA
    # terakhir. Samakan dulu supaya validation/test benar-benar dari checkpoint terbaik (dipilih dengan validation).
    alg.load_model(str(Path(alg.save_dir) / alg.save_name / 'model_best.pth'))
    alg.ema.load(alg.ema_model)
    root = split_root(alg.args)
    manifest = read_manifest(root)
    task = manifest['task_type']
    levels = np.asarray(manifest['target_values'], dtype=np.int64) if task == 'ordinal_regression' else None
    result = {}
    for split, dest in (('validation', 'eval'), ('test', 'test')):
        prediction = np.asarray(alg.evaluate(dest, return_logits=True)[f'{dest}/logits'])
        truth = read_split(root, split, manifest)[1]
        if task == 'classification':
            prediction, truth = prediction.astype(np.int64), truth.astype(np.int64)
        result[split] = scores(task, truth.astype(prediction.dtype), prediction, levels)
    return {'name': variant, 'dataset': dataset, 'label_percent': label_percent, 'seed': seed, 'task': task,
            'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in VARIANTS[variant].items()}, **result,
            'best_it': int(alg.best_it), 'best_val_selection': float(alg.best_eval_mae),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'seconds': time.time() - started}
