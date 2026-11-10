"""Validation-only experiments with conditional augmentation and optional soft rank."""
import argparse
import importlib
import json
import tempfile
from pathlib import Path
import pilot
from conditional_aug import ConditionalAug, soft_rank_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=pilot.FAMILIES, required=True)
    parser.add_argument('--percent', type=int, choices=(1, 10, 30), required=True)
    parser.add_argument('--method', choices=('conditional_random', 'conditional_group', 'conditional_group_soft'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.method.endswith('_soft') and args.dataset == 'pendigits':
        raise ValueError('Pairwise soft rank is only for ordinal and regression')
    module = importlib.import_module(f'experiments.{pilot.FAMILIES[args.dataset]}.gaussian')
    if args.method.endswith('_soft'):
        module.pseudo_rank_loss = lambda weak, strong, cutoff: soft_rank_loss(module, weak, strong, cutoff)
    pilot.LocalTransport = lambda **kwargs: ConditionalAug(grouped=args.method != 'conditional_random')
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        result = pilot.run(args.dataset, args.percent, 'local_transport', Path('/data'), temporary,
                           epochs=120, max_steps=100)
        src = temporary / f'label_{args.percent:02d}' / args.dataset
        out = args.output_root / f'label_{args.percent:02d}' / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{args.method}_history.csv').write_bytes((src / 'local_transport_history.csv').read_bytes())
        result['method'] = args.method
        result['pilot_config'].pop('radius', None)
        result['pilot_config']['effective_weak_strength'] = 0.0
        result['pilot_config']['strong_strength'] = .3
        result['pilot_config']['donor_pool'] = 2048
        result['pilot_config']['donor_k'] = 5
        if args.method.endswith('_soft'):
            result['mean_pseudo_weight'] = result.pop('mean_pseudo_accept_rate')
            result['best_epoch_pseudo_weight'] = result.pop('best_epoch_pseudo_accept_rate')
            result['history_pseudo_accept_rate_is_mean_soft_weight'] = True
        (out / f'{args.method}_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print('COMPLETED', args.dataset, args.percent, args.method, result['validation'], flush=True)


if __name__ == '__main__':
    main()
