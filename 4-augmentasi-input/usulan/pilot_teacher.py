"""Validation-only RankUp with a small set of promoted tree pseudo-labels."""
import argparse
import importlib
import json
import tempfile
from pathlib import Path
import pilot
from conditional_aug import ConditionalAug
from pseudo_teacher import promote


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=pilot.FAMILIES, required=True)
    p.add_argument('--percent', type=int, choices=(1, 10, 30), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    module = importlib.import_module(f'experiments.{pilot.FAMILIES[args.dataset]}.gaussian')
    original_fit = module.fit_model
    teacher_info = {}
    def fit_with_teacher(*positional, **kwargs):
        vals = list(positional)
        vals[1], vals[2], info = promote(vals[1], vals[2], vals[3], args.dataset == 'pendigits')
        teacher_info.update(info)
        return original_fit(*vals, **kwargs)
    module.fit_model = fit_with_teacher
    pilot.LocalTransport = lambda **kwargs: ConditionalAug(grouped=True)
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        result = pilot.run(args.dataset, args.percent, 'local_transport', Path('/data'), temp,
                           epochs=120, max_steps=100)
        src = temp / f'label_{args.percent:02d}' / args.dataset
        out = args.output_root / f'label_{args.percent:02d}' / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / 'tree_rankup_history.csv').write_bytes((src / 'local_transport_history.csv').read_bytes())
        result['method'] = 'tree_rankup'
        result['pilot_config'].pop('radius', None)
        result['pilot_config'].update({'effective_weak_strength': 0., 'strong_strength': .3,
                                       'donor_pool': 2048, 'donor_k': 5, 'teacher': teacher_info})
        (out / 'tree_rankup_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print('COMPLETED', args.dataset, args.percent, result['validation'], teacher_info, flush=True)


if __name__ == '__main__':
    main()
