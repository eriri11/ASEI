#!/usr/bin/env python3
"""Configuration-driven standalone ASEI entry point."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.config import load_config, resolve_path

from asei.data import build_dataset, build_loader
from asei.models import build_model
from asei.metrics import evaluate
from asei.utils import load_checkpoint, make_device, write_json


def main():
    parser = argparse.ArgumentParser(description='Evaluate the final EMA teacher: case-wise 3D Dice and ASSD')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Default: output.root/teacher_final.pt')
    parser.add_argument('--device')
    parser.add_argument('--domain', choices=['ct', 'mr'], help='Default: target_domain')
    parser.add_argument('--output', help='JSON destination, relative to ASEI root')
    parser.add_argument('--no-lcc', action='store_true', help='Disable per-organ 3D largest connected component')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config['runtime']['device'] = args.device
    device = make_device(config['runtime']['device'])
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else str(Path(config['output']['root']) / 'teacher_final.pt')
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, device)
    dataset = build_dataset(config, args.domain or config['experiment']['target_domain'], 'test', True)
    if not dataset.volume_sequence_verified:
        raise ValueError('3D evaluation requires manifests/test_cases.csv with verified volume identities')
    metrics = evaluate(model, build_loader(config, dataset), 5, device,
                       compute_assd=True, largest_component=not args.no_lcc)
    metrics['checkpoint'] = checkpoint
    destination = resolve_path(args.output) if args.output else str(Path(config['output']['root']) / 'evaluation.json')
    write_json(destination, metrics)
    for organ in ['AA', 'LA', 'LV', 'MYO']:
        distance = metrics[f'assd_{organ}']
        assd = f'{distance:.4f}' if distance is not None else 'undefined'
        print(f'{organ}: Dice={100*metrics[f"dice_{organ}"]:.2f}%, ASSD={assd}')
    print(f'Mean Dice={100*metrics["mean_dice"]:.2f}%, Mean ASSD={metrics["mean_assd"]}')
    print(destination)


if __name__ == '__main__':
    main()
