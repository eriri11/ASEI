#!/usr/bin/env python3
"""Configuration-driven standalone ASEI entry point."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.config import load_config, resolve_path

from asei.adaptation import adapt_target


def main():
    parser = argparse.ArgumentParser(description='Run ASEI: GLFP + ECSS with Mean Teacher adaptation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', help='Override runtime.device, e.g. cuda:1')
    parser.add_argument('--source-checkpoint', help='Override initializer path')
    parser.add_argument('--skip-warmup', action='store_true', help='Initializer is already a pre-adapted MT teacher')
    parser.add_argument('--prepare-only', action='store_true', help='Generate GLFP labels and ECSS sets without Stage2 training')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config['runtime']['device'] = args.device
    if args.source_checkpoint:
        config['source_checkpoint'] = resolve_path(args.source_checkpoint)
    if args.skip_warmup:
        config['adaptation']['warmup']['enabled'] = False
    if args.prepare_only:
        config['adaptation']['prepare_only'] = True
    checkpoint = adapt_target(config)
    if checkpoint:
        print(checkpoint)


if __name__ == '__main__':
    main()
