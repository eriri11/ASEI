#!/usr/bin/env python3
"""Configuration-driven standalone ASEI entry point."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.config import load_config, resolve_path

from asei.source import train_source


def main():
    parser = argparse.ArgumentParser(description='Train a source-domain segmentation model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', help='Override runtime.device, e.g. cuda:1')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config['runtime']['device'] = args.device
    print(train_source(config))


if __name__ == '__main__':
    main()
