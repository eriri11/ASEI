"""Read YAML with paths anchored at the standalone project directory."""
from copy import deepcopy
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge(defaults, values):
    out = deepcopy(defaults)
    for key, value in values.items():
        if key not in defaults:
            raise ValueError(f'Unknown option: {key}')
        out[key] = merge(defaults[key], value) if isinstance(defaults[key], dict) else deepcopy(value)
    return out


def resolve_path(value):
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    for required in ['experiment', 'runtime', 'data', 'model', 'output']:
        if required not in config:
            raise ValueError(f'Missing configuration section: {required}')
    for domain in ['source_domain', 'target_domain']:
        if config['experiment'].get(domain) not in ['ct', 'mr']:
            raise ValueError(f'experiment.{domain} must be ct or mr')
    config['data']['root'] = resolve_path(config['data']['root'])
    if config['data'].get('volume_root'):
        config['data']['volume_root'] = resolve_path(config['data']['volume_root'])
    config['output']['root'] = resolve_path(config['output']['root'])
    if config.get('source_checkpoint'):
        config['source_checkpoint'] = resolve_path(config['source_checkpoint'])
    if 'adaptation' in config:
        for key in ['medsam_checkpoint', 'embedding_cache_root']:
            config['adaptation'][key] = resolve_path(config['adaptation'][key])
    return config


def save_config(config, directory):
    path = Path(directory) / 'config.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
