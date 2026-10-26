#!/usr/bin/env python3
"""Prepare the public cardiac release using our original NPZ/manifest format.

CT training uses released train+val records; MR uses train records only.
The four independent NIfTI test volumes per domain stay in their raw location.
This matches the existing convert_mmwhs + prepare_mmwhs_train_test workflow.
No new random split, resampling, normalization or label remapping is applied.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from asei._tfrecord import read_single_mmwhs_tfrecord
from asei.config import resolve_path
from asei.utils import sha_file, write_json


def natural_key(path):
    return [int(value) if value.isdigit() else value for value in re.split(r'(\d+)', path.name)]


def nifti_case_id(path, prefix):
    name = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
    return name[len(prefix):]


def prepare_data(raw_root, output_root, expected_train=9600, expected_test=4):
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root).resolve()
    if raw_root == output_root or raw_root in output_root.parents:
        raise ValueError('Use a separate output directory outside the downloaded raw-data directory')
    pool = raw_root / 'PnpAda_release_data/train&val'
    files, cases = {}, {}
    for domain, splits in [('ct', ['train', 'val']), ('mr', ['train'])]:
        records = []
        for split in splits:
            directory = pool / f'{domain}_{split}_tfs'
            current = sorted(directory.glob('*.tfrecords'), key=natural_key)
            if not current:
                raise FileNotFoundError(f'No TFRecords in {directory}')
            records.extend(current)
        if len(records) != expected_train:
            raise ValueError(f'{domain}: expected {expected_train} training records for this protocol, found {len(records)}')
        files[domain] = records
        directory = raw_root / f'test_{domain}_image&labels'
        images = sorted([p for p in directory.glob(f'image_{domain}_*.nii*')
                         if p.name.endswith(('.nii', '.nii.gz'))], key=natural_key)
        if len(images) != expected_test:
            raise ValueError(f'{domain}: expected {expected_test} test volumes, found {len(images)}')
        labels = {nifti_case_id(p, f'gth_{domain}_'):p for p in directory.glob(f'gth_{domain}_*.nii*')
                  if p.name.endswith(('.nii', '.nii.gz'))}
        cases[domain] = []
        for path in images:
            case = nifti_case_id(path, f'image_{domain}_')
            if case not in labels:
                raise FileNotFoundError(f'Missing test label for {path}')
            cases[domain].append((case, path, labels[case]))
    marker = output_root / 'metadata.json'
    if marker.exists():
        raise FileExistsError(f'{output_root} is already prepared; use it directly or choose a new output directory')
    if any(list((output_root / domain / 'train').glob('*.npz')) for domain in ['ct','mr']):
        raise FileExistsError('Output contains training samples; choose an empty output directory')
    summary = dict(protocol='CCRC/SIFA public cardiac split; CT train+val, MR train; independent NIfTI tests',
                   train_volume_sequences_verified=False, domains={})
    output_root.mkdir(parents=True, exist_ok=True)
    for domain, records in files.items():
        directory = output_root / domain / 'train'
        directory.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(records):
            sample = read_single_mmwhs_tfrecord(path)
            image, label = sample['image'], sample['label']
            if image.shape != (3,256,256) or label.shape != image.shape:
                raise ValueError(f'Expected [3,256,256] arrays: {path}')
            if not np.isfinite(image).all() or (label < 0).any() or (label > 4).any():
                raise ValueError(f'Expected finite images and labels 0..4: {path}')
            # Block IDs do not imply verified patient or volume identities.
            name = f'{domain}_case_{index//600:02d}_slice_{index%600:03d}.npz'
            np.savez_compressed(directory / name, image=image.astype(np.float32), label=label.astype(np.int64))
            if (index+1) % 500 == 0 or index+1 == len(records):
                print(f'[{domain}] {index+1}/{len(records)} converted samples', flush=True)
        summary['domains'][domain] = dict(train_samples=len(records), test_cases=len(cases[domain]),
            source_splits=['train','val'] if domain == 'ct' else ['train'])
    manifest = output_root / 'manifests/test_cases.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['domain','split','case_id','image_path','label_path','shape'])
        writer.writeheader()
        for domain, entries in cases.items():
            for case, image_path, label_path in entries:
                image, label = nib.load(str(image_path)), nib.load(str(label_path))
                values = np.asarray(label.get_fdata())
                if len(image.shape) != 3 or image.shape != label.shape or image.shape[:2] != (256,256):
                    raise ValueError(f'Test image/label dimensions mismatch: {image_path}')
                if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)) or (values < 0).any() or (values > 4).any():
                    raise ValueError(f'Expected preprocessed test labels 0..4: {label_path}')
                writer.writerow(dict(domain=domain,split='test',case_id=case,
                    image_path=os.path.relpath(image_path,output_root),label_path=os.path.relpath(label_path,output_root),
                    shape='x'.join(map(str,image.shape))))
    summary['test_manifest_sha256'] = sha_file(manifest)
    write_json(marker, summary)
    return output_root


def main():
    parser = argparse.ArgumentParser(description='Prepare the published MMWHS split without TensorFlow')
    parser.add_argument('--raw-root', required=True, help='Root containing PnpAda_release_data and both test_*_image&labels folders')
    parser.add_argument('--output-root', default='data/mmwhs')
    args = parser.parse_args()
    print(prepare_data(resolve_path(args.raw_root), resolve_path(args.output_root)))


if __name__ == '__main__':
    main()
