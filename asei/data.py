"""Single-channel MMWHS slices; target adaptation never reads label arrays."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize(image, axes):
    image = np.asarray(image, dtype=np.float32).copy()
    low, high = image.min(axis=axes, keepdims=True), image.max(axis=axes, keepdims=True)
    np.subtract(image, low, out=image)
    scale = high - low
    np.divide(image, np.where(scale > 0, scale, 1), out=image)
    return image


class SliceDataset(Dataset):
    def __init__(self, root, domain, split='train', with_labels=False,
                 center_slice=1, transform=None):
        self.root, self.domain, self.split = Path(root), domain, split
        self.with_labels, self.center_slice, self.transform = with_labels, center_slice, transform
        self.files = sorted((self.root / domain / split).glob('*.npz'))
        self.volume_sequence_verified = False
        if not self.files:
            raise FileNotFoundError(f'No NPZ slices in {self.root / domain / split}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as archive:
            image = normalize(archive['image'], (-2, -1))
            label = np.asarray(archive['label'], dtype=np.int64) if self.with_labels else None
        if image.ndim == 2:
            image = image[None]
        center = 0 if image.shape[0] == 1 else self.center_slice
        if image.ndim != 3 or not 0 <= center < image.shape[0]:
            raise ValueError(f'Expected [C,H,W] with a valid center slice: {path}')
        case_id, slice_id = path.stem, str(index)
        if '_slice_' in path.stem:
            prefix, suffix = path.stem.rsplit('_slice_', 1)
            if suffix.isdigit():
                case_id, slice_id = prefix, str(int(suffix))
        sample = dict(image=torch.from_numpy(image[center:center+1].copy()),
                      case_id=case_id, slice_id=slice_id, domain=self.domain, index=index)
        if label is not None:
            target = label if label.ndim == 2 else label[center]
            if target.shape != image.shape[-2:] or np.any((target < 0) | (target > 4)):
                raise ValueError(f'Expected labels 0..4 aligned with image: {path}')
            sample['label'] = torch.from_numpy(target.copy())
        return self.transform(sample) if self.transform else sample


class VolumeDataset(Dataset):
    """Lazy NIfTI test reader, reconstructable by case_id and slice_id."""
    volume_sequence_verified = True

    def __init__(self, root, domain, align_orientation=True, volume_root=None):
        self.root, self.domain = Path(root), domain
        self.align_orientation = align_orientation
        self.volume_root = Path(volume_root) if volume_root else self.root
        with (self.root / 'manifests/test_cases.csv').open() as stream:
            self.cases = [row for row in csv.DictReader(stream) if row['domain'] == domain and row.get('split', 'test') == 'test']
        if not self.cases:
            raise ValueError(f'No {domain} test cases in manifest')
        self.indices = [(i, z) for i, row in enumerate(self.cases) for z in range(int(row['shape'].split('x')[-1]))]
        self.cached_case = None

    def __len__(self):
        return len(self.indices)

    def resolve(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        for base in dict.fromkeys([self.volume_root, self.root, *self.root.parents]):
            candidate = base / path
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f'Cannot resolve manifest path: {value}; set data.volume_root')

    def __getitem__(self, index):
        import nibabel as nib
        case, z = self.indices[index]
        row = self.cases[case]
        if self.cached_case != case:
            image = nib.load(str(self.resolve(row['image_path']))).get_fdata(dtype=np.float32)
            label = np.asarray(nib.load(str(self.resolve(row['label_path']))).get_fdata(), dtype=np.int64)
            if image.shape != label.shape or image.ndim != 3 or image.shape[2] != int(row['shape'].split('x')[-1]):
                raise ValueError(f'Volume/manifest shape mismatch for case {row["case_id"]}')
            image = normalize(image, (0, 1))
            if self.align_orientation:
                image, label = np.flip(image, axis=(0, 1)).copy(), np.flip(label, axis=(0, 1)).copy()
            self.image, self.label, self.cached_case = image, label, case
        target = self.label[:, :, z].copy()
        if np.any((target < 0) | (target > 4)):
            raise ValueError('Test labels must use converted class IDs 0..4')
        return dict(image=torch.from_numpy(self.image[:, :, z][None].copy()),
                    label=torch.from_numpy(target), case_id=row['case_id'], slice_id=str(z), domain=self.domain, index=index)


def build_dataset(config, domain, split='train', with_labels=False, transform=None):
    settings = config['data']
    if split == 'test' and (Path(settings['root']) / 'manifests/test_cases.csv').exists():
        if not with_labels or transform is not None:
            raise ValueError('Volume test reader is evaluation-only')
        return VolumeDataset(settings['root'], domain, settings.get('align_test_orientation', True), settings.get('volume_root'))
    return SliceDataset(settings['root'], domain, split, with_labels, settings.get('center_slice', 1), transform)


def build_loader(config, dataset, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=config['data'].get('batch_size', 16),
                      shuffle=shuffle, num_workers=config['data'].get('num_workers', 4),
                      pin_memory=config['runtime']['device'].startswith('cuda'), generator=generator)
