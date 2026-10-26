"""Synthetic public-release conversion fixture; no real dataset is converted."""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.data import SliceDataset, VolumeDataset
from asei._tfrecord import TFRecordFormatError, read_single_mmwhs_tfrecord

spec = importlib.util.spec_from_file_location('prepare_data', Path(__file__).resolve().parents[1] / 'scripts/prepare_data.py')
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def varint(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def field(number, value):
    return varint((number << 3) | 2) + varint(len(value)) + value


def write_record(path, label_id):
    features = {}
    for kind in ['dsize', 'lsize']:
        for index, value in enumerate([256, 256, 3]):
            features[f'{kind}_dim{index}'] = field(3, field(1, varint(value)))
    image = np.linspace(0, 1, 256*256*3, dtype='<f4').reshape(256, 256, 3)
    label = np.full(image.shape, label_id, dtype='<f4')
    for name, array in [('data_vol',image),('label_vol',label)]:
        features[name] = field(1, field(1, array.tobytes()))
    entries = b''.join(field(1, field(1, key.encode())+field(2, value)) for key,value in features.items())
    example = field(1, entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q',len(example))+b'\0'*4+example+b'\0'*4)
    return image


class PreparationTests(unittest.TestCase):
    def test_conversion_preserves_pool_values_and_test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw'
            pool = raw / 'PnpAda_release_data/train&val'
            image = write_record(pool / 'ct_train_tfs/ct_train_slice0.tfrecords', 1)
            write_record(pool / 'ct_val_tfs/ct_val_slice0.tfrecords', 2)
            write_record(pool / 'mr_train_tfs/mr_train_slice0.tfrecords', 3)
            write_record(pool / 'mr_train_tfs/mr_train_slice1.tfrecords', 4)
            write_record(pool / 'mr_val_tfs/mr_val_slice0.tfrecords', 1)  # Must be excluded.
            for domain in ['ct','mr']:
                directory = raw / f'test_{domain}_image&labels'
                directory.mkdir()
                volume = np.zeros((256,256,2), np.float32)
                label = np.zeros_like(volume, np.int16)
                label[40:80,60:100] = 3
                nib.save(nib.Nifti1Image(volume,np.eye(4)),directory/f'image_{domain}_1003.nii.gz')
                nib.save(nib.Nifti1Image(label,np.eye(4)),directory/f'gth_{domain}_1003.nii.gz')
            before = {p:p.read_bytes() for p in raw.rglob('*') if p.is_file()}
            prepared = preparation.prepare_data(raw,root/'prepared',expected_train=2,expected_test=1)
            for domain, expected in [('ct',[1,2]),('mr',[3,4])]:
                files = sorted((prepared/domain/'train').glob('*.npz'))
                self.assertEqual(len(files),2)
                for path, label_id in zip(files,expected):
                    with np.load(path) as sample:
                        self.assertTrue(np.array_equal(sample['image'],np.moveaxis(image,-1,0)))
                        self.assertTrue((sample['label'] == label_id).all())
                self.assertEqual(len(VolumeDataset(prepared,domain)),2)
                self.assertNotIn('label',SliceDataset(prepared,domain)[0])
            for path,value in before.items():
                self.assertEqual(path.read_bytes(),value)
            with self.assertRaises(FileExistsError):
                preparation.prepare_data(raw,prepared,expected_train=2,expected_test=1)
            with self.assertRaises(ValueError):
                preparation.prepare_data(raw,root/'bad-counts')
            self.assertFalse((root/'bad-counts').exists())

    def test_truncated_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'broken.tfrecords'
            path.write_bytes(struct.pack('<Q',100)+b'\0'*4+b'truncated')
            with self.assertRaises(TFRecordFormatError):
                read_single_mmwhs_tfrecord(path)


if __name__ == '__main__':
    unittest.main()
