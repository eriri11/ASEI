# Data preparation

Use the preprocessed cardiac release linked by [SIFA](https://github.com/cchen-cc/SIFA#data-preparation), following the public split used by [SFDA-CCRC](https://github.com/aoge1993/SFDA-CCRC). Arrange the downloaded files as:

```text
data/raw/MMWHS/
├── PnpAda_release_data/train&val/
│   ├── ct_train_tfs/
│   ├── ct_val_tfs/
│   └── mr_train_tfs/
├── test_ct_image&labels/
└── test_mr_image&labels/
```

Convert them without TensorFlow:

```bash
python scripts/prepare_data.py --raw-root data/raw/MMWHS --output-root data/mmwhs
```

The converter combines the released CT train/validation TFRecords and uses the MR training TFRecords. Each modality provides 9,600 training samples and four independent NIfTI test volumes. It writes `ct/train/*.npz`, `mr/train/*.npz`, and `manifests/test_cases.csv`. Test volumes remain in the raw-data directory, referenced by relative paths in the manifest.

NPZ files contain `image` and `label` arrays of shape `[3, 256, 256]`; the model uses the center slice and per-slice min–max normalization. The supplied release must already use these label IDs:

| Background | MYO | LA | LV | AA |
|---|---|---|---|---|
| 0 | 1 | 2 | 3 | 4 |

Target adaptation reads only the image arrays. Labels are used for source training and final evaluation. For an existing prepared dataset, set `data.root` and skip conversion. If raw volumes move, update the manifest paths or set `data.volume_root`.
