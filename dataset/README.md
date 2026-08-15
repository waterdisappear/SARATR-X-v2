<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# Datasets

This folder holds the datasets used by SARATR-X-v2. Since the actual data files are **too large to be committed**, please download and place them according to the layout below.

## Directory Layout

```
dataset/
├── README.md                          # This file
├── pre-training/                      # Pre-training data (17 SAR datasets, ~360K images)
├── classification/                    # Classification evaluation data
│   ├── SOC_50classes/                 # ATRNet-STAR (SOC-50) — linear probing
│   ├── MSTAR/                         # MSTAR — few-shot linear probing
│   ├── SARVASA/                       # SAR-VSA — k-NN
│   └── FUSAR-Ship/                    # FUSAR-Ship — k-NN (384×384)
├── detection/                         # Detection evaluation data (mmrotate/mmdet format)
│   ├── SARDet-100K/
│   ├── RSAR/
│   ├── SSDD/
│   └── HRSID/
└── segmentation/                      # Segmentation evaluation data (mmseg format)
    ├── AIR-PolSAR-Seg-2.0/
    ├── OpenEarthMap-SAR/
    ├── DDHR-SK/
    └── WHU-OPT-SAR/
```

## Classification Data Format

Each classification dataset follows the ImageFolder layout (`train/` and `test/` (or `Val/`) sub-folders, one folder per class), e.g.:

```
SOC_50classes/
├── train/
│   ├── class_0/xxx.png
│   ├── class_1/xxx.png
│   └── ...
└── test/
    ├── class_0/xxx.png
    └── ...
```

The eval scripts infer the number of classes automatically from the training set (see `classification/linear_eval/SOC_50.py` etc.).

## Detection / Segmentation Data Format

Detection and segmentation data follow the standard MMDetection/MMRotate/MMSegmentation formats (COCO / DOTA / custom JSON annotations or semantic-segmentation label maps). See the config descriptions under `detection/mmrotate/` and `segmentation/mmseg/` for details.

## How to Obtain the Datasets

- **MSTAR**: classic public SAR vehicle dataset (10 classes, 12,092 images), available from official/mirror channels
- **OpenSARShip / FUSAR-Ship**: public ship SAR datasets
- **SARDet-100K / RSAR / SSDD / HRSID**: public SAR object-detection datasets
- **AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / DDHR-SK / WHU-OPT-SAR**: public SAR segmentation datasets
- **ATRNet-STAR / SAR-VSA / M4-SAR / FAIR-CSAR**: contact the authors (see the paper)

## Pre-training Data

Pre-training uses 17 unlabeled SAR intensity-image datasets (~360K images) covering various sensors, resolutions, frequency bands and polarizations. The directory is a raw image collection that can be directly consumed by `SARImageDataset` in `pre-training/util/datasets.py` (with a `train_list` file list).
