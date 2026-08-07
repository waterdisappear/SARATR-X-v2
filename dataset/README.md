# Datasets / 数据集

This folder holds the datasets used by SARATR-X-v2. Since the actual data files are **too large to be committed**, please download and place them according to the layout below.

本目录存放 SARATR-X-v2 实验所需的数据集。由于数据文件较大，**不随仓库上传**，请按下方结构自行下载并放置。

## Directory Layout / 目录结构

```
dataset/
├── README.md                          # This file / 本说明
├── pre-training/                      # Pre-training data (17 SAR datasets, ~360K images) / 预训练数据
├── classification/                    # Classification evaluation data / 分类评测数据
│   ├── SOC_50classes/                 # ATRNet-STAR (SOC-50) — linear probing / 线性探测
│   ├── MSTAR/                         # MSTAR — few-shot linear probing / 少样本线性探测
│   ├── SARVASA/                       # SAR-VSA — k-NN
│   └── FUSAR-Ship/                    # FUSAR-Ship — k-NN (384×384)
├── detection/                         # Detection evaluation data (mmrotate/mmdet format) / 检测评测数据
│   ├── SARDet-100K/
│   ├── RSAR/
│   ├── SSDD/
│   └── HRSID/
└── segmentation/                      # Segmentation evaluation data (mmseg format) / 分割评测数据
    ├── AIR-PolSAR-Seg-2.0/
    ├── OpenEarthMap-SAR/
    ├── DDHR-SK/
    └── WHU-OPT-SAR/
```

## Classification Data Format / 分类数据格式

Each classification dataset follows the ImageFolder layout (`train/` and `test/` (or `Val/`) sub-folders, one folder per class), e.g.:

每个分类数据集采用 ImageFolder 组织（`train/` 与 `test/`（或 `Val/`）子目录，每个类一个子目录），例如：

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
评测脚本会从训练集自动推断类别数（见 `classification/linear_eval/SOC_50.py` 等）。

## Detection / Segmentation Data Format / 检测 / 分割数据格式

Detection and segmentation data follow the standard MMDetection/MMRotate/MMSegmentation formats (COCO / DOTA / custom JSON annotations or semantic-segmentation label maps). See the config descriptions under `detection/mmrotate/` and `segmentation/mmseg/` for details.

检测与分割数据使用 MMDetection/MMRotate/MMSegmentation 的标准格式（COCO / DOTA / 自定义 JSON 标注 或 语义分割标签图），具体结构见 `detection/mmrotate/` 与 `segmentation/mmseg/` 下的配置文件说明。

## How to Obtain the Datasets / 常用数据获取渠道

- **MSTAR**：classic public SAR vehicle dataset (10 classes, 12,092 images), available from official/mirror channels / 公开经典 SAR 车辆数据集
- **OpenSARShip / FUSAR-Ship**：public ship SAR datasets / 公开船舶 SAR 数据集
- **SARDet-100K / RSAR / SSDD / HRSID**：public SAR object-detection datasets / 公开 SAR 目标检测数据集
- **AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / DDHR-SK / WHU-OPT-SAR**：public SAR segmentation datasets / 公开 SAR 分割数据集
- **ATRNet-STAR / SAR-VSA / M4-SAR / FAIR-CSAR**：contact the authors (see the paper) / 请联系作者获取（见论文）

## Pre-training Data / 预训练数据

Pre-training uses 17 unlabeled SAR intensity-image datasets (~360K images) covering various sensors, resolutions, frequency bands and polarizations. The directory is a raw image collection that can be directly consumed by `SARImageDataset` in `pre-training/util/datasets.py` (with a `train_list` file list).

预训练使用 17 个未标注 SAR 强度图数据集（约 360K 张），包含多种传感器、分辨率、频段与极化配置。目录结构为原始图像集合，可直接用于 `pre-training/util/datasets.py` 中的 `SARImageDataset`（`train_list` 指定文件清单）。
