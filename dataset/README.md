# 数据集 (Datasets)

本目录存放 SARATR-X-v2 实验所需的数据集。由于数据文件较大，**不随仓库上传**，
请按下方结构自行下载并放置（可在仓库根目录执行 `git lfs` 或自行组织）。

This folder holds the datasets used by SARATR-X-v2. The actual data files are
**too large to be committed**, so please download and place them according to
the layout below.

## 目录结构 (Directory layout)

```
dataset/
├── README.md                          # 本说明
├── pre-training/                      # 预训练数据（17 个 SAR 数据集，约 360K 图）
├── classification/                    # 分类评测数据
│   ├── SOC_50classes/                 # ATRNet-STAR (SOC-50) —— 线性探测
│   ├── MSTAR/                         # MSTAR —— 少样本线性探测
│   ├── SARVASA/                       # SAR-VSA —— k-NN
│   └── FUSAR-Ship/                    # FUSAR-Ship —— k-NN（384x384）
├── detection/                         # 检测评测数据（mmrotate/mmdet 格式）
│   ├── SARDet-100K/
│   ├── RSAR/
│   ├── SSDD/
│   └── HRSID/
└── segmentation/                      # 分割评测数据（mmseg 格式）
    ├── AIR-PolSAR-Seg-2.0/
    ├── OpenEarthMap-SAR/
    ├── DDHR-SK/
    └── WHU-OPT-SAR/
```

## 分类数据格式 (Classification data format)

每个分类数据集采用 ImageFolder 组织（`train/` 与 `test/`（或 `Val/`）子目录，
每个类一个子目录），例如：

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

评测脚本会从训练集自动推断类别数（见 `classification/linear_eval/SOC_50.py` 等）。

## 检测 / 分割数据格式 (Detection / segmentation data format)

检测与分割数据使用 MMDetection/MMRotate/MMSegmentation 的标准格式
（COCO / DOTA / 自定义 JSON 标注 或 语义分割标签图），
具体结构见 `detection/mmrotate/` 与 `segmentation/mmseg/` 下的配置文件说明。

## 常用数据获取渠道 (How to obtain the datasets)

- **MSTAR**：公开经典 SAR 车辆数据集（10 类，12,092 图），可从官方/镜像渠道获取
- **OpenSARShip / FUSAR-Ship**：公开船舶 SAR 数据集
- **SARDet-100K / RSAR / SSDD / HRSID**：公开 SAR 目标检测数据集
- **AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / DDHR-SK / WHU-OPT-SAR**：公开 SAR 分割数据集
- **ATRNet-STAR / SAR-VSA / M4-SAR / FAIR-CSAR**：请联系作者获取（见论文）

## 预训练数据 (Pre-training data)

预训练使用 17 个未标注 SAR 强度图数据集（约 360K 张），包含多种传感器、分辨率、
频段与极化配置。目录结构为原始图像集合，可直接用于 `pre-training/util/datasets.py`
中的 `SARImageDataset`（`train_list` 指定文件清单）。
