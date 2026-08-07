# detection — SAR Rotated Object Detection (MMRotate Custom Extension) / SAR 旋转目标检测（MMRotate 自定义扩展）

This directory contains the custom code for the **detection downstream task** of SARATR-X-v2, based on [MMRotate](https://github.com/open-mmlab/mmrotate). Only the paper-related custom parts are kept:

本目录为 SARATR-X-v2 论文中**检测下游任务**的自定义代码，基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架实现，只保留论文相关的自定义部分：

- Custom backbones / 自定义骨干：`iTPN_pixel` / `iTPN_CLIP` / `fast_iTPN` / `ReResNet`
- Custom datasets / 自定义数据集：`RSARDataset` / `FAIRCSARDataset` / `SARDataset`
- Custom pipeline / 自定义 Pipeline：`LoadImageFromFileMultiExt`（mixed image extensions / 混合图片后缀）
- Custom hooks / 自定义 Hook：`SWAHook` / `SWARestoreHook` / `MMRotateNumClassCheckHook`
- Custom optimizer / 自定义优化器：`LayerDecayOptimizerConstructor`（含 HiViT layer-decay variant / HiViT 层衰减版）

> The framework itself (mmrotate / mmcv / mmdet) must be installed by the user; this directory provides **custom files + configs + instructions**. Copy them into the mmrotate installation to run.
> 框架本体（mmrotate / mmcv / mmdet）需要用户自行安装；本目录提供 **自定义文件 + 配置 + 说明**，复制到 mmrotate 安装目录即可运行。

## Directory Structure / 目录结构

```
detection/
├── README.md                     # This file / 本文件
├── configs/                      # Paper-related training/eval configs / 论文相关的训练/评测配置
│   ├── redet/                    # ReDet + iTPN (RSAR / FAIR-CSAR / SIVED)
│   ├── roi_trans/                # RoI Transformer + iTPN (FAIR-CSAR FSI/SL)
│   ├── oriented_rcnn/            # Oriented R-CNN + iTPN (RSAR / FAIR-CSAR SL)
│   ├── oriented_reppoints/       # Oriented RepPoints + iTPN (SIVED)
│   ├── rotated_atss/             # Rotated ATSS + iTPN (SIVED)
│   ├── rotated_fcos/             # Rotated FCOS + iTPN (SIVED)
│   ├── rotated_faster_rcnn/      # Rotated Faster R-CNN + iTPN (DOTA / SIVED)
│   ├── rotated_retinanet/        # Rotated RetinaNet + iTPN (FAIR-CSAR FSI)
│   └── _base_/                   # Dataset / schedule / runtime configs / 数据集 / 调度 / 运行时
└── mmrotate/                     # Custom modules (copy into mmrotate install) / 自定义模块
    ├── models/backbones/         # iTPN_pixel、iTPN_CLIP、fast_itpn、ReResNet + mmcv_custom
    ├── datasets/                 # sar.py (RSAR/SAR)、faircsar.py (FAIR-CSAR)
    ├── core/                     # SWA hooks, class-count check hook / SWA 相关 Hook、类别数检查 Hook
    └── apis/train.py             # Training entry (uses custom hooks) / 训练入口
```

## Environment Setup / 环境搭建

See the environment config of the `Rotated_fast_itpn` sub-project (DOTA version):

参考 `Rotated_fast_itpn` 子项目（DOTA 版本）的环境配置：

```bash
# 1. Create conda env (Python 3.8) / 创建 conda 环境
conda create -n rotate_itpn python=3.8 -y
conda activate rotate_itpn

# 2. Install PyTorch (CUDA 11.3 example)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Install mmcv (build from source recommended to avoid some bugs)
pip install mmcv==2.0.0rc4
# or from source / 或源码安装：wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.0.0rc4.zip

# 4. Install mmdet
pip install mmdet==3.0.0rc6

# 5. Install mmrotate
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
```

> Note: This repo's configs use the **MMRotate 0.3.x-style API** (`EpochBasedRunner` / `evaluation` / `custom_hooks`).
> For mmrotate 1.x (MMEngine Runner), replace `runner`/`evaluation`/`optimizer_config` with `train_cfg`/`val_cfg`/`optim_wrapper`. Follow your installed version.
> 注：本仓库配置使用 `EpochBasedRunner` / `evaluation` / `custom_hooks` 的 **MMRotate 0.3.x 风格 API**，若使用 mmrotate 1.x（MMEngine Runner），需替换为 `train_cfg`/`val_cfg`/`optim_wrapper` 等新写法。请以实际安装版本为准。

## Integration Steps / 集成步骤

Copy the custom files from this directory into the mmrotate installation:

把本目录的自定义文件复制到 mmrotate 安装目录：

```bash
# Run in the mmrotate repo root / 在 mmrotate 仓库根目录执行
cp -r mmrotate/ ./          # overwrite custom modules / 覆盖 mmrotate 包内的自定义模块
cp -r configs/ ./           # overwrite/add paper-related configs / 覆盖/新增论文相关配置
```

## Datasets / 数据集

| Dataset / 数据集 | Default path / 默认路径 | Env var / 环境变量 | Description / 说明 |
| --- | --- | --- | --- |
| RSAR | `data/rsar/` | — | DOTA format, 6 classes (ship/aircraft/car/tank/bridge/harbor) |
| FAIR-CSAR SL | `data/faircsar/SL-TRAINVAL/...` | `FAIRCSAR_SL_TRAIN` / `FAIRCSAR_SL_TEST` | Single-look / 单视 |
| FAIR-CSAR FSI | `data/faircsar/FSI-TRAINVAL/...` | `FAIRCSAR_FSI_TRAIN` / `FAIRCSAR_FSI_TEST` | Multi-look / 多视 |
| DOTA | `data/split_dota/...` | — | Standard rotated detection benchmark / 旋转检测标准基准 |
| SIVED | `data/sived/` | `SIVED_ROOT` | DOTA-style TXT annotations |

Data directories live under the mmrotate installation, or can be specified as absolute paths via env vars.
数据目录统一在 mmrotate 安装目录下，或通过环境变量指定绝对路径。

## Pre-trained Weights / 预训练权重

The iTPN pre-trained weights are **not committed**; download from `weights/` and place under mmrotate's `ckpts/`:

iTPN 预训练权重不随仓库提交，请从 `weights/` 获取并放置到 mmrotate 的 `ckpts/` 目录：

| Model / 模型 | Default path / 默认路径 | Env var / 环境变量 |
| --- | --- | --- |
| iTPN-Base (checkpoint-1200) | `ckpts/iTPN/checkpoint-1200.pth` | `ITPN_CKPT` |
| iTPN-Large (checkpoint-1200) | `ckpts/iTPN_large/checkpoint-1200.pth` | `ITPN_LARGE_CKPT` |
| ResNet-50 (R50 baseline) | `ckpts/resnet50-0676ba61.pth` | `RESNET50_CKPT` |
| fast_iTPN (DOTA configs) | `ckpts/iTPN/fast_itpn_*.pt` | — |

## Training / Testing / 训练 / 测试

```bash
# Training (4-GPU example)
bash tools/dist_train.sh configs/redet/redet_itpn_base_3x_rsar.py 4
bash tools/dist_train.sh configs/redet/redet_itpn_base_3x_faircsar_sl.py 4

# Testing (outputs mAP50)
bash tools/dist_test.sh configs/redet/redet_itpn_base_3x_rsar.py \
    work_dirs/redet_itpn_base_3x_rsar/latest.pth 4 --eval mAP

# Set data/weight paths (example)
export FAIRCSAR_SL_TRAIN=/path/to/FAIR-CSAR/SL-TRAINVAL/trainval
export ITPN_CKPT=/path/to/checkpoint-1200.pth
```

## Mapping to Paper Results / 论文对应关系

- **RSAR detection / RSAR 检测**：ReDet / Oriented R-CNN + iTPN-Base（`redet_itpn_base_3x_rsar.py` 等）
- **FAIR-CSAR detection / FAIR-CSAR 检测**：ReDet + iTPN-Base / iTPN-Large（SL 与 FSI 两种设置）
- **DOTA detection / DOTA 检测**：Rotated Faster R-CNN + fast_iTPN（`rotated_faster_rcnn` 下 DOTA 配置）
- **SIVED detection / SIVED 检测**：多种检测头 + iTPN（`rotated_atss` / `rotated_fcos` 等）
- **SARDet-100K / SSDD / HRSID detection（HBB）**：GFL + iTPN — reference configs & training logs under `results/detection/` / 参考配置与训练日志见 `results/detection/`

See the paper's experimental section for the exact numbers; per-dataset logs/configs are under `results/detection/`. 具体数值以论文实验章节为准；各数据集日志与配置见 `results/detection/`。
