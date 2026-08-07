# segmentation — SAR Semantic Segmentation (MMSegmentation Custom Extension) / SAR 语义分割（MMSegmentation 自定义扩展）

This directory contains the custom code for the **segmentation downstream task** of SARATR-X-v2, based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (0.11.0 + mmcv 1.3.x). Only the paper-related custom parts are kept:

本目录为 SARATR-X-v2 论文中**分割下游任务**的自定义代码，基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)（0.11.0 + mmcv 1.3.x）实现，保留论文相关的自定义部分：

- Custom backbones / 自定义骨干：`iTPN`（`backbone/iTPN.py`，same source as the pre-training iTPN / 与预训练 iTPN 同源）、`HiViT`（`backbone/hivit.py`）
- Custom datasets / 自定义数据集：`AIRPolSARSegDataset`（AIR-PolSAR-Seg 6 classes / 6 类）、`EarthMapOEM8OversampleDataset`
- Custom pipelines / 自定义 Pipeline：`LoadHHAs3ChSAR` / `LoadPreprocessedGrayAs3Ch` / `LoadPolSARAmplitudeRGB`（SAR amplitude/polarimetric data loading / SAR 幅度/极化数据加载）
- Custom losses / 自定义损失：`CEFocalLoss`、`CEAndLovaszLoss`
- Custom optimizer / 自定义优化器：`LayerDecayOptimizerConstructor`（with iTPN / HiViT layer-decay variants / 含 iTPN / HiViT 层衰减版）
- Custom evaluation / 自定义评测：`rs_metrics`（PA/OA/mIoU/mPA/Kappa）、`val_test_eval_hooks`
- Custom training API / 自定义训练入口：`mmcv_custom/train_api.py`（`tools/train.py` auto-uses the in-repo implementation / 自动使用仓库内实现）

## Directory Structure / 目录结构

```
segmentation/
├── README.md                     # This file / 本文件
├── backbone/                     # iTPN / HiViT / BEiT backbones (mmseg-registered) / 骨干（mmseg 注册版）
├── mmcv_custom/                  # Custom datasets / pipelines / losses / optimizers / hooks / train API
├── models/                       # Custom models (HiViT-related) / 自定义模型
├── configs/                      # Paper-related training/eval configs / 论文相关训练/评测配置
│   ├── itpn/                     # UperNet + iTPN (AIR-PolarSAR-Seg / EarthMap / WHU-OPT-SAR / ADE20K)
│   ├── hivit/                    # UperNet + HiViT (AIR-PolarSAR-Seg-2.0)
│   └── _base_/                   # Dataset / model / schedule / runtime configs / 基础配置
└── tools/                        # Data conversion and train/eval scripts / 数据转换与训练/评测脚本
```

## Environment Setup / 环境搭建

```bash
conda create -n itpn_seg python=3.7 -y
conda activate itpn_seg
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

> This repo was developed under mmseg 0.11.0 / mmcv 1.3.x. Adapt if using a newer mmseg
> (e.g. `mmseg.utils.get_root_logger` → mmengine-style API).
> 本仓库代码在 mmseg 0.11.0 / mmcv 1.3.x 下开发。若使用新版 mmseg 需适配（如 `mmseg.utils.get_root_logger` → mmengine 风格 API）。

## Data Preparation / 数据准备

| Dataset / 数据集 | Default path / 默认路径 | Description / 说明 |
| --- | --- | --- |
| AIR-PolarSAR-Seg-2.0 | `data/air-polarsar-seg-2.0/{train,val}` | Main segmentation dataset in the paper (amplitude RGB) / 论文分割主数据集 |
| AIR-PolarSAR-Seg | `data/air-polarsar-seg/{train_set,test_set}` | HH-channel version / HH 通道版 |
| EarthMap | `data/earthmap/{train,val,test}` | Land-cover (SAR + OEM8 labels) / 地物分类 |
| WHU-OPT-SAR | `data/whu_opt_sar_mmseg_256_split` | Multi-modal segmentation (256 tiles) / 多模态分割 |
| ADE20K | `data/ade/ADEChallengeData2016` | General segmentation reference (official model transfer) / 通用分割参考 |

Data conversion tools (`tools/`)：

```bash
# Convert AIR-PolSAR-Seg RGB labels to mmseg index labels
python tools/convert_air_polarsar_labels.py --train-dir <train> --test-dir <test>

# Split WHU-OPT-SAR large images into 256 tiles by scene
python tools/split_whu_opt_sar_for_mmseg.py --data-root <root>

# Other helpers: stats_earthmap_label_pixels.py, verify_hivit_pretrain.py,
# visualize_polsar_amplitude_params.py (polarimetric amplitude pseudo-RGB debugging)
```

## Pre-trained Weights / 预训练权重

The iTPN / HiViT pre-trained weights are **not committed**; download from `weights/` and place under mmseg's `ckpts/`:

iTPN / HiViT 预训练权重不随仓库提交，从 `weights/` 获取后放置到 mmseg 目录：

| Model / 模型 | Default path / 默认路径 | Description / 说明 |
| --- | --- | --- |
| iTPN-Base | `ckpts/itpn_base/checkpoint-1200.pth` | Main experiment / 论文主实验 |
| iTPN-Large | `ckpts/itpn_large/checkpoint-1200.pth` | Large-model experiment / 大模型实验 |
| HiViT-Base | `ckpts/hivit/checkpoint-1200.pth` | Comparison / 对比实验 |

Override the default path with `--cfg-options model.pretrained=/path/to/ckpt` in the training command.
可在训练命令中用 `--cfg-options model.pretrained=/path/to/ckpt` 覆盖默认路径。

## Training / Testing / 训练 / 测试

```bash
# Training (8-GPU example, AIR-PolarSAR-Seg-2.0 amplitude version)
bash tools/dist_train.sh \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py 8 \
    --work-dir work_dirs/air_polarsar2 --seed 0 --deterministic

# Testing (outputs mIoU / mPA / Kappa)
bash tools/dist_test.sh \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py \
    work_dirs/air_polarsar2/latest.pth 8 --eval mIoU

# Override weight/data paths
python tools/train.py \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py \
    --cfg-options model.pretrained=/path/to/checkpoint-1200.pth data_root=/path/to/data
```

## Mapping to Paper Results / 论文对应关系

- **AIR-PolarSAR-Seg-2.0 segmentation / 分割**：UperNet + iTPN-Base / iTPN-Large（`pixel_upernet_itpn_*_air_polarsar2_amp_linux.py`）
- **AIR-PolarSAR-Seg segmentation / 分割**：UperNet + iTPN（`pixel_upernet_itpn_*_air_polarsar_hh_linux*.py`）
- **WHU-OPT-SAR segmentation / 分割**：UperNet + iTPN（`pixel_upernet_itpn_*_whu_opt_sar_sar_linux.py`）
- **Comparison experiments / 对比实验**：UperNet + HiViT（`configs/hivit/`）、CLIP-UperNet + iTPN（`clip_upernet_itpn_*`）

See the paper's experimental section for the exact numbers. 具体数值以论文实验章节为准。
