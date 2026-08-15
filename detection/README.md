<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# detection — SAR Rotated Object Detection (MMRotate Custom Extension)

This directory contains the custom code for the **detection downstream task** of SARATR-X-v2, based on [MMRotate](https://github.com/open-mmlab/mmrotate). Only the paper-related custom parts are kept:

- Custom backbones: `iTPN_pixel` / `iTPN_CLIP` / `fast_iTPN` / `ReResNet`
- Custom datasets: `RSARDataset` / `FAIRCSARDataset` / `SARDataset`
- Custom pipeline: `LoadImageFromFileMultiExt` (mixed image extensions)
- Custom hooks: `SWAHook` / `SWARestoreHook` / `MMRotateNumClassCheckHook`
- Custom optimizer: `LayerDecayOptimizerConstructor` (HiViT layer-decay variant)

> The framework itself (mmrotate / mmcv / mmdet) must be installed by the user; this directory provides **custom files + configs + instructions**. Copy them into the mmrotate installation to run.

## Directory Structure

```
detection/
├── README.md                     # This file
├── configs/                      # Paper-related training/eval configs
│   ├── redet/                    # ReDet + iTPN (RSAR / FAIR-CSAR / SIVED)
│   ├── roi_trans/                # RoI Transformer + iTPN (FAIR-CSAR FSI/SL)
│   ├── oriented_rcnn/            # Oriented R-CNN + iTPN (RSAR / FAIR-CSAR SL)
│   ├── oriented_reppoints/       # Oriented RepPoints + iTPN (SIVED)
│   ├── rotated_atss/             # Rotated ATSS + iTPN (SIVED)
│   ├── rotated_fcos/             # Rotated FCOS + iTPN (SIVED)
│   ├── rotated_faster_rcnn/      # Rotated Faster R-CNN + iTPN (DOTA / SIVED)
│   ├── rotated_retinanet/        # Rotated RetinaNet + iTPN (FAIR-CSAR FSI)
│   └── _base_/                   # Dataset / schedule / runtime configs
└── mmrotate/                     # Custom modules (copy into mmrotate install)
    ├── models/backbones/         # iTPN_pixel, iTPN_CLIP, fast_itpn, ReResNet + mmcv_custom
    ├── datasets/                 # sar.py (RSAR/SAR), faircsar.py (FAIR-CSAR)
    ├── core/                     # SWA hooks, class-count check hook
    └── apis/train.py             # Training entry (uses custom hooks)
```

## Environment Setup

See the environment config of the `Rotated_fast_itpn` sub-project (DOTA version):

```bash
# 1. Create conda env (Python 3.8)
conda create -n rotate_itpn python=3.8 -y
conda activate rotate_itpn

# 2. Install PyTorch (CUDA 11.3 example)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Install mmcv (build from source recommended to avoid some bugs)
pip install mmcv==2.0.0rc4
# or from source: wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.0.0rc4.zip

# 4. Install mmdet
pip install mmdet==3.0.0rc6

# 5. Install mmrotate
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
```

> Note: This repo's configs use the **MMRotate 0.3.x-style API** (`EpochBasedRunner` / `evaluation` / `custom_hooks`).
> For mmrotate 1.x (MMEngine Runner), replace `runner`/`evaluation`/`optimizer_config` with `train_cfg`/`val_cfg`/`optim_wrapper`. Follow your installed version.

## Integration Steps

Copy the custom files from this directory into the mmrotate installation:

```bash
# Run in the mmrotate repo root
cp -r mmrotate/ ./          # overwrite custom modules
cp -r configs/ ./           # overwrite/add paper-related configs
```

## Datasets

| Dataset | Default path | Env var | Description |
| --- | --- | --- | --- |
| RSAR | `data/rsar/` | — | DOTA format, 6 classes (ship/aircraft/car/tank/bridge/harbor) |
| FAIR-CSAR SL | `data/faircsar/SL-TRAINVAL/...` | `FAIRCSAR_SL_TRAIN` / `FAIRCSAR_SL_TEST` | Single-look |
| FAIR-CSAR FSI | `data/faircsar/FSI-TRAINVAL/...` | `FAIRCSAR_FSI_TRAIN` / `FAIRCSAR_FSI_TEST` | Multi-look |
| DOTA | `data/split_dota/...` | — | Standard rotated detection benchmark |
| SIVED | `data/sived/` | `SIVED_ROOT` | DOTA-style TXT annotations |

Data directories live under the mmrotate installation, or can be specified as absolute paths via env vars.

## Pre-trained Weights

The iTPN pre-trained weights are **not committed**; download from `weights/` and place under mmrotate's `ckpts/`:

| Model | Default path | Env var |
| --- | --- | --- |
| iTPN-Base (checkpoint-1200) | `ckpts/iTPN/checkpoint-1200.pth` | `ITPN_CKPT` |
| iTPN-Large (checkpoint-1200) | `ckpts/iTPN_large/checkpoint-1200.pth` | `ITPN_LARGE_CKPT` |
| ResNet-50 (R50 baseline) | `ckpts/resnet50-0676ba61.pth` | `RESNET50_CKPT` |
| fast_iTPN (DOTA configs) | `ckpts/iTPN/fast_itpn_*.pt` | — |

## Training / Testing

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

## Mapping to Paper Results

- **RSAR detection**: ReDet / Oriented R-CNN + iTPN-Base (`redet_itpn_base_3x_rsar.py`, etc.)
- **FAIR-CSAR detection**: ReDet + iTPN-Base / iTPN-Large (SL and FSI settings)
- **DOTA detection**: Rotated Faster R-CNN + fast_iTPN (DOTA configs under `rotated_faster_rcnn`)
- **SIVED detection**: Multiple detection heads + iTPN (`rotated_atss` / `rotated_fcos`, etc.)
- **SARDet-100K / SSDD / HRSID detection (HBB)**: GFL + iTPN — reference configs & training logs under `results/detection/`

See the paper's experimental section for the exact numbers; per-dataset logs/configs are under `results/detection/`.
