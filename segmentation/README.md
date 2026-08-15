<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# segmentation — SAR Semantic Segmentation (MMSegmentation Custom Extension)

This directory contains the custom code for the **segmentation downstream task** of SARATR-X-v2, based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (0.11.0 + mmcv 1.3.x). Only the paper-related custom parts are kept:

- Custom backbones: `iTPN` (`backbone/iTPN.py`, same source as the pre-training iTPN), `HiViT` (`backbone/hivit.py`)
- Custom datasets: `AIRPolSARSegDataset` (AIR-PolSAR-Seg 6 classes), `EarthMapOEM8OversampleDataset`
- Custom pipelines: `LoadHHAs3ChSAR` / `LoadPreprocessedGrayAs3Ch` / `LoadPolSARAmplitudeRGB` (SAR amplitude/polarimetric data loading)
- Custom losses: `CEFocalLoss`, `CEAndLovaszLoss`
- Custom optimizer: `LayerDecayOptimizerConstructor` (with iTPN / HiViT layer-decay variants)
- Custom evaluation: `rs_metrics` (PA/OA/mIoU/mPA/Kappa), `val_test_eval_hooks`
- Custom training API: `mmcv_custom/train_api.py` (`tools/train.py` auto-uses the in-repo implementation)

## Directory Structure

```
segmentation/
├── README.md                     # This file
├── backbone/                     # iTPN / HiViT / BEiT backbones (mmseg-registered)
├── mmcv_custom/                  # Custom datasets / pipelines / losses / optimizers / hooks / train API
├── models/                       # Custom models (HiViT-related)
├── configs/                      # Paper-related training/eval configs
│   ├── itpn/                     # UperNet + iTPN (AIR-PolarSAR-Seg / EarthMap / WHU-OPT-SAR / ADE20K)
│   ├── hivit/                    # UperNet + HiViT (AIR-PolarSAR-Seg-2.0)
│   └── _base_/                   # Dataset / model / schedule / runtime configs
└── tools/                        # Data conversion and train/eval scripts
```

## Environment Setup

```bash
conda create -n itpn_seg python=3.7 -y
conda activate itpn_seg
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

> This repo was developed under mmseg 0.11.0 / mmcv 1.3.x. Adapt if using a newer mmseg
> (e.g. `mmseg.utils.get_root_logger` → mmengine-style API).

## Data Preparation

| Dataset | Default path | Description |
| --- | --- | --- |
| AIR-PolarSAR-Seg-2.0 | `data/air-polarsar-seg-2.0/{train,val}` | Main segmentation dataset in the paper (amplitude RGB) |
| AIR-PolarSAR-Seg | `data/air-polarsar-seg/{train_set,test_set}` | HH-channel version |
| EarthMap | `data/earthmap/{train,val,test}` | Land-cover (SAR + OEM8 labels) |
| WHU-OPT-SAR | `data/whu_opt_sar_mmseg_256_split` | Multi-modal segmentation (256 tiles) |
| ADE20K | `data/ade/ADEChallengeData2016` | General segmentation reference (official model transfer) |

Data conversion tools (`tools/`):

```bash
# Convert AIR-PolSAR-Seg RGB labels to mmseg index labels
python tools/convert_air_polarsar_labels.py --train-dir <train> --test-dir <test>

# Split WHU-OPT-SAR large images into 256 tiles by scene
python tools/split_whu_opt_sar_for_mmseg.py --data-root <root>

# Other helpers: stats_earthmap_label_pixels.py, verify_hivit_pretrain.py,
# visualize_polsar_amplitude_params.py (polarimetric amplitude pseudo-RGB debugging)
```

## Pre-trained Weights

The iTPN / HiViT pre-trained weights are **not committed**; download from `weights/` and place under mmseg's `ckpts/`:

| Model | Default path | Description |
| --- | --- | --- |
| iTPN-Base | `ckpts/itpn_base/checkpoint-1200.pth` | Main experiment |
| iTPN-Large | `ckpts/itpn_large/checkpoint-1200.pth` | Large-model experiment |
| HiViT-Base | `ckpts/hivit/checkpoint-1200.pth` | Comparison |

Override the default path with `--cfg-options model.pretrained=/path/to/ckpt` in the training command.

## Training / Testing

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

## Mapping to Paper Results

- **AIR-PolarSAR-Seg-2.0 segmentation**: UperNet + iTPN-Base / iTPN-Large (`pixel_upernet_itpn_*_air_polarsar2_amp_linux.py`)
- **AIR-PolarSAR-Seg segmentation**: UperNet + iTPN (`pixel_upernet_itpn_*_air_polarsar_hh_linux*.py`)
- **WHU-OPT-SAR segmentation**: UperNet + iTPN (`pixel_upernet_itpn_*_whu_opt_sar_sar_linux.py`)
- **DDHR-SK segmentation**: UperNet + iTPN — reference config & training log under `results/segmentation/ddhr-sk/`
- **Comparison experiments**: UperNet + HiViT (`configs/hivit/`), CLIP-UperNet + iTPN (`clip_upernet_itpn_*`)

See the paper's experimental section for the exact numbers; per-dataset logs/configs (including WHU-OPT-SAR and DDHR-SK) are under `results/segmentation/`.
