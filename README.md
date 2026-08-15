<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

<div align="center">

# SARATR-X-v2: Scale-Aware Structural Pre-Training for SAR Foundation Models

<h5 align="center"><em> Weijie Li, Yafei Song, Yongxiang Liu, Bowen Peng, Jie Zhou, Jingyuan Xia, Wei Yang, Tianpeng Liu, Zhen Liu, and Li Liu </em></h5>

<p align="center">
  <a href="#Introduction">📖 Introduction</a> |
  <a href="#Pre-training">⚙️ Pre-training</a> |
  <a href="#Classification">📊 Classification</a> |
  <a href="#Detection">🎯 Detection</a> |
  <a href="#Segmentation">🧩 Segmentation</a> |
  <a href="#Statement">📜 Statement</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.23238"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
  <a href="https://github.com/waterdisappear/SARATR-X-v2"><img src="https://img.shields.io/badge/Code-GitHub-blue"></a>
  <a href="https://github.com/waterdisappear/SARATR-X-v2/releases"><img src="https://img.shields.io/badge/Data%26Checkpoint-Release-yellow"></a>
  <a href="https://zhuanlan.zhihu.com/p/2070277357513004786"><img src="https://img.shields.io/badge/文章-知乎-blue"></a>
</p>

<p align="center">
  <img src="docs/figures/fig_framework_v3.png" width="88%">
</p>

<div align="center"><b>Fig. 1 | Overall framework of SARATR-X-v2 with scale-aware structural pre-training.</b> Bottom: fixed multi-scale structural extractors spanning six receptive fields produce scale-specific responses, fused by learnable cross-scale weights into one target <i>y</i>, each operator robust to multiplicative speckle. Top: hierarchical encoder extracts multi-scale latent features from the masked input, and a decoder reconstructs <i>y</i> under an L2 loss — <i>design the target to carry the physics</i>.</div>

</div>

## Introduction

SARATR-X-v2 is a vision foundation model for SAR target recognition, built upon **scale-aware structural masked pre-training**. Its core idea is a reconstruction target that is **robust to speckle noise** and **semantically consistent across scales**, enabling state-of-the-art transfer performance on classification, detection, and segmentation.

This repository provides a clean, well-commented, and executable implementation of *SARATR-X-v2*, including pre-training, linear / few-shot classification, rotated object detection, semantic segmentation, and paper-figure visualization.

<p align="center">
  <img src="docs/figures/fig1_motivation_v2.png" width="85%">
</p>

<div align="center"><b>Fig. 2 | Perturbation-sensitive supervision and task-scale mismatch limit SAR pre-training.</b> (a) SAR-specific speckle perturbations destabilize pixel-space supervision, while downstream tasks require representations at different scales; (b) our method constructs a fine-to-coarse structural target and performs hierarchical pre-training, yielding perturbation-stable and scale-compatible representations; (c) the resulting representations achieve leading transfer performance across twelve downstream benchmarks.</div>

## Highlights

- **Multi-scale structural target**: reconstruct a physical-prior target rather than raw pixels;
  - Finest scale `S1`: **blind-spot averaging**, 3×3 kernel with zero center weight, insensitive to speckle;
  - Larger scales `S2`–`S6`: **log-ratio directional contrast**, disjoint half-region kernels separate structure from speckle;
- **Learnable cross-scale fusion**: softmax-constrained weights fuse multi-scale structural responses into a single scale-aware target;
- **Speckle robustness**: `S_multi < S_k < S_pixel`, the fused target is the most stable under coherent-imaging perturbation (see `visualize/`);
- **12 SAR benchmarks**: classification (ATRNet-STAR / MSTAR / SAR-VSA / FUSAR-Ship), detection (SARDet-100K / RSAR / SSDD / HRSID), segmentation (AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK).

## Repository Structure

```
SARATR-X-v2/
├── pre-training/            # Pre-training (iTPN-FPN + multi-scale structural target)
├── classification/          # Downstream classification: linear/k-NN + few-shot (Dassl)
├── detection/               # Downstream detection: MMRotate custom extension
├── segmentation/            # Downstream segmentation: MMSeg custom extension
├── visualize/               # Paper-figure reproduction scripts
├── dataset/                 # Dataset conventions & acquisition
├── weights/                 # Pre-trained weights (large files, uploaded separately)
├── results/                 # Experiment logs & configs (detection/segmentation) + visualize data
└── docs/figures/            # Paper figures used in this README (PNG/PDF)
```

## Pre-training

SARATR-X-v2 is pre-trained on **500K unlabeled SAR target chips** via scale-aware structural masked pre-training. The reconstruction target is constructed from six fixed structural extractors spanning different receptive fields and fused by learnable weights into one unified supervision signal. See `pre-training/` for details.

| Target | Description |
|---|---|
| `pixel` | Raw-pixel reconstruction (baseline) |
| `single_s1` | S1: 3×3 blind-spot neighborhood aggregation (zero center) |
| `single_s2..s6` | S2~S6: log-ratio directional contrast, r = 3/5/9/13/17 |
| `multi` | **Main setting**: softmax-constrained learnable multi-scale fusion |

```bash
cd pre-training
bash run_pretrain.sh                 # 8 GPUs
python main_pretrain.py --data_path <500K pre-training data> --output_dir <work_dir>  # or:
```

## Classification

Linear probing and k-NN / few-shot classification on SAR benchmarks with a frozen iTPN / HiViT backbone. See `classification/linear_eval/` and `classification/fewshot/`.

```bash
cd classification/linear_eval
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_path ../../dataset/classification/SOC_50classes
```

| Dataset | Method | Paper result |
| --- | --- | --- |
| ATRNet-STAR (SOC-50) | Linear probing, iTPN-B/L | 98.4 |
| MSTAR | Linear probing (few-shot) | 98.7 |
| SAR-VSA | Linear probing / k-NN | 90.8 |
| FUSAR-Ship | k-NN | 93.0 |

## Detection

Rotated object detection on SAR benchmarks with iTPN backbones. This repo ships the **custom MMRotate parts** (backbones, datasets, configs, hooks) for RSAR / FAIR-CSAR / DOTA / SIVED; the framework itself is installed by the user. The paper also reports detection results on SARDet-100K / SSDD / HRSID (GFL + iTPN, horizontal boxes); their reference configs and training logs are provided under `results/detection/`. See `detection/`.

```bash
# after copying detection/ into the mmrotate install:
bash tools/dist_train.sh configs/redet/redet_itpn_base_3x_rsar.py 4
bash tools/dist_test.sh configs/redet/redet_itpn_base_3x_rsar.py \
    work_dirs/redet_itpn_base_3x_rsar/latest.pth 4 --eval mAP
```

## Segmentation

Semantic segmentation on SAR benchmarks with UperNet + iTPN / HiViT. This repo ships the **custom MMSeg parts** for AIR-PolarSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR (and ADE20K as reference); the framework itself is installed by the user. The paper also reports segmentation results on DDHR-SK / WHU-OPT-SAR, whose reference configs and training logs are provided under `results/segmentation/`. See `segmentation/`.

```bash
bash tools/dist_train.sh \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py 8
```

## Results

SARATR-X-v2 achieves state-of-the-art transfer performance on **12 SAR benchmarks** across classification, detection, and segmentation. The detection/segmentation training logs and reference configs are under `results/detection/` and `results/segmentation/`; figure-reproduction scripts are under `visualize/`.

<p align="center">
  <img src="docs/figures/sar_benchmark_4x3_horizontal_bar.png" width="85%">
</p>

<div align="center"><b>Fig. 3 | Comprehensive comparison of SARATR-X-v2 on twelve SAR benchmarks</b> spanning classification (ATRNet-STAR, MSTAR, SAR-VSA, FUSAR-Ship), object detection (SARDet-100K, RSAR, SSDD, HRSID), and semantic segmentation (AIR-PolSAR-Seg-2.0, OpenEarthMap-SAR, WHU-OPT-SAR, DDHR-SK). SARATR-X-v2 achieves the best result on 10 of 12 benchmarks and second-best on the remaining two.</div>

Under synthetic speckle variation, the proposed target reduces perturbation drift in the learned representation by **nearly two orders of magnitude** relative to pixel-space supervision.

<p align="center">
  <img src="docs/figures/stability_transfer_scatter.png" width="55%">
</p>

<div align="center"><b>Fig. 4 | Stability–transfer relationship across pre-training targets.</b> Each point is one supervision target (pixel, single-scale S1–S6, multi-scale fusion). Horizontal axis: mean ℓ₁ drift under speckle at σ = 0.15; vertical axis: 10-shot linear-probe accuracy on ATRNet-STAR (SOC-50) with frozen iTPN-B. Lower drift correlates with higher accuracy (ρ = −0.93, p = 0.002).</div>

## Data

See `dataset/README.md` for directory conventions and how to obtain the datasets.

- **Pre-training**: FAIR-CSAR large-scale SAR images (500K)
- **Classification**: SOC-50 / SAR-VSA / FUSAR-Ship / ATRNet-STAR / MSTAR
- **Detection**: RSAR / FAIR-CSAR / SARDet-100K / SSDD / HRSID / DOTA
- **Segmentation**: AIR-PolarSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK

## Weights & Results

Pre-trained weights are large and **not committed**; download them from the release and place them per `weights/README.md`. Experiment logs/configs (detection/segmentation) and the small visualization csv/json are committed under `results/`.

- `weights/README.md`: pre-trained weight layout (iTPN-B/L × fusion targets/ablations) and mapping to downstream tasks; `checkpoint-1200.pth` by default;
- `results/README.md`: downstream result layout;

## Statement

### Citation

```bibtex
@article{saratrxv2,
  title={SARATR-X-v2: Scale-Aware Structural Pre-Training for SAR Foundation Models},
  author={Li, Weijie and Song, Yafei and Liu, Yongxiang and Peng, Bowen and Zhou, Jie and
          Xia, Jingyuan and Yang, Wei and Liu, Tianpeng and Liu, Zhen and Liu, Li},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  year={2026},
  eprint={2607.23238},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

Paper: [arXiv:2607.23238](https://arxiv.org/abs/2607.23238)

### Acknowledgement

This implementation references the following open-source projects: iTPN, HiViT, Dassl (CoOp/CoCoOp), MMRotate, MMSegmentation, mmcv, timm.

### License

This repository is released under the MIT License (see `LICENSE`).
