<div align="center">

# SARATR-X-v2

**Scale-Aware Structural Pre-Training for SAR Foundation Models**

[![arXiv](https://img.shields.io/badge/arXiv-2607.23238-b31b1b.svg)](https://arxiv.org/abs/2607.23238)

Official implementation (clean & executable) of *SARATR-X-v2*.

</div>

## Introduction / 简介

SARATR-X-v2 is a vision foundation model for SAR target recognition, built upon **scale-aware structural masked pre-training**. Its core idea is a reconstruction target that is **robust to speckle noise** and **semantically consistent across scales**, enabling state-of-the-art transfer performance on classification, detection, and segmentation.

SARATR-X-v2 是一个面向 SAR 目标识别的视觉基础模型，采用 **尺度感知结构掩码预训练（Scale-aware Structural Masked Pre-training）**。其核心是一套对 **相干斑噪声（speckle）稳健**、且跨尺度语义一致的重建目标，通过统一的预训练在分类、检测、分割三类下游任务上取得全面领先。

This repository provides a clean, well-commented, and executable implementation of *SARATR-X-v2*, including pre-training, linear / few-shot classification, rotated object detection, semantic segmentation, and paper-figure visualization.

本仓库提供 *SARATR-X-v2* 的清晰、易读、可执行的官方实现，涵盖预训练、线性 / 少样本分类、旋转目标检测、语义分割以及论文图表复现可视化。

## Highlights / 方法亮点

- **Multi-scale structural target / 多尺度结构目标**：reconstruct a physical-prior target rather than raw pixels; 以 SAR 物理先验构造重建目标，而非直接重建像素。
  - Finest scale `S1` / 最细尺度 `S1`：**blind-spot averaging / 盲点平均**，3×3 kernel with zero center weight, insensitive to speckle; 3×3 核中心权重为 0，对斑点不敏感。
  - Larger scales `S2`–`S6` / 更大尺度 `S2`–`S6`：**log-ratio directional contrast / 对数比方向对比**，disjoint half-region kernels separate structure from speckle; 相邻半区互斥核分离结构与斑点。
- **Learnable cross-scale fusion / 可学习跨尺度融合**：softmax-constrained weights fuse multi-scale structural responses into a single scale-aware target; softmax 约束的融合权重将多尺度结构响应合成为单一尺度感知目标。
- **Speckle robustness / 斑点稳健性**：`S_multi < S_k < S_pixel`, the fused target is the most stable under coherent-imaging perturbation (see `visualize/`); 融合目标对相干成像扰动最稳定（见 `visualize/` 的稳定性实验）。
- **12 SAR benchmarks / 12 个 SAR 基准全面领先**：classification (ATRNet-STAR / MSTAR / SAR-VSA / FUSAR-Ship), detection (SARDet-100K / RSAR / SSDD / HRSID), segmentation (AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK); 分类（ATRNet-STAR / MSTAR / SAR-VSA / FUSAR-Ship）、检测（SARDet-100K / RSAR / SSDD / HRSID）、分割（AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK）。

## Repository Structure / 仓库结构

```
SARATR-X-v2/
├── pre-training/            # Pre-training (iTPN-FPN + multi-scale structural target) / 预训练（iTPN-FPN + 多尺度结构目标）
├── classification/          # Downstream classification: linear/k-NN + few-shot (Dassl) / 下游分类：线性/k-NN + 少样本
├── detection/               # Downstream detection: MMRotate custom extension / 下游检测：MMRotate 自定义扩展
├── segmentation/            # Downstream segmentation: MMSeg custom extension / 下游分割：MMSeg 自定义扩展
├── visualize/               # Paper-figure reproduction scripts / 论文图表复现脚本
├── dataset/                 # Dataset conventions & acquisition / 数据集说明与获取方式
├── weights/                 # Pre-trained weights (large files, uploaded separately) / 预训练权重（大文件，单独上传）
└── results/                 # Experiment results (large files; results/visualize committed) / 实验结果（大文件；results/visualize 随仓库提交）
```

## Quick Start / 快速开始

Each module has an independent environment; please install dependencies per its README. 各模块环境相互独立，请按各自 README 安装依赖：

| Module / 模块 | Environment / 环境 | Entry / 入口 |
| --- | --- | --- |
| Pre-training / 预训练 | Python 3.8+, torch, timm | `pre-training/main_pretrain.py` |
| Linear / k-NN eval / 线性 / k-NN 评测 | torch, faiss | `classification/linear_eval/SOC_50.py` |
| Few-shot / 少样本 | Dassl (CoOp) | `classification/fewshot/train.py` |
| Detection / 检测 | mmrotate / mmcv / mmdet | `detection/configs/` + framework tools |
| Segmentation / 分割 | mmsegmentation 0.11 | `segmentation/configs/` + framework tools |
| Visualization / 可视化 | torch, matplotlib, opencv | `visualize/*.py` |

### 1. Pre-training / 预训练

```bash
cd pre-training
bash run_pretrain.sh                 # or / 或：
python main_pretrain.py --data_path <500K pre-training data> --output_dir <work_dir>
```

### 2. Classification (linear / k-NN) / 分类评测（线性 / k-NN）

```bash
cd classification/linear_eval
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_root ../../dataset/classification/SOC_50classes
```

### 3. Detection / Segmentation / 检测 / 分割

Detection and segmentation use external frameworks (mmrotate / mmseg); this repo provides the **custom parts** (backbones, datasets, configs). Integrate them into the framework as described in each sub-directory README.

检测与分割使用外部框架（mmrotate / mmseg），本仓库提供 **自定义部分**（骨干、数据集、配置），按各子目录 README 集成到框架即可。

## Data / 数据

See `dataset/README.md` for directory conventions. 数据集的目录约定见 `dataset/README.md`。Main datasets / 主要数据集：

- **Pre-training / 预训练**：FAIR-CSAR large-scale SAR images (500K) / 大规模 SAR 影像（500K 张）
- **Classification / 分类**：SOC-50 / SAR-VSA / FUSAR-Ship / ATRNet-STAR / MSTAR
- **Detection / 检测**：RSAR / FAIR-CSAR / SARDet-100K / SSDD / HRSID / DOTA
- **Segmentation / 分割**：AIR-PolarSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK

## Weights & Results / 权重与结果

- `weights/README.md`：pre-trained weight layout (iTPN-B/L × fusion targets/ablations) and mapping to downstream tasks; `checkpoint-1200.pth` by default; 预训练权重目录结构与下游任务对应关系，默认使用 `checkpoint-1200.pth`。
- `results/README.md`：downstream result layout; `results/visualize/` csv/json are committed for figure reproduction; 下游实验结果目录结构；`results/visualize/` 下的实验数据随仓库提交，可直接复现论文图表。

Weights and results are large and **not committed**; download from the release and place them per the READMEs. 权重与结果文件较大，**不随仓库提交**，请从 release 附件获取后按 README 放置。

## Citation / 引用

```bibtex
@article{saratrxv2,
  title={SARATR-X-v2: Scale-Aware Structural Pre-Training for SAR Foundation Models},
  author={Li, Weijie and Song, Yafei and Liu, Yongxiang and others},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  year={2026},
  eprint={2607.23238},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

Paper: [arXiv:2607.23238](https://arxiv.org/abs/2607.23238) / 论文链接：[arXiv:2607.23238](https://arxiv.org/abs/2607.23238)

## Acknowledgement / 致谢

This implementation references the following open-source projects: iTPN, HiViT, Dassl (CoOp/CoCoOp), MMRotate, MMSegmentation, mmcv, timm.

本实现参考了以下开源项目：iTPN、HiViT、Dassl (CoOp/CoCoOp)、MMRotate、MMSegmentation、mmcv、timm。

## License / 许可证

This repository is released under the MIT License (see `LICENSE`). 本仓库代码以 MIT License 发布（详见 `LICENSE`）。
