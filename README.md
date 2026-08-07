<div align="center">

# SARATR-X-v2

**Scale-Aware Structural Pre-Training for SAR Foundation Models**

Official implementation (clean & executable) of *SARATR-X-v2*.

</div>

SARATR-X-v2 是一个面向 SAR 目标识别的视觉基础模型，采用 **尺度感知结构掩码预训练（Scale-aware Structural Masked Pre-training）**。其核心是一套对**相干斑噪声（speckle）稳健**、且跨尺度语义一致的重建目标，通过统一的预训练在分类、检测、分割三类下游任务上取得全面领先。

This repository provides a clean, well-commented, and executable implementation of *SARATR-X-v2*, including pre-training, linear / few-shot classification, rotated object detection, semantic segmentation, and paper-figure visualization.

## Highlights（方法亮点）

- **多尺度结构目标（Multi-scale Structural Target）**：以 SAR 物理先验构造重建目标，而非直接重建像素。
  - 最细尺度 `S1`：**盲点平均（blind-spot averaging）**，3×3 核中心权重为 0，对斑点不敏感。
  - 更大尺度 `S2`–`S6`：**对数比方向对比（log-ratio directional contrast）**，相邻半区互斥核分离结构与斑点。
- **可学习跨尺度融合（Learnable Cross-scale Fusion）**：softmax 约束的融合权重，将多尺度结构响应合成为单一尺度感知目标。
- **斑点稳健性设计**：`S_multi < S_k < S_pixel`，融合目标对相干成像扰动最稳定（见 `visualize/` 的稳定性实验）。
- **12 个 SAR 基准全面领先**：分类（ATRNet-STAR / MSTAR / SAR-VSA / FUSAR-Ship）、检测（SARDet-100K / RSAR / SSDD / HRSID）、分割（AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK）。

## Repository Structure（仓库结构）

```
SARATR-X-v2/
├── pre-training/            # 预训练（iTPN-FPN + 多尺度结构目标，main_pretrain.py）
├── classification/          # 下游分类：linear_eval（线性/k-NN）+ fewshot（Dassl 少样本）
├── detection/               # 下游检测：MMRotate 自定义扩展（iTPN 骨干 + SAR 数据集）
├── segmentation/            # 下游分割：MMSeg 自定义扩展（UperNet + iTPN / HiViT）
├── visualize/               # 论文图表复现脚本（与论文图一一对应）
├── dataset/                 # 数据集说明与获取方式
├── weights/                 # 预训练权重（大文件，单独上传）
├── results/                 # 实验结果（大文件，单独上传；results/visualize 随仓库提交）
└── docs/                    # 附加文档
```

## Quick Start（快速开始）

各模块环境相互独立，请按各自 README 安装依赖：

| 模块 | 环境 | 入口 |
| --- | --- | --- |
| 预训练 | Python 3.8+, torch, timm | `pre-training/main_pretrain.py` |
| 线性 / k-NN 评测 | torch, faiss | `classification/linear_eval/SOC_50.py` 等 |
| 少样本 | Dassl (CoOp) 框架 | `classification/fewshot/train.py` |
| 检测 | mmrotate / mmcv / mmdet | `detection/configs/` + 框架 tools |
| 分割 | mmsegmentation 0.11 | `segmentation/configs/` + 框架 tools |
| 可视化 | torch, matplotlib, opencv | `visualize/*.py` |

### 1. 预训练

```bash
cd pre-training
bash run_pretrain.sh                 # 或：
python main_pretrain.py --data_path <500K 预训练数据> --output_dir <work_dir>
```

### 2. 分类评测（线性 / k-NN）

```bash
cd classification/linear_eval
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_root ../../dataset/classification/SOC_50classes
```

### 3. 检测 / 分割

检测与分割使用外部框架（mmrotate / mmseg），本仓库提供**自定义部分**（骨干、数据集、配置），按各子目录 README 集成到框架即可。

## Data（数据）

数据集的目录约定见 `dataset/README.md`，主要数据集：

- **预训练**：FAIR-CSAR 大规模 SAR 影像（500K 张）
- **分类**：SOC-50 / SAR-VSA / FUSAR-Ship / ATRNet-STAR / MSTAR
- **检测**：RSAR / FAIR-CSAR / SARDet-100K / SSDD / HRSID / DOTA
- **分割**：AIR-PolarSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK

## Weights & Results（权重与结果）

- `weights/README.md`：预训练权重目录结构（iTPN-B/L × 融合目标/消融）与下游任务对应关系，默认使用 `checkpoint-1200.pth`。
- `results/README.md`：下游实验结果目录结构；`results/visualize/` 下的实验数据（csv/json）随仓库提交，可直接复现论文图表。

权重与结果文件较大，**不随仓库提交**，请从 release 附件获取后按 README 放置。

## Citation（引用）

```bibtex
@article{saratrxv2,
  title={SARATR-X-v2: Scale-Aware Structural Pre-Training for SAR Foundation Models},
  author={Li, Weijie and Song, Yafei and Liu, Yongxiang and others},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  year={2026}
}
```

## Acknowledgement（致谢）

本实现参考了以下开源项目：iTPN、HiViT、Dassl (CoOp/CoCoOp)、MMRotate、MMSegmentation、mmcv、timm。

## License

本仓库代码以 MIT License 发布（详见 `LICENSE`）。
