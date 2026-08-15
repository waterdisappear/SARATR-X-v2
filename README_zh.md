<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

<div align="center">

# SARATR-X-v2: Scale-Aware Structural Pre-Training for SAR Foundation Models

<h5 align="center"><em> 李玮杰、宋娅菲、刘永祥、彭渤文、周洁、夏靖远、杨威、刘天鹏、刘振、刘丽 </em></h5>

<p align="center">
  <a href="#简介">📖 简介</a> |
  <a href="#预训练">⚙️ 预训练</a> |
  <a href="#分类">📊 分类</a> |
  <a href="#检测">🎯 检测</a> |
  <a href="#分割">🧩 分割</a> |
  <a href="#声明">📜 声明</a>
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

<div align="center"><b>图 1 | SARATR-X-v2 尺度感知结构预训练总体框架。</b> 下半部分：覆盖六个感受野的固定多尺度结构算子产生各尺度响应，经可学习跨尺度权重融合为统一目标 <i>y</i>，每个算子对乘性斑点稳健；上半部分：层次编码器从掩码输入提取多尺度潜在特征，解码器在 L2 损失下重建 <i>y</i> —— “让目标承载物理”。</div>

</div>

## 简介

SARATR-X-v2 是一个面向 SAR 目标识别的视觉基础模型，采用 **尺度感知结构掩码预训练（Scale-aware Structural Masked Pre-training）**。其核心是一套对 **相干斑噪声（speckle）稳健**、且跨尺度语义一致的重建目标，通过统一的预训练在分类、检测、分割三类下游任务上取得全面领先。

本仓库提供 *SARATR-X-v2* 官方实现，涵盖预训练、线性 / 少样本分类、旋转目标检测、语义分割以及论文图表复现可视化。

<p align="center">
  <img src="docs/figures/fig1_motivation_v2.png" width="85%">
</p>

<div align="center"><b>图 2 | 扰动敏感监督与任务尺度错配制约 SAR 预训练。</b> (a) SAR 特有的斑点扰动使像素空间监督不稳定，而下游任务需要不同尺度的表示；(b) 本方法构造由细到粗的结构目标并进行层次化预训练，得到扰动稳定、尺度兼容的表示；(c) 所得表示在 12 个下游基准上取得领先的迁移性能。</div>

## 方法亮点

- **多尺度结构目标**：以 SAR 物理先验构造重建目标，而非直接重建像素。
  - 最细尺度 `S1`：**盲点平均**，3×3 核中心权重为 0，对斑点不敏感。
  - 更大尺度 `S2`–`S6`：**对数比方向对比**，相邻半区互斥核分离结构与斑点。
- **可学习跨尺度融合**：softmax 约束的融合权重将多尺度结构响应合成为单一尺度感知目标。
- **斑点稳健性**：`S_multi < S_k < S_pixel`，融合目标对相干成像扰动最稳定（见 `visualize/` 的稳定性实验）。
- **12 个 SAR 基准全面领先**：分类（ATRNet-STAR / MSTAR / SAR-VSA / FUSAR-Ship）、检测（SARDet-100K / RSAR / SSDD / HRSID）、分割（AIR-PolSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK）。

## 仓库结构

```
SARATR-X-v2/
├── pre-training/            # 预训练（iTPN-FPN + 多尺度结构目标）
├── classification/          # 下游分类：线性/k-NN + 少样本
├── detection/               # 下游检测：MMRotate 自定义扩展
├── segmentation/            # 下游分割：MMSeg 自定义扩展
├── visualize/               # 论文图表复现脚本
├── dataset/                 # 数据集说明与获取方式
├── weights/                 # 预训练权重（大文件，单独上传）
├── results/                 # 实验结果日志与配置 + 可视化数据
└── docs/figures/            # 论文图
```

## 预训练

SARATR-X-v2 在 **500K 张无标注 SAR 目标切片**上进行尺度感知结构掩码预训练。重建目标由六个覆盖不同感受野的固定结构算子构造，并通过可学习权重融合为统一的监督信号。详见 `pre-training/`。

| 目标 | 说明 |
|---|---|
| `pixel` | 原始像素重建（基线） |
| `single_s1` | S1: 3×3 盲点邻域聚合 |
| `single_s2..s6` | S2~S6: 对数比方向对比，r = 3/5/9/13/17 |
| `multi` | **论文主设置**：可学习多尺度融合 |

```bash
cd pre-training
bash run_pretrain.sh                 # 8 GPUs / 或：
python main_pretrain.py --data_path <500K pre-training data> --output_dir <work_dir>
```

## 分类

在 SAR 基准上对冻结的 iTPN / HiViT 骨干进行线性探测与 k-NN / 少样本分类评测。见 `classification/linear_eval/` 与 `classification/fewshot/`。

```bash
cd classification/linear_eval
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_path ../../dataset/classification/SOC_50classes
```

| 数据集 | 方法 | 论文结果 |
| --- | --- | --- |
| ATRNet-STAR (SOC-50) | Linear probing, iTPN-B/L | 98.4 |
| MSTAR | Linear probing (few-shot) | 98.7 |
| SAR-VSA | Linear probing / k-NN | 90.8 |
| FUSAR-Ship | k-NN | 93.0 |

## 检测

在 SAR 基准上进行旋转目标检测，采用 ReDet / RoI Transformer / Oriented R-CNN + iTPN。本仓库提供 **MMRotate 自定义部分**（骨干、数据集、配置、Hook），覆盖 RSAR / FAIR-CSAR / DOTA / SIVED；框架由用户自行安装。论文中的 SARDet-100K / SSDD / HRSID（GFL + iTPN，水平框）检测结果，其参考配置与训练日志见 `results/detection/`。见 `detection/`。

```bash
# 将 detection/ 复制进 mmrotate 安装目录后：
bash tools/dist_train.sh configs/redet/redet_itpn_base_3x_rsar.py 4
bash tools/dist_test.sh configs/redet/redet_itpn_base_3x_rsar.py \
    work_dirs/redet_itpn_base_3x_rsar/latest.pth 4 --eval mAP
```

## 分割

在 SAR 基准上进行语义分割，采用 UperNet + iTPN / HiViT。本仓库提供 **MMSeg 自定义部分**，覆盖 AIR-PolarSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR（及参考用的 ADE20K）；框架由用户自行安装。论文中的 DDHR-SK / WHU-OPT-SAR 分割结果，其参考配置与训练日志见 `results/segmentation/`。见 `segmentation/`。

```bash
bash tools/dist_train.sh \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py 8
```

## 实验结果

SARATR-X-v2 在覆盖分类、检测、分割的 **12 个 SAR 基准**上取得全面领先的迁移性能。检测与分割的训练日志与参考配置见 `results/detection/` 与 `results/segmentation/`；图表复现脚本见 `visualize/`。

<p align="center">
  <img src="docs/figures/sar_benchmark_4x3_horizontal_bar.png" width="85%">
</p>

<div align="center"><b>图 3 | SARATR-X-v2 在 12 个 SAR 基准上的综合对比</b>，涵盖分类（ATRNet-STAR、MSTAR、SAR-VSA、FUSAR-Ship）、目标检测（SARDet-100K、RSAR、SSDD、HRSID）与语义分割（AIR-PolSAR-Seg-2.0、OpenEarthMap-SAR、WHU-OPT-SAR、DDHR-SK）。SARATR-X-v2 在 12 个基准中取得 10 个最优、2 个次优。</div>

在合成斑点扰动下，所提目标将学习表示的扰动漂移相对像素空间监督降低了**近两个数量级**。

<p align="center">
  <img src="docs/figures/stability_transfer_scatter.png" width="55%">
</p>

<div align="center"><b>图 4 | 各预训练目标的稳定性–迁移关系。</b> 每个点为一种监督目标（像素、单尺度 S1–S6、多尺度融合）。横轴：σ = 0.15 斑点下的均值 ℓ₁ 漂移；纵轴：冻结 iTPN-B 在 ATRNet-STAR（SOC-50）上的 10-shot 线性探测精度。漂移越小精度越高（ρ = −0.93，p = 0.002）。</div>

## 数据

数据集的目录约定与获取方式见 `dataset/README_zh.md`。主要数据集：

- **预训练**：FAIR-CSAR 大规模 SAR 影像（500K 张）
- **分类**：SOC-50 / SAR-VSA / FUSAR-Ship / ATRNet-STAR / MSTAR
- **检测**：RSAR / FAIR-CSAR / SARDet-100K / SSDD / HRSID / DOTA
- **分割**：AIR-PolarSAR-Seg-2.0 / OpenEarthMap-SAR / WHU-OPT-SAR / DDHR-SK

## 权重与结果

预训练权重较大，**不随仓库提交**；请从 release 附件获取后按 `weights/README_zh.md` 放置。检测与分割的实验日志与配置、以及可视化所需的小体积数据已随仓库提交在 `results/`。

- `weights/README_zh.md`：预训练权重目录结构与下游任务对应关系，默认使用 `checkpoint-1200.pth`。
- `results/README_zh.md`：下游实验结果目录结构。

## 声明

### 引用

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

论文链接：[arXiv:2607.23238](https://arxiv.org/abs/2607.23238)

### 致谢

本实现参考了以下开源项目：iTPN、HiViT、Dassl (CoOp/CoCoOp)、MMRotate、MMSegmentation、mmcv、timm。

### 许可证

本仓库代码以 MIT License 发布（详见 `LICENSE`）。
