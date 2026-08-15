<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

# detection — SAR 旋转目标检测（MMRotate 自定义扩展）

本目录为 SARATR-X-v2 论文中**检测下游任务**的自定义代码，基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架实现，只保留论文相关的自定义部分：

- 自定义骨干：`iTPN_pixel` / `iTPN_CLIP` / `fast_iTPN` / `ReResNet`
- 自定义数据集：`RSARDataset` / `FAIRCSARDataset` / `SARDataset`
- 自定义 Pipeline：`LoadImageFromFileMultiExt`（混合图片后缀）
- 自定义 Hook：`SWAHook` / `SWARestoreHook` / `MMRotateNumClassCheckHook`
- 自定义优化器：`LayerDecayOptimizerConstructor`（HiViT 层衰减版）

> 框架本体（mmrotate / mmcv / mmdet）需要用户自行安装；本目录提供 **自定义文件 + 配置 + 说明**，复制到 mmrotate 安装目录即可运行。

## 目录结构

```
detection/
├── README.md                     # 本说明
├── configs/                      # 论文相关的训练/评测配置
│   ├── redet/                    # ReDet + iTPN (RSAR / FAIR-CSAR / SIVED)
│   ├── roi_trans/                # RoI Transformer + iTPN (FAIR-CSAR FSI/SL)
│   ├── oriented_rcnn/            # Oriented R-CNN + iTPN (RSAR / FAIR-CSAR SL)
│   ├── oriented_reppoints/       # Oriented RepPoints + iTPN (SIVED)
│   ├── rotated_atss/             # Rotated ATSS + iTPN (SIVED)
│   ├── rotated_fcos/             # Rotated FCOS + iTPN (SIVED)
│   ├── rotated_faster_rcnn/      # Rotated Faster R-CNN + iTPN (DOTA / SIVED)
│   ├── rotated_retinanet/        # Rotated RetinaNet + iTPN (FAIR-CSAR FSI)
│   └── _base_/                   # 数据集 / 调度 / 运行时
└── mmrotate/                     # 自定义模块（复制到 mmrotate 安装目录）
    ├── models/backbones/         # iTPN_pixel、iTPN_CLIP、fast_itpn、ReResNet + mmcv_custom
    ├── datasets/                 # sar.py (RSAR/SAR)、faircsar.py (FAIR-CSAR)
    ├── core/                     # SWA 相关 Hook、类别数检查 Hook
    └── apis/train.py             # 训练入口
```

## 环境搭建

参考 `Rotated_fast_itpn` 子项目（DOTA 版本）的环境配置：

```bash
# 1. 创建 conda 环境
conda create -n rotate_itpn python=3.8 -y
conda activate rotate_itpn

# 2. 安装 PyTorch（示例 CUDA 11.3）
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 3. 安装 mmcv（建议源码编译以避免部分 bug）
pip install mmcv==2.0.0rc4
# 或源码安装：wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.0.0rc4.zip

# 4. 安装 mmdet
pip install mmdet==3.0.0rc6

# 5. 安装 mmrotate
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
```

> 注：本仓库配置使用 `EpochBasedRunner` / `evaluation` / `custom_hooks` 的 **MMRotate 0.3.x 风格 API**，若使用 mmrotate 1.x（MMEngine Runner），需替换为 `train_cfg`/`val_cfg`/`optim_wrapper` 等新写法。请以实际安装版本为准。

## 集成步骤

把本目录的自定义文件复制到 mmrotate 安装目录：

```bash
# 在 mmrotate 仓库根目录执行
cp -r mmrotate/ ./          # 覆盖 mmrotate 包内的自定义模块
cp -r configs/ ./           # 覆盖/新增论文相关配置
```

## 数据集

| 数据集 | 默认路径 | 环境变量 | 说明 |
| --- | --- | --- | --- |
| RSAR | `data/rsar/` | — | DOTA 格式，6 类（ship/aircraft/car/tank/bridge/harbor） |
| FAIR-CSAR SL | `data/faircsar/SL-TRAINVAL/...` | `FAIRCSAR_SL_TRAIN` / `FAIRCSAR_SL_TEST` | 单视 |
| FAIR-CSAR FSI | `data/faircsar/FSI-TRAINVAL/...` | `FAIRCSAR_FSI_TRAIN` / `FAIRCSAR_FSI_TEST` | 多视 |
| DOTA | `data/split_dota/...` | — | 旋转检测标准基准 |
| SIVED | `data/sived/` | `SIVED_ROOT` | DOTA 风格 TXT 标注 |

数据目录统一在 mmrotate 安装目录下，或通过环境变量指定绝对路径。

## 预训练权重

iTPN 预训练权重不随仓库提交，请从 `weights/` 获取并放置到 mmrotate 的 `ckpts/` 目录：

| 模型 | 默认路径 | 环境变量 |
| --- | --- | --- |
| iTPN-Base (checkpoint-1200) | `ckpts/iTPN/checkpoint-1200.pth` | `ITPN_CKPT` |
| iTPN-Large (checkpoint-1200) | `ckpts/iTPN_large/checkpoint-1200.pth` | `ITPN_LARGE_CKPT` |
| ResNet-50 (R50 baseline) | `ckpts/resnet50-0676ba61.pth` | `RESNET50_CKPT` |
| fast_iTPN (DOTA configs) | `ckpts/iTPN/fast_itpn_*.pt` | — |

## 训练 / 测试

```bash
# 训练（4 卡示例）
bash tools/dist_train.sh configs/redet/redet_itpn_base_3x_rsar.py 4
bash tools/dist_train.sh configs/redet/redet_itpn_base_3x_faircsar_sl.py 4

# 测试（输出 mAP50）
bash tools/dist_test.sh configs/redet/redet_itpn_base_3x_rsar.py \
    work_dirs/redet_itpn_base_3x_rsar/latest.pth 4 --eval mAP

# 设置数据/权重路径（示例）
export FAIRCSAR_SL_TRAIN=/path/to/FAIR-CSAR/SL-TRAINVAL/trainval
export ITPN_CKPT=/path/to/checkpoint-1200.pth
```

## 论文对应关系

- **RSAR 检测**：ReDet / Oriented R-CNN + iTPN-Base（`redet_itpn_base_3x_rsar.py` 等）
- **FAIR-CSAR 检测**：ReDet + iTPN-Base / iTPN-Large（SL 与 FSI 两种设置）
- **DOTA 检测**：Rotated Faster R-CNN + fast_iTPN（`rotated_faster_rcnn` 下 DOTA 配置）
- **SIVED 检测**：多种检测头 + iTPN（`rotated_atss` / `rotated_fcos` 等）
- **SARDet-100K / SSDD / HRSID 检测（HBB）**：GFL + iTPN — 参考配置与训练日志见 `results/detection/`

具体数值以论文实验章节为准；各数据集日志与配置见 `results/detection/`。
