# segmentation — SAR 语义分割（MMSegmentation 自定义扩展）

本目录为 SARATR-X-v2 论文中**分割下游任务**的自定义代码，基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)（0.11.0 + mmcv 1.3.x）实现，保留论文相关的自定义部分：

- 自定义骨干：`iTPN`（`backbone/iTPN.py`，与预训练 iTPN 同源）、`HiViT`（`backbone/hivit.py`）
- 自定义数据集：`AIRPolSARSegDataset`（AIR-PolSAR-Seg 6 类）、`EarthMapOEM8OversampleDataset`
- 自定义 Pipeline：`LoadHHAs3ChSAR` / `LoadPreprocessedGrayAs3Ch` / `LoadPolSARAmplitudeRGB`（SAR 幅度/极化数据加载）
- 自定义损失：`CEFocalLoss`、`CEAndLovaszLoss`
- 自定义优化器：`LayerDecayOptimizerConstructor`（含 iTPN / HiViT 层衰减版）
- 自定义评测：`rs_metrics`（PA/OA/mIoU/mPA/Kappa）、`val_test_eval_hooks`
- 自定义训练入口：`mmcv_custom/train_api.py`（`tools/train.py` 自动使用仓库内实现）

## 目录结构

```
segmentation/
├── README.md                     # 本文件
├── backbone/                     # iTPN / HiViT / BEiT 骨干（mmseg 注册版）
├── mmcv_custom/                  # 自定义数据集 / Pipeline / 损失 / 优化器 / Hook / 训练 API
├── models/                       # 自定义模型（HiViT 相关）
├── configs/                      # 论文相关训练/评测配置
│   ├── itpn/                     # UperNet + iTPN（AIR-PolarSAR-Seg / EarthMap / WHU-OPT-SAR / ADE20K）
│   ├── hivit/                    # UperNet + HiViT（AIR-PolarSAR-Seg-2.0）
│   └── _base_/                   # 数据集 / 模型 / 学习率调度 / 运行时
└── tools/                        # 数据转换与训练/评测脚本
```

## 环境搭建

```bash
conda create -n itpn_seg python=3.7 -y
conda activate itpn_seg
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

> 本仓库代码在 mmseg 0.11.0 / mmcv 1.3.x 下开发。若使用新版 mmseg 需适配
> （如 `mmseg.utils.get_root_logger` → mmengine 风格 API）。

## 数据准备

| 数据集 | 默认路径 | 说明 |
| --- | --- | --- |
| AIR-PolarSAR-Seg-2.0 | `data/air-polarsar-seg-2.0/{train,val}` | 论文分割主数据集（幅度 RGB） |
| AIR-PolarSAR-Seg | `data/air-polarsar-seg/{train_set,test_set}` | HH 通道版 |
| EarthMap | `data/earthmap/{train,val,test}` | 地物分类（SAR + OEM8 标签） |
| WHU-OPT-SAR | `data/whu_opt_sar_mmseg_256_split` | 多模态分割（256 瓦片） |
| ADE20K | `data/ade/ADEChallengeData2016` | 通用分割参考（官方模型迁移） |

数据转换工具（`tools/`）：

```bash
# 将 AIR-PolSAR-Seg RGB 标签转换为 mmseg 索引标签
python tools/convert_air_polarsar_labels.py --train-dir <train> --test-dir <test>

# 将 WHU-OPT-SAR 大图按场景切分为 256 瓦片
python tools/split_whu_opt_sar_for_mmseg.py --data-root <root>

# 其它辅助：stats_earthmap_label_pixels.py（标签像素统计）、
# verify_hivit_pretrain.py（HiViT 预训练权重校验）、
# visualize_polsar_amplitude_params.py（极化幅度伪 RGB 参数调试）
```

## 预训练权重

iTPN / HiViT 预训练权重不随仓库提交，从 `weights/` 获取后放置到 mmseg 目录：

| 模型 | 默认路径 | 说明 |
| --- | --- | --- |
| iTPN-Base | `ckpts/itpn_base/checkpoint-1200.pth` | 论文主实验 |
| iTPN-Large | `ckpts/itpn_large/checkpoint-1200.pth` | 大模型实验 |
| HiViT-Base | `ckpts/hivit/checkpoint-1200.pth` | 对比实验 |

可在训练命令中用 `--cfg-options model.pretrained=/path/to/ckpt` 覆盖默认路径。

## 训练 / 测试

```bash
# 训练（8 GPU 示例，AIR-PolarSAR-Seg-2.0 幅度版）
bash tools/dist_train.sh \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py 8 \
    --work-dir work_dirs/air_polarsar2 --seed 0 --deterministic

# 测试（输出 mIoU / mPA / Kappa）
bash tools/dist_test.sh \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py \
    work_dirs/air_polarsar2/latest.pth 8 --eval mIoU

# 覆盖权重/数据路径
python tools/train.py \
    configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py \
    --cfg-options model.pretrained=/path/to/checkpoint-1200.pth data_root=/path/to/data
```

## 论文对应关系

- **AIR-PolarSAR-Seg-2.0 分割**：UperNet + iTPN-Base / iTPN-Large（`pixel_upernet_itpn_*_air_polarsar2_amp_linux.py`）
- **AIR-PolarSAR-Seg 分割**：UperNet + iTPN（`pixel_upernet_itpn_*_air_polarsar_hh_linux*.py`）
- **WHU-OPT-SAR 分割**：UperNet + iTPN（`pixel_upernet_itpn_*_whu_opt_sar_sar_linux.py`）
- **对比实验**：UperNet + HiViT（`configs/hivit/`）、CLIP-UperNet + iTPN（`clip_upernet_itpn_*`）

具体数值以论文实验章节为准。
