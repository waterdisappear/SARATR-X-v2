<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

# 少样本分类评测

本目录基于 Dassl 框架，实现 SAR 少样本分类评测（论文协议：冻结 backbone 的线性探测）：

- `MIM_linear_itpn`：iTPN 骨干线性探测（论文 ATRNet-STAR / MSTAR 的主评测）
- `MIM_linear`：HiViT 骨干线性探测（v1 SARATR-X 的骨干，作对比参考）
- `MIM_finetune`：HiViT 少样本全量微调（作对比参考）

## 目录结构

```
fewshot/
├── train.py                    # 评测入口
├── trainers/                   # 训练器
│   ├── MIM_linear_itpn.py
│   ├── MIM_linear.py
│   └── MIM_finetune.py
├── model/                      # 骨干网络
│   ├── models_itpn.py          # iTPN
│   └── hivit.py                # HiViT
├── datasets/                   # SAR 数据集定义
├── configs/
│   ├── datasets/               # 数据集配置
│   └── trainers/               # 训练器配置
└── scripts/
    ├── MIM_linear_itpn/        # iTPN 线性探测脚本 + 目标消融脚本
    ├── MIM_linear/
    └── MIM_finetune/
```

## 环境

需要 Dassl（[KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)）以及 `timm`、`scipy`。安装 Dassl 后将其加入 `PYTHONPATH`。

## 使用

```bash
cd classification/fewshot

# MSTAR 5-shot 线性探测（iTPN-B 骨干）
bash scripts/MIM_linear_itpn/main.sh MSTAR_SOC vit_b16 5

# SOC-50 10-shot 线性探测
bash scripts/MIM_linear_itpn/main.sh SOC_50classes vit_b16 10
```

预训练权重通过环境变量指定，或默认从仓库 `weights/` 目录加载（见 `trainers/MIM_linear_itpn.py` 的 `resolve_pretrain_ckpt`）。

## 消融实验（重建目标）

`scripts/MIM_linear_itpn/run_target_ablation.sh` 依次用 8 种重建目标（pixel、single_s1~s6、multi）的预训练权重在 SOC-50 上做 10-shot 线性探测，权重放在 `weights/base/<mode>/checkpoint-49.pth`。
