<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

# 线性 / k-NN 评测

本目录包含 SAR 基准上的线性探测与 k-NN 分类评测脚本（论文协议：冻结骨干的线性评测 / k-NN）。

## 脚本

| 脚本 | 数据集 | 骨干 | 任务 |
| --- | --- | --- | --- |
| `SOC_50.py` | ATRNet-STAR (SOC-50) | iTPN-Base | 线性探测 |
| `SOC_50_large.py` | ATRNet-STAR (SOC-50) | iTPN-Large | 线性探测 |
| `SARVASA.py` | SAR-VSA | iTPN-Base | 线性探测 |
| `SARVSA_large.py` | SAR-VSA | iTPN-Large | 线性探测 |
| `zero_shot.py` | ImageFolder 数据集（如 SAR-VSA） | iTPN-Base | k-NN 分类 |
| `zero_shot_newfusar.py` | FUSAR-Ship | iTPN-Base | k-NN 分类 |

`utils/` 中的共享工具：`DataLoad.py`（数据加载）、`TrainTest.py`（线性训练评测）、`knn_utils.py` / `knn_eval.py`（k-NN 工具）。

## 使用

```bash
cd classification/linear_eval

# SOC-50 线性探测（iTPN-Base）
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_path ../../dataset/classification/SOC_50classes

# FUSAR-Ship k-NN
python zero_shot_newfusar.py --data_path ../../dataset/classification/FUSAR-Ship
```

- 权重通过 `SARATRX_PRETRAIN_CKPT` 解析（默认 `weights/base/jiaquan_simple/checkpoint-1200.pth`），加载逻辑见 `model/models_itpn.py`。
- 数据按 ImageFolder 组织，位于 `../../dataset/classification/`（见 [`../../dataset/README_zh.md`](../../dataset/README_zh.md)）。

## 依赖

`torch`、`torchvision`、`numpy`、`opencv-python`、`faiss-cpu`（k-NN）。脚本从自身目录运行即可（`utils` / `model` 包就在本目录下，无需额外 PYTHONPATH）。
