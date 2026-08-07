# Linear / k-NN Evaluation / 线性 / k-NN 评测

This folder contains linear-probing and k-NN classification evaluation scripts on SAR benchmarks (protocol in the paper: frozen-backbone linear evaluation / k-NN).

本目录包含 SAR 基准上的线性探测与 k-NN 分类评测脚本（论文协议：冻结骨干的线性评测 / k-NN）。

## Scripts / 脚本

| Script / 脚本 | Dataset / 数据集 | Backbone / 骨干 | Task / 任务 |
| --- | --- | --- | --- |
| `SOC_50.py` | ATRNet-STAR (SOC-50) | iTPN-Base | Linear probing / 线性探测 |
| `SOC_50_large.py` | ATRNet-STAR (SOC-50) | iTPN-Large | Linear probing / 线性探测 |
| `SARVASA.py` | SAR-VSA | iTPN-Base | Linear probing / 线性探测 |
| `SARVSA_large.py` | SAR-VSA | iTPN-Large | Linear probing / 线性探测 |
| `zero_shot.py` | ImageFolder datasets (e.g. SAR-VSA) | iTPN-Base | k-NN (N-shot) / k-NN 分类 |
| `zero_shot_newfusar.py` | FUSAR-Ship | iTPN-Base | k-NN, 384×384 input / k-NN 分类 |

Shared utilities in `utils/`：`DataLoad.py`（ImageFolder loading / 数据加载）、`TrainTest.py`（linear train/eval / 线性训练评测）、`knn_utils.py` / `knn_eval.py`（FAISS k-NN / k-NN 工具）。

## Usage / 使用

```bash
cd classification/linear_eval

# Linear probing on SOC-50 (iTPN-Base)
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_path ../../dataset/classification/SOC_50classes

# k-NN on FUSAR-Ship
python zero_shot_newfusar.py --data_path ../../dataset/classification/FUSAR-Ship
```

- Weights are resolved via `SARATRX_PRETRAIN_CKPT` (default `weights/base/jiaquan_simple/checkpoint-1200.pth`). See `model/models_itpn.py` for the weight-loading logic.
  权重通过 `SARATRX_PRETRAIN_CKPT` 解析（默认 `weights/base/jiaquan_simple/checkpoint-1200.pth`），加载逻辑见 `model/models_itpn.py`。
- Data follows the ImageFolder layout under `../../dataset/classification/`（see `dataset/README.md`）.
  数据按 ImageFolder 组织，位于 `../../dataset/classification/`（见 `dataset/README.md`）。

## Requirements / 依赖

`torch`, `torchvision`, `numpy`, `opencv-python`, `faiss-cpu`（k-NN）。脚本从自身目录运行即可（`utils` / `model` 包就在本目录下，无需额外 PYTHONPATH）。
