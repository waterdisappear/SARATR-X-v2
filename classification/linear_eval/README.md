<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# Linear / k-NN Evaluation

This folder contains linear-probing and k-NN classification evaluation scripts on SAR benchmarks (protocol in the paper: frozen-backbone linear evaluation / k-NN).

## Scripts

| Script | Dataset | Backbone | Task |
| --- | --- | --- | --- |
| `SOC_50.py` | ATRNet-STAR (SOC-50) | iTPN-Base | Linear probing |
| `SOC_50_large.py` | ATRNet-STAR (SOC-50) | iTPN-Large | Linear probing |
| `SARVASA.py` | SAR-VSA | iTPN-Base | Linear probing |
| `SARVSA_large.py` | SAR-VSA | iTPN-Large | Linear probing |
| `zero_shot.py` | ImageFolder datasets (e.g. SAR-VSA) | iTPN-Base | k-NN (N-shot) |
| `zero_shot_newfusar.py` | FUSAR-Ship | iTPN-Base | k-NN, 384×384 input |

Shared utilities in `utils/`: `DataLoad.py` (ImageFolder loading), `TrainTest.py` (linear train/eval), `knn_utils.py` / `knn_eval.py` (FAISS k-NN).

## Usage

```bash
cd classification/linear_eval

# Linear probing on SOC-50 (iTPN-Base)
export SARATRX_PRETRAIN_CKPT=/path/to/weights/base/jiaquan_simple/checkpoint-1200.pth
python SOC_50.py --data_path ../../dataset/classification/SOC_50classes

# k-NN on FUSAR-Ship
python zero_shot_newfusar.py --data_path ../../dataset/classification/FUSAR-Ship
```

- Weights are resolved via `SARATRX_PRETRAIN_CKPT` (default `weights/base/jiaquan_simple/checkpoint-1200.pth`). See `model/models_itpn.py` for the weight-loading logic.
- Data follows the ImageFolder layout under `../../dataset/classification/` (see `dataset/README.md`).

## Requirements

`torch`, `torchvision`, `numpy`, `opencv-python`, `faiss-cpu` (k-NN). Scripts can be run from this directory (`utils` / `model` packages are local; no extra `PYTHONPATH` needed).
