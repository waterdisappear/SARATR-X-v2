<p align="right">
  <b>English</b> | <a href="./README_zh.md">中文</a>
</p>

# results — Experiment Results

This directory stores the experiment data of SARATR-X-v2. It contains the raw training logs / configs of the **detection and segmentation** downstream experiments (copied from the original run folder `D:\2024_SARatrX_2\result`), plus the small csv/json data needed by the visualization scripts.

> Note: Model checkpoints (`.pth`, 1–4 GB each) are **not committed** — they live under `weights/` and are uploaded separately. The logs here contain the final metrics used in the paper tables.

## Directory Structure

```
results/
├── detection/                   # Detection logs & configs
│   ├── rsar/                    # RSAR (OBB, ReDet / RoI Trans / Oriented R-CNN + iTPN)
│   ├── sardet/                  # SARDet-100K (HBB, GFL + iTPN)
│   ├── ssdd/                    # SSDD (HBB, GFL + iTPN)
│   └── hrsid/                   # HRSID (HBB, GFL + iTPN)
├── segmentation/                # Segmentation logs & configs
│   ├── air_polarsar2/           # AIR-PolSAR-Seg-2.0 (UperNet + iTPN)
│   ├── whu-opt-sar/             # WHU-OPT-SAR (UperNet + iTPN)
│   └── ddhr-sk/                 # DDHR-SK (UperNet + iTPN)
└── visualize/                   # Experiment data for visualization scripts (csv/json)
```

Each downstream folder contains both `base` and `large` (prefix `large_`) log files plus the exact config that produced them. The `*.py` configs here cover the benchmarks that have **no config in `detection/` or `segmentation/`** (SARDet-100K / SSDD / HRSID / WHU-OPT-SAR / DDHR-SK); they follow the MMDetection / MMSegmentation style and can be adapted for reproduction.

## Main Results

### Detection

| Dataset | Protocol | Method | Metric | iTPN-Base | iTPN-Large |
| --- | --- | --- | --- | --- | --- |
| RSAR | OBB | ReDet + iTPN | mAP@50 (test) | 74.2 | 77.3 |
| SARDet-100K | HBB | GFL + iTPN | bbox mAP | 63.4 | 68.7 |
| SSDD | HBB | GFL + iTPN | bbox mAP | 71.0 | 71.8 |
| HRSID | HBB | GFL + iTPN | bbox mAP | 71.5 | 73.5 |

### Segmentation

| Dataset | Method | Metric | iTPN-Base | iTPN-Large |
| --- | --- | --- | --- | --- |
| AIR-PolSAR-Seg-2.0 | UperNet + iTPN | mIoU | 90.8 | 95.3 |
| WHU-OPT-SAR | UperNet + iTPN | mIoU | 46.1 | 47.1 |
| DDHR-SK | UperNet + iTPN | mIoU | 83.4 | 83.6 |

> **Correspondence with the paper.** These numbers are exactly the values reported in the paper (Appendix tables `tab_RSAR`, `tab_SARDet-100K`, `tab_SSDD`, `tab_HRSID`, `tab_airpolsar_amplitude`, `tab_whu_opt_sar`, `tab_ddhr_sk`). SSDD / HRSID and all segmentation logs match the paper exactly; the RSAR row uses the *test* set (the log field `test_mAP50`).

## Contents under visualize/

| File(s) | Description |
| --- | --- |
| `speckle_stability_out/`、`speckle_stability_out_50ep/` | Speckle-stability experiment data (1200-epoch / 50-epoch settings) |
| `stability_transfer_out/` | Stability–transfer scatter data (x: drift, y: 10-shot acc.) |
| `target_ablation_soc_10shot_summary.csv/.json` | 10-shot target-ablation summary on SOC-50 |

## Usage

- Visualization scripts (`visualize/`) read the csv/json under `results/visualize/` directly.
- Downstream detection/segmentation configs are also available under `detection/` and `segmentation/` (iTPN-related parts); the extra configs here cover the remaining paper benchmarks.
