# -*- coding: utf-8 -*-
"""
SOC_50classes linear probe for target-ablation pretrain weights (Windows/Linux).

Equivalent to:
  bash scripts/MIM_linear_itpn/run_target_ablation.sh 10

Set checkpoint via env ITPN_PRETRAIN_CKPT (see trainers/MIM_linear_itpn.py).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 消融权重根目录：默认 <repo>/weights/target_ablation/<mode>/checkpoint-49.pth，
# 可用环境变量 TARGET_WEIGHT_ROOT 覆盖。
# (ablation weight root: default <repo>/weights/target_ablation/<mode>/checkpoint-49.pth,
#  overridable via the TARGET_WEIGHT_ROOT environment variable)
WEIGHT_ROOT = Path(os.environ.get("TARGET_WEIGHT_ROOT", str(ROOT.parents[1] / "weights" / "target_ablation")))
CKPT_NAME = "checkpoint-49.pth"
TARGET_MODES = [
    "pixel",
    "single_s1", "single_s2", "single_s3", "single_s4", "single_s5", "single_s6",
    "multi",
]


def parse_accuracy(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\* accuracy:\s*([\d.]+)%", text)
    return float(matches[-1]) if matches else None


def run_one(mode: str, seed: int, shots: int, dataset: str, cfg: str, dry_run: bool) -> Path | None:
    ckpt = WEIGHT_ROOT / mode / CKPT_NAME
    # Use forward slashes: dassl transforms.py splits OUTPUT_DIR by '/'
    out_rel = f"output/{dataset}/MIM_linear_itpn/{mode}_{cfg}_{shots}shots/seed{seed}"
    out_dir = ROOT / out_rel.replace("/", os.sep)
    if not ckpt.is_file():
        print(f"[skip] missing checkpoint: {ckpt}")
        return None
    if out_dir.is_dir() and parse_accuracy(out_dir / "log.txt") is not None:
        print(f"[skip] exists: {out_dir}")
        return out_dir

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--root", str(ROOT.parents[1] / "dataset" / "classification"),
        "--seed", str(seed),
        "--trainer", "MIM_linear_itpn",
        "--dataset-config-file", f"configs/datasets/{dataset}.yaml",
        "--config-file", f"configs/trainers/MIM_linear_itpn/{cfg}.yaml",
        "--output-dir", out_rel,
        "DATASET.NUM_SHOTS", str(shots),
    ]
    print("\n" + "=" * 60)
    print(f"mode={mode} seed={seed} shots={shots}")
    print(f"SARATRX_PRETRAIN_CKPT={ckpt}")
    print(" ".join(cmd))
    if dry_run:
        return out_dir

    env = os.environ.copy()
    env["SARATRX_PRETRAIN_CKPT"] = str(ckpt)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Run SOC linear probe for target ablation weights.")
    p.add_argument("--dataset", default="SOC_50classes")
    p.add_argument("--cfg", default="vit_b16")
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--only", default="", help="single mode, e.g. multi or single_s3")
    p.add_argument("--seed", type=int, default=-1, help="single seed; default all 0..9")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    modes = [args.only] if args.only else TARGET_MODES
    if args.only and args.only not in TARGET_MODES:
        raise ValueError(f"--only must be one of {TARGET_MODES}")
    seeds = [args.seed] if args.seed >= 0 else list(range(10))

    for mode in modes:
        for seed in seeds:
            run_one(mode, seed, args.shots, args.dataset, args.cfg, args.dry_run)

    if not args.dry_run:
        print("\nSummarize:")
        print("  python scripts/MIM_linear_itpn/collect_target_ablation_results.py")


if __name__ == "__main__":
    main()
