# -*- coding: utf-8 -*-
"""Collect SOC_50classes 10-shot linear probe results for target ablation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = ROOT / "output"
TARGET_MODES = [
    "pixel",
    "single_s1", "single_s2", "single_s3", "single_s4", "single_s5", "single_s6",
    "multi",
]
DISPLAY = {
    "pixel": "Pixel",
    "single_s1": "S1 (3×3)",
    "single_s2": "S2",
    "single_s3": "S3",
    "single_s4": "S4",
    "single_s5": "S5",
    "single_s6": "S6",
    "multi": "Multi (simple)",
}


def parse_log(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    acc = re.findall(r"\* accuracy:\s*([\d.]+)%", text)
    f1 = re.findall(r"\* macro_f1:\s*([\d.]+)%", text)
    out = {}
    if acc:
        out["accuracy"] = float(acc[-1])
    if f1:
        out["macro_f1"] = float(f1[-1])
    return out


def collect(out_root: Path, dataset: str, cfg: str, shots: int, seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = list(range(10))
    summary = {}
    for mode in TARGET_MODES:
        base = out_root / dataset / "MIM_linear_itpn" / f"{mode}_{cfg}_{shots}shots"
        seed_accs = []
        seed_f1s = []
        for seed in seeds:
            log = base / f"seed{seed}" / "log.txt"
            if not log.is_file():
                continue
            m = parse_log(log)
            if "accuracy" in m:
                seed_accs.append(m["accuracy"])
            if "macro_f1" in m:
                seed_f1s.append(m["macro_f1"])
        if not seed_accs:
            continue
        summary[mode] = {
            "display": DISPLAY.get(mode, mode),
            "n_seeds": len(seed_accs),
            "accuracy_mean": sum(seed_accs) / len(seed_accs),
            "accuracy_std": (
                (sum((x - sum(seed_accs) / len(seed_accs)) ** 2 for x in seed_accs) / len(seed_accs)) ** 0.5
                if len(seed_accs) > 1 else 0.0
            ),
            "accuracy_per_seed": seed_accs,
            "macro_f1_mean": sum(seed_f1s) / len(seed_f1s) if seed_f1s else None,
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--dataset", default="SOC_50classes")
    p.add_argument("--cfg", default="vit_b16")
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--save_json", default=str(ROOT / "output" / "target_ablation_soc_10shot_summary.json"))
    args = p.parse_args()

    out_root = Path(args.out_root)
    summary = collect(out_root, args.dataset, args.cfg, args.shots, args.seeds)

    print(f"\nSOC_50classes {args.shots}-shot linear probe (iTPN-Base, mean±std over seeds)\n")
    print(f"{'Mode':<18} {'Acc (%)':>12} {'Macro-F1 (%)':>14} {'Seeds':>6}")
    print("-" * 54)
    for mode in TARGET_MODES:
        if mode not in summary:
            print(f"{DISPLAY.get(mode, mode):<18} {'—':>12} {'—':>14} {'0':>6}")
            continue
        s = summary[mode]
        f1_str = f"{s['macro_f1_mean']:.1f}" if s["macro_f1_mean"] is not None else "—"
        print(
            f"{s['display']:<18} "
            f"{s['accuracy_mean']:.1f}±{s['accuracy_std']:.1f} "
            f"{f1_str:>14} "
            f"{s['n_seeds']:>6}"
        )

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "shots": args.shots,
        "cfg": args.cfg,
        "seeds": args.seeds,
        "results": summary,
    }
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
