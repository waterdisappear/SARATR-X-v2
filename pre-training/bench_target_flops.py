#!/usr/bin/env python3
"""Hardware-independent approximate FLOPs for SARATR-X-v2 structural targets.

Counts dense conv2d MACs as implemented in models/masked_autoencoder.py
(plus cheap fusion MACs for ``multi``). No GPU required.

Canonical size: H = W = 224 (paper pre-training resolution), batch = 1.
"""
from __future__ import annotations

import json
from pathlib import Path


H = W = 224
N = H * W  # spatial locations after valid conv on reflect-padded input

# Radii as in My_SAR_feature: S1 uses SAR_Lay(k=1); S2..S6 use SAR_Layer(r)
RADII = {
    "S1": 1,
    "S2": 3,
    "S3": 5,
    "S4": 9,
    "S5": 13,
    "S6": 17,
}


def flops_s1() -> int:
    """One 3x3 convolution (9 taps) over N locations."""
    return N * 9


def flops_sk(radius: int) -> int:
    """Four MxM convolutions, M = 2r+1 (left/right/up/down half-region filters)."""
    m = 2 * radius + 1
    return 4 * N * (m * m)


def flops_fuse() -> int:
    """Six scalar-map products + five adds ≈ 11N."""
    return 11 * N


def branch_flops() -> dict[str, int]:
    out = {"S1": flops_s1()}
    for name, r in RADII.items():
        if name == "S1":
            continue
        out[name] = flops_sk(r)
    return out


def mode_flops(branches: dict[str, int]) -> dict[str, int]:
    multi = sum(branches.values()) + flops_fuse()
    return {
        "pixel": 0,
        "single_s1": branches["S1"],
        "single_s6": branches["S6"],
        "multi": multi,
    }


def fmt_gmacs(macs: int) -> str:
    return f"{macs / 1e9:.4f}"


def main() -> None:
    branches = branch_flops()
    modes = mode_flops(branches)
    s1 = modes["single_s1"]
    multi = modes["multi"]

    rows = []
    for mode, macs in modes.items():
        rows.append(
            {
                "mode": mode,
                "macs": macs,
                "gmacs": round(macs / 1e9, 6),
                "rel_to_single_s1": None if s1 == 0 else round(macs / s1, 2),
                "rel_to_multi": round(macs / multi, 4) if multi else None,
                "pct_of_multi": round(100.0 * macs / multi, 2) if multi else None,
            }
        )

    payload = {
        "input_hw": [H, W],
        "spatial_locations": N,
        "counting_rule": (
            "Dense conv2d MACs as implemented (F.conv2d); "
            "log/sigmoid/pad omitted; multi adds 11N fusion MACs."
        ),
        "branch_macs": branches,
        "modes": rows,
    }

    out_path = Path(__file__).with_name("bench_target_flops.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Input: {H}x{W}, N={N}")
    print(f"Counting: {payload['counting_rule']}")
    print()
    print(f"{'Branch':<8}{'MACs':>16}{'GMACs':>10}{'% of multi':>12}")
    for name, macs in branches.items():
        pct = 100.0 * macs / multi
        print(f"{name:<8}{macs:>16,}{fmt_gmacs(macs):>10}{pct:>11.1f}%")
    fuse = flops_fuse()
    print(f"{'fusion':<8}{fuse:>16,}{fmt_gmacs(fuse):>10}{100.0 * fuse / multi:>11.1f}%")
    print(f"{'multi':<8}{multi:>16,}{fmt_gmacs(multi):>10}{'100.0%':>12}")
    print()
    print("Reference: one iTPN-B masked-pretrain forward @224 is ~10.9 GMACs; "
          f"multi/iTPN-B ≈ {100.0 * multi / 1e9 / 10.9:.1f}%.")
    print("Mode totals:")
    for row in rows:
        print(f"  {row['mode']:<12} {fmt_gmacs(row['macs'])} GMACs")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
