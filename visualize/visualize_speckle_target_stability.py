"""
Compare speckle perturbation stability of different supervision targets
(pixel vs single-scale SAR features vs multi-scale fused), aligned with
models/masked_autoencoder.py (My_SAR_feature + patchify target).

Metrics (mean L1 over spatial/patch dimensions, averaged over image pairs):
  S_pixel  = E || patchify(x^(a)) - patchify(x^(b)) ||_1
  S_k      = E || patchify(y_k(x^(a))) - patchify(y_k(x^(b))) ||_1
  S_multi  = E || patchify(y_fused(x^(a))) - patchify(y_fused(x^(b))) ||_1

Expectation trend (paper-style): S_multi < S_k < S_pixel (on average).

Multi-scale fusion weights are loaded from checkpoint (e.g. hogs.simple_weights).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 脚本在 visualize/ 下运行时，需把项目根目录加入 path，才能 import models.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.masked_autoencoder import My_SAR_feature
from plot_speckle_stability_figure import plot_stability_lines

NAMES = ["pixel"] + [f"S{k}" for k in range(1, 7)] + ["multi"]


def resolve_pretrain_ckpt() -> str:
    """解析预训练权重路径 (resolve the pretrained checkpoint path)。

    优先使用环境变量 SARATRX_PRETRAIN_CKPT；否则默认指向仓库
    weights/jiaquan_simple/checkpoint-1200.pth。
    (env var first, otherwise <repo>/weights/jiaquan_simple/checkpoint-1200.pth)
    """
    ckpt = os.environ.get("SARATRX_PRETRAIN_CKPT", "").strip()
    if ckpt:
        return ckpt
    default = _REPO_ROOT / "weights" / "jiaquan_simple" / "checkpoint-1200.pth"
    return str(default)

# masked_autoencoder.SAR_Layer4 使用 k=17 的 reflect pad，要求 H,W > 17；训练常用 224。
# 裁剪后若过小（如 16×16）会触发 RuntimeError。
DEFAULT_MIN_SPATIAL_FOR_SAR = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speckle stability: pixel vs SAR scales vs fused target.")
    p.add_argument(
        "--data_path",
        type=str,
        default="dataset/pre-training",
        help="Folder of SAR images (recursive walk) or single image path.",
    )
    p.add_argument("--max_images", type=int, default=64, help="Max images to use from folder.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=resolve_pretrain_ckpt(),
        help="Pretrained checkpoint containing hogs.* (My_SAR_feature) weights.",
    )
    p.add_argument("--patch_size", type=int, default=16, help="Patch size for patchify (match MAE target).")
    p.add_argument("--num_realizations", type=int, default=5, help="Number K of independent speckle versions per image.")
    p.add_argument(
        "--num_pair_samples",
        type=int,
        default=0,
        help="Random pair samples per image (0 = use all C(K,2) unordered pairs).",
    )
    p.add_argument(
        "--speckle_mode",
        type=str,
        default="lognormal",
        choices=("lognormal", "gamma"),
        help="Multiplicative speckle model.",
    )
    p.add_argument(
        "--speckle_std",
        type=float,
        default=0.15,
        help="Deprecated if --speckle_std_list is set; single lognormal strength.",
    )
    p.add_argument(
        "--speckle_std_list",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25",
        help="Comma-separated σ for lognormal; one curve point per value (perturbation sweep).",
    )
    p.add_argument(
        "--speckle_L",
        type=float,
        default=4.0,
        help="Single gamma L if --speckle_L_list is empty.",
    )
    p.add_argument(
        "--speckle_L_list",
        type=str,
        default="",
        help="Comma-separated L for gamma mode (e.g. 8,6,4,3,2). Smaller L => stronger speckle. Empty: use --speckle_L only.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./visualize/speckle_stability_out")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--min_spatial",
        type=int,
        default=DEFAULT_MIN_SPATIAL_FOR_SAR,
        help="若裁剪后短边小于此值，则按比例放大，使 SAR_Layer(k=17) 的 reflect pad 合法；且宽高仍为 patch 的倍数。",
    )
    return p.parse_args()


def _extract_state_dict_from_checkpoint(ckpt_obj: object) -> dict:
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: cannot find state dict.")


def load_pretrained_hogs(model: My_SAR_feature, pretrained_path: str, strict: bool = False) -> None:
    if not pretrained_path or not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")
    ckpt_obj = torch.load(pretrained_path, map_location="cpu")
    raw = _extract_state_dict_from_checkpoint(ckpt_obj)
    model_state = model.state_dict()
    filtered = {}
    for k, v in raw.items():
        ck = k
        if ck.startswith("module."):
            ck = ck[len("module.") :]
        if ck.startswith("hogs."):
            ck = ck[len("hogs.") :]
        if ck in model_state and model_state[ck].shape == v.shape:
            filtered[ck] = v
    if not filtered:
        raise RuntimeError("No hogs.* keys matched My_SAR_feature. Check checkpoint.")
    msg = model.load_state_dict(filtered, strict=strict)
    print(f"[load] {pretrained_path} | matched={len(filtered)} missing={len(msg.missing_keys)}")


def collect_image_paths(data_path: str, max_images: int) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths: List[str] = []
    if os.path.isfile(data_path):
        return [data_path]
    if not os.path.isdir(data_path):
        raise FileNotFoundError(data_path)
    for root, _, files in os.walk(data_path):
        for f in sorted(files):
            if len(paths) >= max_images:
                return paths
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return paths


def read_sar_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def center_crop_to_patch(img: np.ndarray, patch: int) -> np.ndarray:
    h, w = img.shape
    nh = (h // patch) * patch
    nw = (w // patch) * patch
    if nh < patch or nw < patch:
        raise ValueError("Image too small.")
    t = (h - nh) // 2
    l = (w - nw) // 2
    return img[t : t + nh, l : l + nw]


def upsample_if_too_small_for_sar(img: np.ndarray, patch: int, min_side: int) -> np.ndarray:
    """短边 < min_side 时双线性放大，并保证 H、W 为 patch 的整数倍。"""
    h, w = img.shape
    if min(h, w) >= min_side:
        return img
    scale = min_side / float(min(h, w))
    nh = max(int(np.ceil(h * scale / patch)) * patch, patch)
    nw = max(int(np.ceil(w * scale / patch)) * patch, patch)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def patchify_1ch(x: torch.Tensor, p: int) -> torch.Tensor:
    """x: [B,1,H,W] -> [B, L, p*p] 与 MAE patchify 一致；H、W 可不等（L=(H/p)*(W/p)）。"""
    b, c, h, w = x.shape
    assert c == 1 and h % p == 0 and w % p == 0
    hh = h // p
    ww = w // p
    x = x.reshape(b, c, hh, p, ww, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(b, hh * ww, p * p * c)


def speckle_lognormal(x: torch.Tensor, std: float, g: torch.Generator) -> torch.Tensor:
    eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
    n = torch.exp(std * eps)
    return (x * n).clamp(0.0, 1.0)


def speckle_gamma(x: torch.Tensor, L: float, g: torch.Generator) -> torch.Tensor:
    conc = torch.tensor(L, device=x.device, dtype=x.dtype)
    dist = torch.distributions.Gamma(concentration=conc, rate=conc)
    n = dist.rsample(x.shape)  # mean 1
    return (x * n).clamp(0.0, 1.0)


@torch.no_grad()
def forward_scales(model: My_SAR_feature, x01: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """x01: [B,1,H,W] in [0,1]. Returns list of 6 maps [B,1,H,W], fused [B,1,H,W], softmax weights [6]."""
    y1 = model.SAR_Lay(x01)
    y2 = model.SAR_Layer0(x01)
    y3 = model.SAR_Layer1(x01)
    y4 = model.SAR_Layer2(x01)
    y5 = model.SAR_Layer3(x01)
    y6 = model.SAR_Layer4(x01)
    w = F.softmax(model.simple_weights, dim=0)
    fused = w[0] * y1 + w[1] * y2 + w[2] * y3 + w[3] * y4 + w[4] * y5 + w[5] * y6
    return [y1, y2, y3, y4, y5, y6], fused, w


def mean_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean L1 over all elements (per batch), return [B]."""
    return (a - b).abs().flatten(1).mean(dim=1)


def sample_pairs(K: int, num_samples: int, rng: random.Random) -> List[Tuple[int, int]]:
    pairs = []
    if num_samples <= 0:
        for i in range(K):
            for j in range(i + 1, K):
                pairs.append((i, j))
        return pairs
    seen = set()
    while len(pairs) < num_samples:
        i, j = rng.randint(0, K - 1), rng.randint(0, K - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
    return pairs


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def compute_mean_stability(
    paths: List[str],
    model: My_SAR_feature,
    device: torch.device,
    patch_size: int,
    min_spatial: int,
    K: int,
    num_pair_samples: int,
    speckle_mode: str,
    speckle_std: float,
    speckle_L: float,
    run_seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """返回 (各目标在图上的平均 mean L1, 每张图的行列表)。"""
    rng_py = random.Random(run_seed)
    names = NAMES
    sum_metrics = {n: 0.0 for n in names}
    count_metrics = 0
    per_image_rows: List[Dict[str, float]] = []

    for ip, path in enumerate(paths):
        img = center_crop_to_patch(read_sar_gray(path), patch_size)
        img = upsample_if_too_small_for_sar(img, patch_size, min_spatial)
        x0 = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        gens = [
            torch.Generator(device=device).manual_seed(run_seed + 10007 * ip + r) for r in range(K)
        ]
        xs = []
        for r in range(K):
            g = gens[r]
            if speckle_mode == "lognormal":
                xs.append(speckle_lognormal(x0, speckle_std, g))
            else:
                xs.append(speckle_gamma(x0, speckle_L, g))

        pairs = sample_pairs(K, num_pair_samples, rng_py)
        acc = {n: 0.0 for n in names}
        n_pairs = 0
        for ia, ib in pairs:
            xa, xb = xs[ia], xs[ib]
            tp_a = patchify_1ch(xa, patch_size)
            tp_b = patchify_1ch(xb, patch_size)
            acc["pixel"] += float(mean_l1(tp_a, tp_b).mean().item())

            scales_a, fused_a, _ = forward_scales(model, xa)
            scales_b, fused_b, _ = forward_scales(model, xb)
            for k in range(6):
                pa = patchify_1ch(scales_a[k], patch_size)
                pb = patchify_1ch(scales_b[k], patch_size)
                acc[f"S{k + 1}"] += float(mean_l1(pa, pb).mean().item())
            pm_a = patchify_1ch(fused_a, patch_size)
            pm_b = patchify_1ch(fused_b, patch_size)
            acc["multi"] += float(mean_l1(pm_a, pm_b).mean().item())
            n_pairs += 1

        row: Dict[str, float] = {"path": path}
        for n in names:
            v = acc[n] / max(n_pairs, 1)
            row[n] = v
            sum_metrics[n] += v
        count_metrics += 1
        per_image_rows.append(row)

    mean_all = {n: sum_metrics[n] / max(count_metrics, 1) for n in names}
    return mean_all, per_image_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.data_path:
        raise SystemExit("Please set --data_path to a SAR image or folder.")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = collect_image_paths(args.data_path, args.max_images)
    if not paths:
        raise SystemExit("No images found.")

    model = My_SAR_feature(kensize=1).to(device)
    model.eval()
    load_pretrained_hogs(model, args.checkpoint, strict=False)
    fusion_w = F.softmax(model.simple_weights, dim=0).detach().cpu().tolist()
    print("Fusion weights (softmax simple_weights):", [round(x, 6) for x in fusion_w])

    p = args.patch_size
    K = args.num_realizations

    if args.speckle_mode == "lognormal":
        levels = parse_float_list(args.speckle_std_list)
        if not levels:
            levels = [args.speckle_std]
        xlabel = r"Speckle strength $\sigma$ (log-normal multiplicative)"
        x_tick_labels = [f"{v:g}" for v in levels]
        param_key = "speckle_std"
    else:
        if args.speckle_L_list.strip():
            levels = parse_float_list(args.speckle_L_list)
        else:
            levels = [args.speckle_L]
        if not levels:
            levels = [args.speckle_L]
        xlabel = r"Effective looks $L$ (Gamma noise, mean$=$1; smaller $L$ $\Rightarrow$ stronger speckle)"
        x_tick_labels = [f"{v:g}" for v in levels]
        param_key = "speckle_L"

    by_condition: List[Dict] = []
    series: Dict[str, List[float]] = {n: [] for n in NAMES}
    last_per_image: List[Dict[str, float]] = []

    for ci, param in enumerate(levels):
        run_seed = args.seed + ci * 1_000_003
        if args.speckle_mode == "lognormal":
            std, L = param, args.speckle_L
        else:
            std, L = args.speckle_std, param

        mean_all, per_image = compute_mean_stability(
            paths,
            model,
            device,
            p,
            args.min_spatial,
            K,
            args.num_pair_samples,
            args.speckle_mode,
            std,
            L,
            run_seed,
        )
        for n in NAMES:
            series[n].append(mean_all[n])
        by_condition.append({param_key: param, "mean_over_images": mean_all})
        last_per_image = per_image

    # 按条件汇总的 CSV
    cond_csv = os.path.join(args.output_dir, "speckle_stability_per_condition.csv")
    with open(cond_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [param_key] + NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_condition:
            out_row = {param_key: row[param_key]}
            out_row.update(row["mean_over_images"])
            writer.writerow(out_row)

    # 最后一档扰动下的逐图明细（便于复查）
    img_csv = os.path.join(args.output_dir, "speckle_stability_per_image_last_condition.csv")
    with open(img_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path"] + NAMES)
        writer.writeheader()
        for row in last_per_image:
            writer.writerow(row)

    json_path = os.path.join(args.output_dir, "speckle_stability_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "by_condition": by_condition,
                "num_images": len(paths),
                "num_realizations_K": K,
                "speckle_mode": args.speckle_mode,
                "speckle_std_list": levels if args.speckle_mode == "lognormal" else None,
                "speckle_L_list": levels if args.speckle_mode == "gamma" else None,
                "patch_size": p,
                "min_spatial": args.min_spatial,
                "checkpoint": args.checkpoint,
                "fusion_weights": fusion_w,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    title_suffix = (
        rf"($K$={K}, patch={p}, mode={args.speckle_mode})"
    )
    fig_path = os.path.join(args.output_dir, "speckle_stability_lines.png")
    plot_stability_lines(
        levels,
        x_tick_labels,
        xlabel,
        series,
        fig_path,
        title="Target stability under speckle perturbation " + title_suffix,
        dpi=float(args.dpi),
    )

    print("By condition:", json.dumps(by_condition, indent=2, ensure_ascii=False))
    print(f"Saved: {cond_csv}")
    print(f"Saved: {img_csv}")
    print(f"Saved: {json_path}")
    print(f"Saved: {fig_path}")

    # 在最强扰动档上打印趋势（列表最后一项）
    mean_last = by_condition[-1]["mean_over_images"]
    sp = mean_last["pixel"]
    sm = mean_last["multi"]
    worst_s = max(mean_last[f"S{k}"] for k in range(1, 7))
    best_s = min(mean_last[f"S{k}"] for k in range(1, 7))
    print(
        f"[last perturbation] S_multi={sm:.6f} vs S_pixel={sp:.6f} | "
        f"best single S={best_s:.6f} worst single S={worst_s:.6f}"
    )
    if sm < sp:
        print("Trend OK: S_multi < S_pixel (last condition).")
    else:
        print("Note: S_multi >= S_pixel on last condition.")


if __name__ == "__main__":
    main()
