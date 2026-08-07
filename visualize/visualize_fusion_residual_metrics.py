"""
在「显著目标区域」内度量融合残差集中度，与 visualize_fusion_residual.py 使用相同的
预处理与 \(y^{\mathrm{fuse}}, f^{(K)}\)（最后一层 SAR 特征）定义。

目标区域 \(\mathcal{T}\) 默认不再用人标框（漏标/不准），而是用 **同一套预训练 My_SAR_feature 权重**
得到的显著性图，再二值化（类似显著性检测）：
  - saliency_source：融合响应 fused，或多尺度响应的 mean_abs / max_abs 等；
  - binarize：分位数阈值或 Otsu。

指标（\(\epsilon\) 防止除零）：
  R_res  = sum_{u in T} |y^fuse(u) - f^(K)(u)| / (sum_u |...| + epsilon)
  R_area = |T| / (|T|+|B|)

可选 --target_mode annotations 回退到 COCO JSON 框（需 Annotations/*.json）。

示例::
  python visualize/visualize_fusion_residual_metrics.py --target_mode saliency
  python visualize/visualize_fusion_residual_metrics.py --saliency_source mean_abs_scales --percentile 85
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import visualize_sar_targets as vsar


def _upsample_if_too_small(img: np.ndarray, patch: int, min_side: int) -> np.ndarray:
    h, w = img.shape
    if min(h, w) >= min_side:
        return img
    scale = min_side / float(min(h, w))
    nh = max(int(np.ceil(h * scale / patch)) * patch, patch)
    nw = max(int(np.ceil(w * scale / patch)) * patch, patch)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def preprocess_with_geometry(
    img01: np.ndarray, patch: int, min_side: int
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], Tuple[int, int, int, int]]:
    """
    与 visualize_fusion_residual.read_preprocess 一致。
    返回：最终图、(hr,wr) 上采样后完整尺寸、裁剪起点 (top,left)、(nh,nw) 输出尺寸。
    """
    im = _upsample_if_too_small(img01, patch, min_side)
    hr, wr = im.shape
    h1, w1 = hr, wr
    new_h = (h1 // patch) * patch
    new_w = (w1 // patch) * patch
    top = (h1 - new_h) // 2
    left = (w1 - new_w) // 2
    out = im[top : top + new_h, left : left + new_w]
    return out.astype(np.float32), (hr, wr), (top, left), (new_h, new_w)


def map_coco_boxes_to_preprocessed(
    boxes_xywh: Sequence[Sequence[float]],
    orig_hw: Tuple[int, int],
    after_resize_hw: Tuple[int, int],
    crop_top_left: Tuple[int, int],
    out_hw: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    """COCO [x,y,w,h]（基于读入后的原始尺寸 orig_hw）映射到 preprocess 后像素坐标。"""
    ho, wo = orig_hw
    hr, wr = after_resize_hw
    top, left = crop_top_left
    H, W = out_hw
    sx = wr / float(wo)
    sy = hr / float(ho)

    out: List[Tuple[int, int, int, int]] = []
    for b in boxes_xywh:
        x, y, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        xmin = x * sx - left
        ymin = y * sy - top
        xmax = (x + bw) * sx - left
        ymax = (y + bh) * sy - top
        xi1 = int(np.floor(xmin))
        yi1 = int(np.floor(ymin))
        xi2 = int(np.ceil(xmax))
        yi2 = int(np.ceil(ymax))
        xi1 = max(0, min(W - 1, xi1))
        yi1 = max(0, min(H - 1, yi1))
        xi2 = max(0, min(W, xi2))
        yi2 = max(0, min(H, yi2))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        out.append((xi1, yi1, xi2, yi2))
    return out


def build_target_mask_from_boxes(
    h: int, w: int, boxes: Sequence[Tuple[int, int, int, int]]
) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 1
    return m.astype(np.float32)


def center_square_mask(h: int, w: int, frac_side: float = 0.5) -> np.ndarray:
    """边长为 frac_side * min(h,w) 的居中方形近似目标区。"""
    side = int(round(frac_side * min(h, w)))
    side = max(side, 1)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    m = np.zeros((h, w), dtype=np.float32)
    m[y0 : y0 + side, x0 : x0 + side] = 1.0
    return m


def load_coco_index(annotation_json: str) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    with open(annotation_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    by_id = {im["id"]: im for im in coco["images"]}
    anns_by_im: Dict[int, List[dict]] = {}
    for a in coco["annotations"]:
        iid = a["image_id"]
        anns_by_im.setdefault(iid, []).append(a)
    return by_id, anns_by_im


def build_saliency_map(
    scales_np: List[np.ndarray],
    fused_np: np.ndarray,
    source: str,
) -> np.ndarray:
    """
    使用预训练 SAR 前向响应构造单通道显著性（与残差同尺寸）。
    mean_abs_scales / max_abs_scales：多尺度幅值聚合，与 R_res 中「fuse−最后一层」的对比相对独立。
    fused：融合层响应作显著性（与 y^fuse 直接相关，解释时需注意）。
    """
    if source == "fused":
        return np.abs(fused_np.astype(np.float64))
    stacked = np.stack([np.abs(s.astype(np.float64)) for s in scales_np], axis=0)
    if source == "mean_abs_scales":
        return np.mean(stacked, axis=0)
    if source == "max_abs_scales":
        return np.max(stacked, axis=0)
    raise ValueError(f"Unknown saliency_source: {source}")


def binarize_saliency(
    saliency: np.ndarray,
    method: str,
    percentile: float,
    mean_scale: float,
) -> np.ndarray:
    """返回与 saliency 同尺寸的 0/1 浮点 mask（目标区为 1）。"""
    s = saliency.astype(np.float64)
    flat = s.ravel()
    if method == "percentile":
        thr = float(np.percentile(flat, percentile))
        m = (s >= thr).astype(np.float32)
    elif method == "otsu":
        smin, smax = float(flat.min()), float(flat.max())
        if smax - smin < 1e-12:
            return np.zeros_like(s, dtype=np.float32)
        u8 = ((s - smin) / (smax - smin) * 255.0).astype(np.uint8)
        _, bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (bw > 0).astype(np.float32)
    elif method == "mean_scale":
        mu = float(np.mean(flat))
        m = (s >= mu * mean_scale).astype(np.float32)
    else:
        raise ValueError(f"Unknown binarize method: {method}")

    if np.sum(m > 0.5) == 0:
        thr = float(np.percentile(flat, 90.0))
        m = (s >= thr).astype(np.float32)
    return m


def resolve_pretrained_path(path: str) -> str:
    """允许传入目录：使用该目录下按名称排序后的最后一个 checkpoint*.pth。"""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("checkpoint*.pth"))
        if not cands:
            raise FileNotFoundError(f"目录中未找到 checkpoint*.pth: {path}")
        return str(cands[-1])
    return path


def list_images_in_jpeg_dir(jpeg_dir: Path) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    out: List[str] = []
    for dirpath, _, fnames in os.walk(str(jpeg_dir)):
        for fn in fnames:
            if Path(fn).suffix.lower() in exts:
                out.append(os.path.join(dirpath, fn))
    return out


def compute_residual_concentration(
    residual_abs: np.ndarray,
    target_mask: np.ndarray,
    epsilon: float,
) -> Tuple[float, float]:
    """
    residual_abs, target_mask: 同尺寸 [H,W]，mask 在目标区域为 1。
    返回 (R_res, R_area)。
    """
    r = residual_abs.astype(np.float64)
    t = target_mask.astype(np.float64)
    num = float(np.sum(r * t))
    den = float(np.sum(r) + epsilon)
    r_res = num / den
    nt = float(np.sum(t > 0.5))
    tot = float(t.size)
    r_area = nt / tot if tot > 0 else 0.0
    return r_res, r_area


def monte_carlo_random_mask_r_res(
    residual_abs: np.ndarray,
    k_pixels: int,
    n_trials: int,
    rng: np.random.Generator,
    epsilon: float,
) -> Tuple[float, float]:
    """
    在整幅图上随机重复抽取与显著掩膜同像素数 k 的子集，计算 R_res 的蒙特卡洛均值。
    理论上若残差与位置近似独立，均值 ≈ k/N（与 R_area 同阶），用于对照「显著掩膜是否优于瞎蒙」。
    """
    r = residual_abs.astype(np.float64).ravel()
    den = float(np.sum(r) + epsilon)
    n_pix = r.size
    if k_pixels <= 0 or k_pixels >= n_pix or n_trials <= 0:
        return float("nan"), float("nan")
    vals = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        idx = rng.permutation(n_pix)[:k_pixels]
        vals[t] = float(np.sum(r[idx]) / den)
    return float(np.mean(vals)), float(np.std(vals))


def pearson_residual_saliency(residual_abs: np.ndarray, saliency: np.ndarray) -> float:
    """逐像素 Pearson：残差强度是否与显著性连续值同向共变（不依赖二值阈值）。"""
    a = residual_abs.astype(np.float64).ravel()
    b = saliency.astype(np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    c = np.corrcoef(a, b)[0, 1]
    return float(c)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="64 图（可配置）平均：目标区域残差集中度 R_res vs 面积占比 R_area。"
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/pre-training",
        help="含 JPEGImages 的根目录；annotations 模式还需 Annotations/*.json。",
    )
    p.add_argument(
        "--target_mode",
        type=str,
        choices=("saliency", "annotations"),
        default="saliency",
        help="saliency：预训练 SAR 显著性图二值化作为 T；annotations：COCO 框。",
    )
    p.add_argument(
        "--saliency_source",
        type=str,
        choices=("mean_abs_scales", "max_abs_scales", "fused"),
        default="mean_abs_scales",
        help="显著性强度图定义（默认多尺度平均幅值，减轻与 y^fuse 的耦合）。",
    )
    p.add_argument(
        "--binarize",
        type=str,
        choices=("percentile", "otsu", "mean_scale"),
        default="percentile",
        help="显著性二值化方式。",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=80.0,
        help="binarize=percentile 时：阈值取该分位数，≥ 阈值的像素为显著（约 top (100-p)%%）。",
    )
    p.add_argument(
        "--mean_scale",
        type=float,
        default=1.15,
        help="binarize=mean_scale 时：阈值 = mean(saliency) * 该系数。",
    )
    p.add_argument(
        "--annotation_json",
        type=str,
        default="train.json",
        help="target_mode=annotations 时：Annotations 目录下 JSON 文件名（COCO）。",
    )
    p.add_argument("--num_images", type=int, default=64, help="随机抽样图像数。")
    p.add_argument("--seed", type=int, default=0, help="随机种子。")
    p.add_argument(
        "--pretrained",
        type=str,
        default="weights/base/jiaquan_simple",
        help="My_SAR_feature 权重 .pth 路径，或含 checkpoint*.pth 的目录（自动取排序最后一个）。",
    )
    p.add_argument("--strict_load", action="store_true", help="strict=True 加载 checkpoint。")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--min_spatial", type=int, default=64)
    p.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="R_res 分母稳定项。",
    )
    p.add_argument(
        "--center_fallback_frac",
        type=float,
        default=0.5,
        help="无检测框时，用边长为该系数×min(H,W) 的居中方形近似目标区。",
    )
    p.add_argument(
        "--output_txt",
        type=str,
        default="",
        help="保存数值摘要的路径；默认 visualize/fusion_residual_out/residual_concentration_metrics.txt",
    )
    p.add_argument(
        "--random_trials",
        type=int,
        default=64,
        help="每张图：与同面积随机掩膜对照的蒙特卡洛次数；0 表示不做随机基线。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pretrained_ckpt = resolve_pretrained_path(args.pretrained)
    root = Path(args.dataset_root)
    jpeg_dir = root / "JPEGImages"
    if not jpeg_dir.is_dir():
        raise FileNotFoundError(f"未找到 JPEGImages: {jpeg_dir}")

    rng = np.random.default_rng(args.seed)
    rng_mc = np.random.default_rng(int(args.seed) + 90127)
    n = int(args.num_images)

    pick_ids: Optional[np.ndarray] = None
    image_paths: List[str] = []

    if args.target_mode == "annotations":
        ann_path = root / "Annotations" / args.annotation_json
        if not ann_path.is_file():
            raise FileNotFoundError(f"未找到标注文件: {ann_path}")
        by_id, anns_by_im = load_coco_index(str(ann_path))
        ids_with_boxes = [iid for iid, lst in anns_by_im.items() if len(lst) > 0]
        if len(ids_with_boxes) == 0:
            raise RuntimeError("标注文件中没有任何目标的图像。")
        if len(ids_with_boxes) < n:
            raise ValueError(f"含标注图像仅 {len(ids_with_boxes)} 张，少于 --num_images={n}。")
        pick_ids = rng.choice(ids_with_boxes, size=n, replace=False)
        for iid in pick_ids:
            image_paths.append(str(jpeg_dir / by_id[int(iid)]["file_name"]))
    else:
        all_im = list_images_in_jpeg_dir(jpeg_dir)
        if len(all_im) == 0:
            raise FileNotFoundError(f"目录下未找到图像: {jpeg_dir}")
        if len(all_im) < n:
            raise ValueError(f"JPEGImages 内仅 {len(all_im)} 张图，少于 --num_images={n}。")
        pick_idx = rng.choice(len(all_im), size=n, replace=False)
        image_paths = [all_im[i] for i in pick_idx]

    r_res_list: List[float] = []
    r_area_list: List[float] = []
    diff_list: List[float] = []
    fnames: List[str] = []
    mc_mean_list: List[float] = []
    excess_rand_list: List[float] = []
    corr_rs_list: List[float] = []

    for idx, ipath in enumerate(image_paths):
        fnames.append(Path(ipath).name)
        img = vsar.read_sar_image(ipath)
        h_img, w_img = img.shape

        img_proc, (hr, wr), (top, left), (nh, nw) = preprocess_with_geometry(
            img, args.patch_size, args.min_spatial
        )

        scales_np, fused_np, _w = vsar.extract_multiscale_sar_features(
            img_proc,
            pretrained_path=pretrained_ckpt,
            strict_load=args.strict_load,
        )
        f_k = scales_np[-1].astype(np.float32)
        y_fuse = fused_np.astype(np.float32)
        residual = np.abs(y_fuse - f_k)

        sal: Optional[np.ndarray] = None
        if args.target_mode == "saliency":
            sal = build_saliency_map(scales_np, fused_np, source=args.saliency_source)
            tgt = binarize_saliency(
                sal,
                method=args.binarize,
                percentile=args.percentile,
                mean_scale=args.mean_scale,
            )
        else:
            assert pick_ids is not None
            iid = int(pick_ids[idx])
            anns = anns_by_im.get(iid, [])
            boxes_xywh = [a["bbox"] for a in anns if "bbox" in a]
            if len(boxes_xywh) == 0:
                tgt = center_square_mask(nh, nw, frac_side=args.center_fallback_frac)
            else:
                xyxy = map_coco_boxes_to_preprocessed(
                    boxes_xywh,
                    orig_hw=(h_img, w_img),
                    after_resize_hw=(hr, wr),
                    crop_top_left=(top, left),
                    out_hw=(nh, nw),
                )
                if len(xyxy) == 0:
                    tgt = center_square_mask(nh, nw, frac_side=args.center_fallback_frac)
                else:
                    tgt = build_target_mask_from_boxes(nh, nw, xyxy)

        r_res, r_area = compute_residual_concentration(
            residual, tgt, epsilon=args.epsilon
        )
        r_res_list.append(r_res)
        r_area_list.append(r_area)
        diff_list.append(r_res - r_area)

        k_tgt = int(np.sum(tgt > 0.5))
        if args.random_trials > 0 and k_tgt > 0 and k_tgt < nh * nw:
            mc_mean, _mc_std = monte_carlo_random_mask_r_res(
                residual,
                k_tgt,
                args.random_trials,
                rng_mc,
                args.epsilon,
            )
            if not np.isnan(mc_mean):
                mc_mean_list.append(mc_mean)
                excess_rand_list.append(r_res - mc_mean)
            else:
                mc_mean_list.append(float("nan"))
                excess_rand_list.append(float("nan"))
        else:
            mc_mean_list.append(float("nan"))
            excess_rand_list.append(float("nan"))

        if sal is not None:
            corr_rs_list.append(pearson_residual_saliency(residual, sal))
        else:
            corr_rs_list.append(float("nan"))

    mean_res = float(np.mean(r_res_list))
    mean_area = float(np.mean(r_area_list))
    mean_diff = float(np.mean(diff_list))
    frac_gt = float(np.mean(np.array(diff_list) > 0))

    ratio = np.array(r_res_list, dtype=np.float64) / np.maximum(
        np.array(r_area_list, dtype=np.float64), 1e-12
    )
    mean_lift_ratio = float(np.mean(ratio) - 1.0)
    median_lift_ratio = float(np.median(ratio) - 1.0)

    er_valid = np.array(excess_rand_list, dtype=np.float64)
    er_ok = er_valid[np.isfinite(er_valid)]
    mean_excess_rand = float(np.mean(er_ok)) if er_ok.size else float("nan")
    frac_excess_rand = (
        float(np.mean(er_ok > 0)) if er_ok.size else float("nan")
    )

    cr = np.array(corr_rs_list, dtype=np.float64)
    cr_ok = cr[np.isfinite(cr)]
    mean_corr = float(np.mean(cr_ok)) if cr_ok.size else float("nan")

    lines = [
        f"dataset_root: {root}",
        f"target_mode: {args.target_mode}",
        f"pretrained: {pretrained_ckpt}",
    ]
    if args.target_mode == "saliency":
        lines.extend(
            [
                f"saliency_source: {args.saliency_source}",
                f"binarize: {args.binarize}  "
                f"(percentile={args.percentile}, mean_scale={args.mean_scale})",
            ]
        )
    else:
        lines.append(f"annotation_json: {args.annotation_json}")
    lines.extend(
        [
            f"num_images: {n}  seed: {args.seed}",
            f"epsilon: {args.epsilon}",
            f"random_trials (MC vs same-area random mask): {args.random_trials}",
            "",
            "Why R_area ~ constant with percentile binarize:",
            "  binarize=percentile 时阈值固定在分位数 p，显著像素比例≈(100-p)%%，故 R_area 被钉死",
            "  （你看到的 ~0.200）。此时 R_res≈R_area 并不代表「无现象」，而是「相对该面积的偏离」很小。",
            "  更有对照意义：① relative lift = mean(R_res/R_area)-1；② 随机掩膜基线；③ 连续相关 rho。",
            "",
            "Relative lift R_res/R_area - 1 (mean / median over images):",
            f"  mean:   {mean_lift_ratio * 100:.4f} %%",
            f"  median: {median_lift_ratio * 100:.4f} %%",
            "",
            "Per-image R_res, R_area mean ± std:",
        ]
    )
    lines.extend(
        [
            f"  R_res:   {mean_res:.6f} ± {float(np.std(r_res_list)):.6f}",
            f"  R_area:  {mean_area:.6f} ± {float(np.std(r_area_list)):.6f}",
            f"  R_res - R_area (mean): {mean_diff:.6f}",
            f"  fraction(R_res > R_area): {frac_gt:.4f}",
            "",
            "Monte Carlo: R_res vs random mask with SAME pixel count as salient mask:",
            f"  mean(R_res - R_random): {mean_excess_rand:.6f}",
            f"  fraction(R_res > R_random): {frac_excess_rand:.4f}",
            "",
            "Pearson rho(|y^fuse-f^(K)|, saliency continuous map), saliency mode only:",
            f"  mean rho: {mean_corr:.6f}",
            "",
            "Suggested settings for clearer area contrast: --binarize otsu or mean_scale;",
            "or lower --percentile so salient area is not ~fixed 20%%.",
            "",
            "Interpretation: residual mass alignment with saliency is summarized by lift vs area,",
            "excess vs random same-area baseline, and Pearson correlation.",
        ]
    )
    text = "\n".join(lines)
    print(text)

    out_txt = args.output_txt
    if not out_txt:
        out_dir = _VIZ_DIR / "fusion_residual_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_txt = str(out_dir / "residual_concentration_metrics.txt")
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\nPer-image values:\n")
        for i in range(n):
            if args.target_mode == "annotations" and pick_ids is not None:
                head = f"  image_id={int(pick_ids[i])}  file={fnames[i]}  "
            else:
                head = f"  file={fnames[i]}  "
            extra = ""
            if np.isfinite(mc_mean_list[i]):
                extra += f"  R_rand_mc={mc_mean_list[i]:.8f}  excess_vs_rand={excess_rand_list[i]:.8f}"
            if np.isfinite(corr_rs_list[i]):
                extra += f"  rho={corr_rs_list[i]:.6f}"
            f.write(
                f"{head}"
                f"R_res={r_res_list[i]:.8f}  R_area={r_area_list[i]:.8f}  diff_area={diff_list[i]:.8f}"
                f"{extra}\n"
            )
    print(f"\nSaved summary: {out_txt}")


if __name__ == "__main__":
    main()
