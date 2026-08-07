"""
图 2(a)：与论文符号一致——输入 \(x\)，第 \(S\) 个尺度 SAR 特征 \(f_S(\cdot)\) 的输出图、
融合监督目标 \(y=\sum_s \alpha_s f_s\)（与实现中 softmax 权重一致）、逐像素 \(|y-f_S|\)。
（表中 \(r\) 表示掩码率，此处残差不记作 \(r\)，避免混淆。）

每行一个样本，四列：\(x\)（灰度）| \(f_S\)（viridis）| \(y\)（viridis）| \(|y-f_S|\)（magma）。
首行列标题为「英文说明 + 括号内符号」。无总标题。

依赖 My_SAR_feature 与 checkpoint 的加载方式同 visualize_sar_targets.py。

示例（从目录随机 5 张，默认 scan_dir 或显式）::
  python visualize/visualize_fusion_residual.py
  python visualize/visualize_fusion_residual.py --images "dataset/pre-training"
  python visualize/visualize_fusion_residual.py --num_random 5 --seed 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

_IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# 同目录的 visualize_sar_targets 需加入 path；其内部会把项目根目录加入 path
_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import cv2
import matplotlib.pyplot as plt
import numpy as np

import visualize_sar_targets as vsar

# 与 plot_speckle_stability_figure.py 一致的论文字体
FONT_RC = {
    "font.family": "Times New Roman",
    "font.sans-serif": "Times New Roman",
    "font.size": 9.5,
    "mathtext.fontset": "stix",
    "font.serif": "Times New Roman",
}

# 与 visualize_sar_targets.py 中多尺度 / 融合特征一致的伪彩
_CMAP_LAST_AND_FUSE = "viridis"
_CMAP_RESIDUAL = "magma"

# constrained_layout 间距（仅此一处改即可；子图与画布边 w_pad/h_pad 为英寸，子图之间 wspace/hspace 为相对尺寸）
_FUSION_RESIDUAL_W_PAD_IN = 0.017
_FUSION_RESIDUAL_H_PAD_IN = 0.017
_FUSION_RESIDUAL_WSPACE = 0.012
_FUSION_RESIDUAL_HSPACE = 0.012


def read_preprocess(
    path: str, patch: int, min_side: int
) -> np.ndarray:
    """[0,1] 灰度，短边过小则放大，再裁成 patch 整数倍（与训练可视化一致）。"""
    img = vsar.read_sar_image(path)
    img = _upsample_if_too_small(img, patch, min_side)
    return vsar.crop_to_patch_multiple(img, patch)


def _upsample_if_too_small(img: np.ndarray, patch: int, min_side: int) -> np.ndarray:
    h, w = img.shape
    if min(h, w) >= min_side:
        return img
    scale = min_side / float(min(h, w))
    nh = max(int(np.ceil(h * scale / patch)) * patch, patch)
    nw = max(int(np.ceil(w * scale / patch)) * patch, patch)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _list_images_under_dir(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, fnames in os.walk(root):
        for fn in fnames:
            if Path(fn).suffix.lower() in _IMAGE_EXTS:
                out.append(os.path.join(dirpath, fn))
    return out


def collect_image_paths(args: argparse.Namespace) -> List[str]:
    """--image 单张优先；否则 --images 为若干文件或**一个目录**（目录内随机 --num_random 张）；均未给则用 --scan_dir 随机抽。"""
    rng = np.random.default_rng(args.seed)
    n = int(args.num_random)

    if args.image and os.path.isfile(args.image):
        return [args.image]

    if args.images:
        raw = [p for p in args.images if p]
    else:
        raw = [args.scan_dir]

    files = [p for p in raw if os.path.isfile(p)]
    dirs = [p for p in raw if os.path.isdir(p)]
    missing = [p for p in raw if p not in files and p not in dirs]
    if missing:
        raise FileNotFoundError(f"路径不存在: {missing}")

    if dirs and files:
        raise ValueError("请要么只指定若干图像文件，要么只指定一个目录用于随机抽样。")
    if len(dirs) > 1:
        raise ValueError("随机抽样只支持单个目录；请使用 --images 一个文件夹路径。")

    if dirs:
        all_im = _list_images_under_dir(dirs[0])
        if len(all_im) == 0:
            raise FileNotFoundError(f"目录下未找到支持的图像 ({', '.join(sorted(_IMAGE_EXTS))}): {dirs[0]}")
        if len(all_im) < n:
            raise ValueError(f"目录内仅 {len(all_im)} 张图，少于 --num_random={n}。")
        pick = rng.choice(len(all_im), size=n, replace=False)
        return [all_im[i] for i in pick]

    if files:
        return files

    raise ValueError("无有效图像路径。")


def plot_fusion_residual_grid(
    image_paths: List[str],
    pretrained: str,
    strict_load: bool,
    patch_size: int,
    min_spatial: int,
    out_path: str,
    dpi: int,
) -> None:
    n = len(image_paths)
    plt.rcParams.update(FONT_RC)

    # 布局：figsize 为 (4k, nk) 使每格在英寸上近似正方形；constrained_layout 控制「子图外框」间距。
    # 注意：imshow(..., aspect="equal") 会在单格内对非方形图留白（letterbox），那是格内空白，
    # 不是 wspace/hspace；调小 wspace 只能缩小「相邻 axes 边框」距离，肉眼可能仍觉得缝宽。
    # savefig 未使用 bbox_inches="tight" 时，pad_inches 一般不参与布局，不会覆盖上述间距。
    ncols = 4
    cell_in = 1.55
    fig, axes = plt.subplots(
        n,
        ncols,
        figsize=(ncols * cell_in, n * cell_in),
        squeeze=False,
        layout="constrained",
    )

    # 文字说明 + 符号（与论文记号一致）
    col_labels_top = [
        r"Input SAR image $x$",
        r"Last-scale feature $f_{6}$",
        r"Fused target feature $y$",
        r"Residual error $|y - f_{6}|$",
    ]

    for row, ipath in enumerate(image_paths):
        img = read_preprocess(ipath, patch_size, min_spatial)
        scales_np, fused_np, _w = vsar.extract_multiscale_sar_features(
            img, pretrained_path=pretrained, strict_load=strict_load
        )
        f_S = scales_np[-1]
        y_map = fused_np.astype(np.float32)
        abs_diff = np.abs(y_map - f_S.astype(np.float32))

        disp_gray = vsar.normalize_for_display(img)
        disp_f_S = vsar.normalize_for_display(f_S)
        disp_y = vsar.normalize_for_display(fused_np)
        disp_abs_diff = vsar.normalize_for_display(abs_diff)

        panel_spec: List[Tuple[np.ndarray, str]] = [
            (disp_gray, "gray"),
            (disp_f_S, _CMAP_LAST_AND_FUSE),
            (disp_y, _CMAP_LAST_AND_FUSE),
            (disp_abs_diff, _CMAP_RESIDUAL),
        ]

        for col in range(ncols):
            ax = axes[row, col]
            data, cmap_name = panel_spec[col]
            ax.imshow(
                data,
                cmap=cmap_name,
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
                interpolation="nearest",
            )
            ax.axis("off")
            if row == 0:
                ax.set_title(col_labels_top[col], pad=1)

    # 全部 artist 加完后再应用间距并强制算一遍布局（避免仅改 142 行却像「没生效」）
    fig.set_constrained_layout_pads(
        w_pad=_FUSION_RESIDUAL_W_PAD_IN,
        h_pad=_FUSION_RESIDUAL_H_PAD_IN,
        wspace=_FUSION_RESIDUAL_WSPACE,
        hspace=_FUSION_RESIDUAL_HSPACE,
    )
    fig.canvas.draw()

    out_parent = Path(out_path).resolve().parent
    out_parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    pdf_path = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"       {pdf_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fig 2(a): SAR input x, last-scale f_S, fused target y, and |y - f_S|."
    )
    p.add_argument(
        "--image",
        type=str,
        default="",
        help="Single SAR image path.",
    )
    p.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="若干 SAR 图像路径；或**仅一个目录**时从该目录递归随机抽 --num_random 张。省略则使用 --scan_dir。",
    )
    p.add_argument(
        "--scan_dir",
        type=str,
        default="dataset/pre-training",
        help="未指定 --image/--images 时，从此目录递归随机抽样 --num_random 张。",
    )
    p.add_argument(
        "--num_random",
        type=int,
        default=5,
        help="从一个目录抽样时的图像数量。",
    )
    p.add_argument("--seed", type=int, default=4, help="随机抽样种子。")
    p.add_argument(
        "--pretrained",
        type=str,
        default=resolve_pretrain_ckpt(),
        help="Checkpoint with hogs.* / My_SAR_feature weights.",
    )
    p.add_argument("--strict_load", action="store_true", help="strict=True for load_state_dict.")
    p.add_argument("--patch_size", type=int, default=16, help="Crop to multiple of this patch size.")
    p.add_argument(
        "--min_spatial",
        type=int,
        default=64,
        help="If min(H,W) < this, upscale before SAR layers (reflect pad safety).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (PDF same stem). Default: visualize/fusion_residual_out/fig2a_fusion_residual.png",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_image_paths(args)
    print(f"[{len(paths)}] images:")
    for p in paths:
        print(" ", p)

    out = args.output
    if not out:
        out_dir = _VIZ_DIR / "fusion_residual_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / "fig2a_fusion_residual.png")

    plot_fusion_residual_grid(
        image_paths=paths,
        pretrained=args.pretrained,
        strict_load=args.strict_load,
        patch_size=args.patch_size,
        min_spatial=args.min_spatial,
        out_path=out,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
