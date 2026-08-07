import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# 脚本在 visualize/ 下运行时，需把项目根目录加入 path，才能 import models.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.masked_autoencoder import My_SAR_feature


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SAR image, masked image, multi-scale SAR features, and fused target feature."
    )
    parser.add_argument(
    "--image_path",
    type=str,
    default=r"D:\2023_SARatrX_1\Pre-Train Data\500K\FAIR_CSAR\FSI-TRAINVAL\PNGImages\GF3_MYN_FSI_030002_E121.5_N31.4_20220422_L1A_VV_L10000000008_03072_16011.png",
    help="Path to the input SAR image."
)
    parser.add_argument("--output_dir", type=str, default="./vis_output", help="Directory to save visualizations.")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=resolve_pretrain_ckpt(),
        help="Path to pretrained checkpoint. Supports full checkpoint or state_dict.",
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Use strict=True when loading matched weights.",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Patch mask ratio in [0, 1).")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for random masking.")
    parser.add_argument(
        "--demo_grid_size",
        type=int,
        default=8,
        help="Number of blocks per side for mask visualization (e.g., 8 means 8x8 blocks).",
    )
    parser.add_argument(
        "--grid_patch_size",
        type=int,
        default=12,
        help="Display size (pixels) of each patch block in mask map.",
    )
    parser.add_argument(
        "--grid_gap",
        type=int,
        default=8,
        help="Gap width (pixels) between patch blocks in mask map.",
    )
    parser.add_argument(
        "--mask_fill_value",
        type=float,
        default=0.0,
        help="Pixel value used to fill masked regions, in [0, 1].",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible masking.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure in an interactive window.",
    )
    return parser.parse_args()


def read_sar_image(image_path: str) -> np.ndarray:
    # Windows 上 cv2.imread 对含非 ASCII 的路径常失败；用字节读取 + imdecode 可正确打开中文路径。
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    buf = np.frombuffer(p.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image (decode failed): {image_path}")
    image = image.astype(np.float32) / 255.0
    return image


def crop_to_patch_multiple(image: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = image.shape
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Image is too small for patch_size={patch_size}.")
    if new_h == h and new_w == w:
        return image
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return image[top : top + new_h, left : left + new_w]


def build_random_patch_mask(
    h: int, w: int, patch_size: int, mask_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in [0, 1).")

    rng = np.random.default_rng(seed)
    gh = h // patch_size
    gw = w // patch_size
    num_patches = gh * gw
    num_keep = int(num_patches * (1.0 - mask_ratio))

    ids = np.arange(num_patches)
    rng.shuffle(ids)
    keep_ids = ids[:num_keep]

    patch_mask_flat = np.ones(num_patches, dtype=np.float32)  # 1: masked
    patch_mask_flat[keep_ids] = 0.0  # 0: kept
    patch_mask = patch_mask_flat.reshape(gh, gw)
    pixel_mask = np.repeat(np.repeat(patch_mask, patch_size, axis=0), patch_size, axis=1)
    return patch_mask, pixel_mask


def build_random_block_mask(
    h: int, w: int, grid_size: int, mask_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_size <= 0:
        raise ValueError("demo_grid_size must be > 0.")
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in [0, 1).")

    rng = np.random.default_rng(seed)
    gh = gw = grid_size
    num_blocks = gh * gw
    num_keep = int(num_blocks * (1.0 - mask_ratio))

    ids = np.arange(num_blocks)
    rng.shuffle(ids)
    keep_ids = ids[:num_keep]

    block_mask_flat = np.ones(num_blocks, dtype=np.float32)  # 1: masked
    block_mask_flat[keep_ids] = 0.0  # 0: kept
    block_mask = block_mask_flat.reshape(gh, gw)

    row_edges = np.linspace(0, h, gh + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, gw + 1, dtype=np.int32)
    pixel_mask = np.zeros((h, w), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            pixel_mask[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = block_mask[i, j]

    return block_mask, pixel_mask, row_edges, col_edges


def _extract_state_dict_from_checkpoint(ckpt_obj: object) -> dict:
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: cannot find state dict.")


def load_pretrained_for_sar_feature(
    model: torch.nn.Module, pretrained_path: str, strict_load: bool = False
) -> Tuple[int, int]:
    if not pretrained_path:
        return 0, 0
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    ckpt_obj = torch.load(pretrained_path, map_location="cpu")
    raw_state_dict = _extract_state_dict_from_checkpoint(ckpt_obj)

    model_state = model.state_dict()
    filtered_state = {}
    for k, v in raw_state_dict.items():
        clean_k = k
        if clean_k.startswith("module."):
            clean_k = clean_k[len("module.") :]
        if clean_k.startswith("hogs."):
            clean_k = clean_k[len("hogs.") :]
        if clean_k in model_state and model_state[clean_k].shape == v.shape:
            filtered_state[clean_k] = v

    if len(filtered_state) == 0:
        raise RuntimeError(
            "No matching keys found for My_SAR_feature. "
            "Please check whether checkpoint contains 'hogs.*' or compatible keys."
        )

    msg = model.load_state_dict(filtered_state, strict=strict_load)
    print(f"[Weight Loading] from: {pretrained_path}")
    print(f"[Weight Loading] matched keys: {len(filtered_state)}")
    print(f"[Weight Loading] missing keys: {len(msg.missing_keys)}")
    print(f"[Weight Loading] unexpected keys: {len(msg.unexpected_keys)}")
    return len(msg.missing_keys), len(msg.unexpected_keys)


def extract_multiscale_sar_features(
    image: np.ndarray, pretrained_path: str = "", strict_load: bool = False
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    model = My_SAR_feature(kensize=1).to(device)
    load_pretrained_for_sar_feature(model, pretrained_path=pretrained_path, strict_load=strict_load)
    model.eval()

    with torch.no_grad():
        y1 = model.SAR_Lay(x)
        y2 = model.SAR_Layer0(x)
        y3 = model.SAR_Layer1(x)
        y4 = model.SAR_Layer2(x)
        y5 = model.SAR_Layer3(x)
        y6 = model.SAR_Layer4(x)

        weights = torch.softmax(model.simple_weights, dim=0)
        fused = (
            weights[0] * y1
            + weights[1] * y2
            + weights[2] * y3
            + weights[3] * y4
            + weights[4] * y5
            + weights[5] * y6
        )

    features = [y1, y2, y3, y4, y5, y6]
    features_np = [f.squeeze(0).squeeze(0).cpu().numpy() for f in features]
    fused_np = fused.squeeze(0).squeeze(0).cpu().numpy()
    return features_np, fused_np, weights.cpu().numpy()


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    a_min, a_max = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, a_min, a_max)
    if a_max - a_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a_min) / (a_max - a_min)


def to_patch_grid_view(
    image: np.ndarray, patch_size: int, line_value: float = 0.15, line_width: int = 1
) -> np.ndarray:
    h, w = image.shape
    gh, gw = h // patch_size, w // patch_size
    patch_values = image.reshape(gh, patch_size, gw, patch_size).mean(axis=(1, 3))

    out_h = gh * patch_size + (gh + 1) * line_width
    out_w = gw * patch_size + (gw + 1) * line_width
    canvas = np.full((out_h, out_w), line_value, dtype=np.float32)

    for i in range(gh):
        for j in range(gw):
            y0 = i * (patch_size + line_width) + line_width
            x0 = j * (patch_size + line_width) + line_width
            canvas[y0 : y0 + patch_size, x0 : x0 + patch_size] = patch_values[i, j]
    return canvas


def expand_patch_mask_to_grid(
    patch_mask: np.ndarray, patch_size: int, line_value: float = 0.35, line_width: int = 3
) -> np.ndarray:
    gh, gw = patch_mask.shape
    out_h = gh * patch_size + (gh + 1) * line_width
    out_w = gw * patch_size + (gw + 1) * line_width
    canvas = np.full((out_h, out_w), line_value, dtype=np.float32)

    for i in range(gh):
        for j in range(gw):
            y0 = i * (patch_size + line_width) + line_width
            x0 = j * (patch_size + line_width) + line_width
            # white: masked (1), black: kept (0)
            canvas[y0 : y0 + patch_size, x0 : x0 + patch_size] = patch_mask[i, j]
    return canvas


def masked_image_to_patch_grid(
    blocks: List[List[np.ndarray]],
    patch_mask: np.ndarray,
    gap: int,
    mask_fill_value: float,
    line_value: float = 1.0,
) -> np.ndarray:
    gh, gw = patch_mask.shape

    first_col_heights = [blocks[i][0].shape[0] for i in range(gh)]
    first_row_widths = [blocks[0][j].shape[1] for j in range(gw)]
    out_h = sum(first_col_heights) + (gh + 1) * gap
    out_w = sum(first_row_widths) + (gw + 1) * gap
    canvas = np.full((out_h, out_w), line_value, dtype=np.float32)

    y_cursor = gap
    for i in range(gh):
        x_cursor = gap
        for j in range(gw):
            src_patch = blocks[i][j]
            patch_h, patch_w = src_patch.shape[:2]
            if patch_mask[i, j] > 0.5:
                patch = np.full((patch_h, patch_w), mask_fill_value, dtype=np.float32)
            else:
                patch = src_patch.astype(np.float32)

            canvas[y_cursor : y_cursor + patch_h, x_cursor : x_cursor + patch_w] = patch
            x_cursor += patch_w + gap
        y_cursor += blocks[i][0].shape[0] + gap

    return canvas


def split_image_to_blocks(image: np.ndarray, row_edges: np.ndarray, col_edges: np.ndarray) -> List[List[np.ndarray]]:
    gh = len(row_edges) - 1
    gw = len(col_edges) - 1
    blocks: List[List[np.ndarray]] = []
    for i in range(gh):
        row_blocks: List[np.ndarray] = []
        for j in range(gw):
            block = image[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            row_blocks.append(block)
        blocks.append(row_blocks)
    return blocks


def save_single_image(path: str, image: np.ndarray, cmap: str = "gray") -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image = read_sar_image(args.image_path)
    image = crop_to_patch_multiple(image, args.patch_size)
    h, w = image.shape

    patch_mask, pixel_mask, row_edges, col_edges = build_random_block_mask(
        h=h,
        w=w,
        grid_size=args.demo_grid_size,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    masked_image = image.copy()
    masked_image[pixel_mask > 0.5] = args.mask_fill_value
    image_blocks = split_image_to_blocks(image, row_edges, col_edges)
    mask_grid = expand_patch_mask_to_grid(
        patch_mask,
        patch_size=args.grid_patch_size,
        line_value=1.0,
        line_width=args.grid_gap,
    )
    masked_grid = masked_image_to_patch_grid(
        blocks=image_blocks,
        patch_mask=patch_mask,
        gap=args.grid_gap,
        mask_fill_value=args.mask_fill_value,
        line_value=1.0,
    )

    features, fused, weights = extract_multiscale_sar_features(
        image, pretrained_path=args.pretrained, strict_load=args.strict_load
    )

    vis_items = [
        ("Original SAR", normalize_for_display(image), "gray"),
        ("Mask Map (Patch Grid)", mask_grid, "gray"),
        ("Masked SAR (Patch Grid)", masked_grid, "gray"),
        ("Feature S1", normalize_for_display(features[0]), "viridis"),
        ("Feature S2", normalize_for_display(features[1]), "viridis"),
        ("Feature S3", normalize_for_display(features[2]), "viridis"),
        ("Feature S4", normalize_for_display(features[3]), "viridis"),
        ("Feature S5", normalize_for_display(features[4]), "viridis"),
        ("Feature S6", normalize_for_display(features[5]), "viridis"),
        ("Fused Target Feature", normalize_for_display(fused), "magma"),
    ]

    cols = 5
    rows = int(np.ceil(len(vis_items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i, (title, img, cmap) in enumerate(vis_items):
        r, c = divmod(i, cols)
        axes[r, c].imshow(img, cmap=cmap)
        axes[r, c].set_title(title, fontsize=11)
        axes[r, c].axis("off")

    for j in range(len(vis_items), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"SAR Target Visualization | mask_ratio={args.mask_ratio} | demo_grid={args.demo_grid_size}x{args.demo_grid_size}\n"
        f"fusion_weights={np.round(weights, 4).tolist()}",
        fontsize=12,
    )
    fig.tight_layout()

    overview_path = os.path.join(args.output_dir, "sar_target_overview.png")
    fig.savefig(overview_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    save_single_image(os.path.join(args.output_dir, "original_sar.png"), normalize_for_display(image), cmap="gray")
    save_single_image(os.path.join(args.output_dir, "mask_map.png"), mask_grid, cmap="gray")
    save_single_image(
        os.path.join(args.output_dir, "masked_sar.png"),
        masked_grid,
        cmap="gray",
    )
    for idx, feat in enumerate(features, start=1):
        save_single_image(
            os.path.join(args.output_dir, f"feature_s{idx}.png"),
            normalize_for_display(feat),
            cmap="viridis",
        )
    save_single_image(
        os.path.join(args.output_dir, "fused_target_feature.png"),
        normalize_for_display(fused),
        cmap="magma",
    )

    info_path = os.path.join(args.output_dir, "fusion_weights.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Fusion weights (softmax over learnable simple_weights):\n")
        f.write(", ".join([f"{v:.8f}" for v in weights.tolist()]) + "\n")
        if args.pretrained:
            f.write(f"Loaded pretrained checkpoint: {args.pretrained}\n")

    if args.show:
        plt.figure(figsize=(10, 6))
        img = cv2.imread(overview_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print(f"Visualization saved to: {args.output_dir}")
    print(f"Overview image: {overview_path}")
    print(f"Fusion weights: {np.round(weights, 6).tolist()}")


if __name__ == "__main__":
    main()
