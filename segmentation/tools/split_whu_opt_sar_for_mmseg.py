import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


VALID_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def normalize_stem(stem: str) -> str:
    name = stem.lower()
    drop_tokens = [
        "_sar",
        "_opt",
        "_optical",
        "_label",
        "_labels",
        "_mask",
        "_gt",
        "_cls",
        "_class",
        "_ann",
    ]
    for token in drop_tokens:
        if name.endswith(token):
            name = name[: -len(token)]
    return name


def collect_images(folder: Path) -> List[Path]:
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VALID_SUFFIXES]
    if not files:
        raise RuntimeError(f"No image files found in: {folder}")
    return files


def build_index(files: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for file in files:
        key = normalize_stem(file.stem)
        if key in index:
            raise RuntimeError(
                f"Duplicate normalized stem '{key}' from files: {index[key]} and {file}"
            )
        index[key] = file
    return index


def pair_files(sar_files: List[Path], ann_files: List[Path]) -> List[Tuple[str, Path, Path]]:
    sar_index = build_index(sar_files)
    ann_index = build_index(ann_files)
    keys = sorted(set(sar_index.keys()) & set(ann_index.keys()))
    if not keys:
        raise RuntimeError("No matched SAR/label pairs found.")
    missed_sar = sorted(set(ann_index.keys()) - set(sar_index.keys()))
    missed_ann = sorted(set(sar_index.keys()) - set(ann_index.keys()))
    if missed_sar:
        print(f"[Warn] Labels without SAR match: {len(missed_sar)}")
    if missed_ann:
        print(f"[Warn] SAR without label match: {len(missed_ann)}")
    return [(k, sar_index[k], ann_index[k]) for k in keys]


def split_pairs(
    pairs: List[Tuple[str, Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[str, Path, Path]]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require: train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def read_gray_image(path: Path, role: str, normalize_to_uint8: bool = False) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = np.array(Image.open(path).convert("L"))

    if role == "sar" and normalize_to_uint8 and arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)
    elif role == "label" and arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return arr


def crop_and_save(
    sar_path: Path,
    ann_path: Path,
    out_img_dir: Path,
    out_ann_dir: Path,
    crop_size: int,
    normalize_sar_to_uint8: bool,
) -> int:
    sar = read_gray_image(sar_path, role="sar", normalize_to_uint8=normalize_sar_to_uint8)
    ann = read_gray_image(ann_path, role="label", normalize_to_uint8=False)

    if sar.shape != ann.shape:
        raise RuntimeError(
            f"Shape mismatch for pair:\nSAR: {sar_path} -> {sar.shape}\nANN: {ann_path} -> {ann.shape}"
        )

    h, w = sar.shape
    grid_h = h // crop_size
    grid_w = w // crop_size
    if grid_h == 0 or grid_w == 0:
        print(f"[Skip] Image too small for crop_size={crop_size}: {sar_path.name}")
        return 0

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    base = normalize_stem(sar_path.stem)
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0 = gy * crop_size
            x0 = gx * crop_size
            y1 = y0 + crop_size
            x1 = x0 + crop_size
            sar_crop = sar[y0:y1, x0:x1]
            ann_crop = ann[y0:y1, x0:x1]

            name = f"{base}_{gy:02d}_{gx:02d}.png"
            Image.fromarray(sar_crop).save(out_img_dir / name)
            Image.fromarray(ann_crop).save(out_ann_dir / name)
            saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split WHU-OPT-SAR big images by scene and crop to mmseg tiles."
    )
    parser.add_argument(
        "--data-root",
        default=r"data/whu-opt-sar",
        help="WHU-OPT-SAR dataset root directory.",
    )
    parser.add_argument(
        "--sar-dir",
        default="sar",
        help=(
            "SAR directory. Default: sar (i.e. <data-root>\\sar). "
            "Relative paths join data-root; absolute paths used as-is."
        ),
    )
    parser.add_argument(
        "--ann-dir",
        default="predict",
        help=(
            "Annotation directory (class index masks). Default: predict. "
            "Relative paths join data-root; absolute paths used as-is."
        ),
    )
    parser.add_argument(
        "--out-name",
        default="whu_opt_sar_mmseg_256_split",
        help="Output folder name under data-root.",
    )
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--normalize-sar-to-uint8",
        action="store_true",
        help="Normalize SAR to [0,255] if original dtype is not uint8.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    sar_dir = Path(args.sar_dir)
    if not sar_dir.is_absolute():
        sar_dir = data_root / sar_dir

    ann_dir = Path(args.ann_dir)
    if not ann_dir.is_absolute():
        ann_dir = data_root / ann_dir

    if not sar_dir.exists():
        raise FileNotFoundError(f"sar_dir not found: {sar_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"ann_dir not found: {ann_dir}")

    print(f"SAR dir: {sar_dir}")
    print(f"ANN dir: {ann_dir}")

    sar_files = collect_images(sar_dir)
    ann_files = collect_images(ann_dir)
    pairs = pair_files(sar_files, ann_files)
    print(f"Matched scenes: {len(pairs)}")

    split_map = split_pairs(
        pairs=pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    for split, split_pairs_list in split_map.items():
        print(f"{split}: {len(split_pairs_list)} scenes")

    out_root = data_root / args.out_name
    total_tiles = 0
    split_tiles: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split, split_pairs_list in split_map.items():
        out_img_dir = out_root / "img_dir" / split
        out_ann_dir = out_root / "ann_dir" / split
        for key, sar_path, ann_path in split_pairs_list:
            _ = key
            num = crop_and_save(
                sar_path=sar_path,
                ann_path=ann_path,
                out_img_dir=out_img_dir,
                out_ann_dir=out_ann_dir,
                crop_size=args.crop_size,
                normalize_sar_to_uint8=args.normalize_sar_to_uint8,
            )
            split_tiles[split] += num
            total_tiles += num

    print(f"Output root: {out_root}")
    print(
        "Tiles generated - "
        f"train: {split_tiles['train']}, val: {split_tiles['val']}, test: {split_tiles['test']}, total: {total_tiles}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
