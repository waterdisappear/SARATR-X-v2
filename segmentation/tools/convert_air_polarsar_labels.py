import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# AIR-PolSAR-Seg color code (paper Fig.7(g)):
# housing(255,255,0), industrial(0,0,255), natural(0,255,0),
# land_use(255,0,0), water(0,255,255), other(255,255,255).
# Black (0,0,0) is treated as ignore.
COLOR_TO_LABEL = {
    (255, 255, 0): 0,    # housing
    (0, 0, 255): 1,      # industrial
    (0, 255, 0): 2,      # natural
    (255, 0, 0): 3,      # land_use
    (0, 255, 255): 4,    # water
    (255, 255, 255): 5,  # other
}
IGNORE_INDEX = 255


def convert_one_label(src_path: Path, dst_path: Path) -> None:
    rgb = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
    label = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)

    for color, cls_id in COLOR_TO_LABEL.items():
        mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        label[mask] = cls_id

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label, mode="L").save(dst_path)


def convert_split(split_dir: Path, out_dirname: str) -> int:
    gt_files = sorted(split_dir.glob("*_gt.png"))
    out_dir = split_dir / out_dirname
    converted = 0

    for gt_file in gt_files:
        out_name = gt_file.name.replace("_gt.png", "_gt_labelTrainIds.png")
        out_path = out_dir / out_name
        convert_one_label(gt_file, out_path)
        converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AIR-PolSAR-Seg RGB masks to mmseg index masks."
    )
    parser.add_argument(
        "--train-dir",
        default=r"data/raw_air_polarsar_seg/train_set",
        help="Path to train_set directory.",
    )
    parser.add_argument(
        "--test-dir",
        default=r"data/raw_air_polarsar_seg/test_set",
        help="Path to test_set directory.",
    )
    parser.add_argument(
        "--out-dirname",
        default="labels_index",
        help="Output directory name under each split dir.",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    train_count = convert_split(train_dir, args.out_dirname)
    test_count = convert_split(test_dir, args.out_dirname)

    print(f"Converted train labels: {train_count}")
    print(f"Converted test labels: {test_count}")
    print(
        "Done. Use ann_dir='labels_index' and seg_map_suffix='_gt_labelTrainIds.png'."
    )


if __name__ == "__main__":
    main()
