# -*- coding: utf-8 -*-
"""
4MSAR large：从 labels/train 选取 VOC 格式 XML，在 JPEGImages 中定位对应图像并绘制检测框。
与 visualize_segmatition.py 一致：含中文路径时用 imdecode / imencode。
默认在图上显示英文类别名（由中文标签映射）；可用 --zh 显示 XML 原始名称。
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 4MSAR 等 VOC 标注里常见中文类名 -> 英文（图上只画英文名）
LABEL_ZH_TO_EN: dict[str, str] = {
    "船只": "ship",
    "船舶": "ship",
    "桥梁": "bridge",
    "桥": "bridge",
    "油罐": "oil tank",
    "油库": "oil depot",
    "飞机": "aircraft",
    "车辆": "vehicle",
    "汽车": "car",
    "港口": "port",
    "码头": "wharf",
    "海岛": "island",
    "浮标": "buoy",
    "建筑": "building",
    "飞机场": "airport",
    "坦克": "tank",
    "军舰": "warship",
}

_RE_HAS_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path: Path, img_bgr: np.ndarray) -> None:
    ext = path.suffix.lower() if path.suffix else ".png"
    if ext not in _IMG_EXTS:
        ext = ".png"
    ok, enc = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode 失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(path))


def _text(elem) -> str:
    if elem is not None and elem.text is not None:
        return elem.text.strip()
    return ""


def parse_voc_objects(xml_path: Path) -> Tuple[List[Tuple[int, int, int, int, str]], Optional[str]]:
    """
    返回 ([(xmin, ymin, xmax, ymax, name), ...], xml 内 <filename> 文本)。
    """
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    fn_el = root.find("filename")
    xml_filename = _text(fn_el) or None

    boxes: List[Tuple[int, int, int, int, str]] = []
    for obj in root.findall("object"):
        name = _text(obj.find("name")) or "object"
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = int(float(_text(bb.find("xmin"))))
            ymin = int(float(_text(bb.find("ymin"))))
            xmax = int(float(_text(bb.find("xmax"))))
            ymax = int(float(_text(bb.find("ymax"))))
        except ValueError:
            continue
        boxes.append((xmin, ymin, xmax, ymax, name))
    return boxes, xml_filename


def resolve_image(jpeg_dir: Path, xml_path: Path, xml_inner_filename: Optional[str]) -> Optional[Path]:
    """先按与 XML 同 stem 找图，再按 XML 内 <filename> 的 basename 找。"""
    stem = xml_path.stem
    for ext in _IMG_EXTS:
        p = jpeg_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    if xml_inner_filename:
        base = Path(xml_inner_filename).name
        p = jpeg_dir / base
        if p.is_file():
            return p
    return None


def _color_for_name(name: str) -> Tuple[float, float, float]:
    h = hashlib.md5(name.encode("utf-8")).digest()
    return (0.3 + 0.6 * h[0] / 255.0, 0.3 + 0.5 * h[1] / 255.0, 0.3 + 0.5 * h[2] / 255.0)


def label_for_display(raw_name: str, english: bool) -> str:
    """英文模式：中文类名查表，未知中文标为 unknown (原类名记入控制台提示由调用方处理)。"""
    if not english:
        return raw_name
    if raw_name in LABEL_ZH_TO_EN:
        return LABEL_ZH_TO_EN[raw_name]
    if _RE_HAS_CJK.search(raw_name):
        return "unknown"
    return raw_name


def apply_matplotlib_font(english: bool) -> None:
    if english:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def draw_boxes_matplotlib(
    rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int, str]],
    line_width: float,
    *,
    english_labels: bool,
) -> plt.Figure:
    h, w = rgb.shape[:2]
    lw = max(line_width, min(h, w) / 800.0)
    fig, ax = plt.subplots(1, 1, figsize=(min(14, w / 120), min(14, h / 120)), dpi=120)
    ax.imshow(rgb)
    for xmin, ymin, xmax, ymax, name in boxes:
        color = _color_for_name(name)
        disp = label_for_display(name, english_labels)
        rect = Rectangle(
            (xmin, ymin),
            max(1, xmax - xmin),
            max(1, ymax - ymin),
            fill=False,
            edgecolor=color,
            linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            max(0, ymin - 4),
            disp,
            color=color,
            fontsize=max(6, min(11, lw * 4)),
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, edgecolor="none"),
        )
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.tight_layout(pad=0)
    return fig


def draw_boxes_cv2(
    img_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int, str]],
    *,
    english_labels: bool,
) -> np.ndarray:
    """画框；可选在框左上角写短英文标签（ASCII，OpenCV 默认字体可显示）。"""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    t = max(1, min(h, w) // 400)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.35, min(0.55, min(h, w) / 3500.0))
    for xmin, ymin, xmax, ymax, name in boxes:
        color = tuple(int(c * 255) for c in _color_for_name(name))
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, thickness=t)
        if english_labels:
            disp = label_for_display(name, True)
            (tw, th), _ = cv2.getTextSize(disp, font, fs, 1)
            ty = max(0, ymin - 2)
            cv2.rectangle(out, (xmin, ty - th - 4), (xmin + tw + 4, ty), (0, 0, 0), -1)
            cv2.putText(out, disp, (xmin + 2, ty - 2), font, fs, color, 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_root = Path("dataset/detection/4msar")
    parser = argparse.ArgumentParser(description="4MSAR：指定或随机 XML + 对应图像检测框可视化。")
    parser.add_argument("--jpeg_dir", type=str, default=str(default_root / "JPEGImages"), help="图像目录。")
    parser.add_argument(
        "--label_dir",
        type=str,
        default=str(default_root / "labels" / "train"),
        help="VOC XML 目录（与 --xml  basename 联用，或在此目录随机抽）。",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="03740",
        help="指定标注，如 03740.xml 或 03740（在 label_dir 下查找）；留空则随机。",
    )
    parser.add_argument(
        "--zh",
        action="store_true",
        help="图上显示 XML 原始类别名（可能为中文）；默认关闭，使用英文映射。",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（仅在不指定 --xml 时有效）。")
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(here / "4msar_det_demo.png"),
        help="matplotlib 结果保存路径。",
    )
    parser.add_argument("--dpi", type=int, default=200, help="保存 DPI。")
    parser.add_argument("--line_width", type=float, default=2.0, help="框线宽（会随分辨率略缩放）。")
    parser.add_argument("--show", action="store_true", help="弹窗显示。")
    parser.add_argument(
        "--save_cv2",
        type=str,
        default="",
        help="可选：另存 OpenCV 版本（默认与主图一致画英文短标签），路径留空则不保存。",
    )
    return parser.parse_args()


def resolve_xml_path(label_dir: Path, xml_arg: str) -> Path:
    """支持绝对路径、相对当前工作目录、或 label_dir 下的 basename / stem。"""
    raw = xml_arg.strip()
    if not raw:
        raise ValueError("empty xml_arg")
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    cand = label_dir / raw
    if cand.is_file():
        return cand.resolve()
    if not raw.lower().endswith(".xml"):
        cand = label_dir / f"{raw}.xml"
        if cand.is_file():
            return cand.resolve()
    raise SystemExit(f"未找到 XML 文件: {xml_arg!r}（已查 label_dir={label_dir}）")


def main() -> None:
    args = parse_args()
    english = not args.zh
    apply_matplotlib_font(english)

    jpeg_dir = Path(args.jpeg_dir)
    label_dir = Path(args.label_dir)
    if not label_dir.is_dir():
        raise SystemExit(f"标签目录不存在: {label_dir}")

    xml_arg = (args.xml or "").strip()
    if xml_arg:
        xml_path = resolve_xml_path(label_dir, xml_arg)
    else:
        xml_files = sorted([p for p in label_dir.iterdir() if p.suffix.lower() == ".xml"])
        if not xml_files:
            raise SystemExit(f"{label_dir} 下未找到 .xml 文件")
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        xml_path = random.choice(xml_files).resolve()
    boxes, xml_fn = parse_voc_objects(xml_path)
    img_path = resolve_image(jpeg_dir, xml_path, xml_fn)
    if img_path is None:
        raise SystemExit(
            f"未在 {jpeg_dir} 找到与 {xml_path.name} 对应的图像"
            f"（已尝试 stem={xml_path.stem!r} 及 xml 内 filename={xml_fn!r}）。"
        )

    bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"无法解码图像: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig = draw_boxes_matplotlib(rgb, boxes, args.line_width, english_labels=english)
    save_path = Path(args.save_path)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"XML: {xml_path}")
    print(f"图像: {img_path}  目标数: {len(boxes)}")
    print(f"已保存: {save_path}")

    if english:
        unmapped = sorted({n for *_, n in boxes if n not in LABEL_ZH_TO_EN and _RE_HAS_CJK.search(n)})
        if unmapped:
            print("警告: 以下中文类名未映射为英文，图上将显示 unknown，可在 LABEL_ZH_TO_EN 中补充:", unmapped)

    if args.save_cv2:
        cv2_path = Path(args.save_cv2)
        imwrite_unicode(cv2_path, draw_boxes_cv2(bgr, boxes, english_labels=english))
        print(f"OpenCV 版: {cv2_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
