import argparse
import os
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


RGBA = Tuple[int, int, int, int]

# Shared face styling (keep PNG and SVG consistent)
# "top" should be lighter & paler (tinted toward white), not just brighter.
FACE_STYLE = {
    "front_factor": 0.95,
    "right_factor": 0.78,
    "top_tint": 0.45,  # 0=no tint, 1=white
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw transparent-background isometric cube grids."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vis_cube_output",
        help="Directory to save generated PNG files.",
    )
    parser.add_argument(
        "--cube_size",
        type=int,
        default=28,
        help="Base cube size. Larger value gives bigger cubes.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Pseudo-3D depth (pixels) for top/right faces.",
    )
    parser.add_argument(
        "--line_alpha",
        type=int,
        default=140,
        help="Alpha of cube edge lines in [0,255].",
    )
    parser.add_argument(
        "--export_svg",
        default=True,
        help="Also export editable SVG files (recommended for Visio editing).",
    )
    return parser.parse_args()


def shade(color: Tuple[int, int, int], factor: float, alpha: int = 255) -> RGBA:
    r, g, b = color
    rr = max(0, min(255, int(r * factor)))
    gg = max(0, min(255, int(g * factor)))
    bb = max(0, min(255, int(b * factor)))
    return rr, gg, bb, alpha


def tint_to_white(color: Tuple[int, int, int], tint: float, alpha: int = 255) -> RGBA:
    """Blend color toward white to make it paler."""
    tint = max(0.0, min(1.0, float(tint)))
    r, g, b = color
    rr = int(r * (1.0 - tint) + 255 * tint)
    gg = int(g * (1.0 - tint) + 255 * tint)
    bb = int(b * (1.0 - tint) + 255 * tint)
    return rr, gg, bb, alpha


def cube_polygons(u: float, v: float, a: float, b: float, h: float) -> Dict[str, List[Tuple[float, float]]]:
    # top center: (u, v-h)
    top = [
        (u, v - h),
        (u + a, v - h + b),
        (u, v - h + 2 * b),
        (u - a, v - h + b),
    ]
    right = [
        (u, v - h + 2 * b),
        (u + a, v - h + b),
        (u + a, v + b),
        (u, v + 2 * b),
    ]
    left = [
        (u - a, v - h + b),
        (u, v - h + 2 * b),
        (u, v + 2 * b),
        (u - a, v + b),
    ]
    return {"top": top, "left": left, "right": right}


def featuremap_cube_polygons(x: float, y: float, w: float, d: float) -> Dict[str, List[Tuple[float, float]]]:
    # Front face: axis-aligned square (x,y) -> (x+w, y+w)
    front = [(x, y), (x + w, y), (x + w, y + w), (x, y + w)]
    # Top face: slants up-right by depth d
    top = [(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)]
    # Right face: slants up-right by depth d
    right = [(x + w, y), (x + w + d, y - d), (x + w + d, y + w - d), (x + w, y + w)]
    return {"front": front, "top": top, "right": right}


def prepare_featuremap_layout(n: int, cube_size: int, depth: int):
    w = cube_size
    d = depth
    margin = cube_size  # compact but with safe padding
    ox = margin + d
    oy = margin + d
    width = ox + n * w + d + margin
    height = oy + n * w + margin

    # Draw order: back to front (top-left first) so outlines look consistent
    items = []
    for i in range(n):
        for j in range(n):
            x = ox + j * w
            y = oy + i * w
            polys = featuremap_cube_polygons(x, y, w, d)
            items.append((i, j, polys))
    return items, int(width), int(height)


def draw_cube_grid(
    n: int,
    base_color: Tuple[int, int, int],
    out_path: str,
    cube_size: int,
    depth: int,
    line_alpha: int,
) -> None:
    items, width, height = prepare_featuremap_layout(n, cube_size, depth)

    # transparent background
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    # face colors (shared with SVG)
    front_color = shade(base_color, FACE_STYLE["front_factor"], 255)
    top_color = tint_to_white(base_color, FACE_STYLE["top_tint"], 255)
    right_color = shade(base_color, FACE_STYLE["right_factor"], 255)
    edge_color = shade((255, 255, 255), 1.0, line_alpha)

    # Only outer top and right faces for a stacked "feature map" block
    for i, j, polys in items:
        front = polys["front"]
        top = polys["top"]
        right = polys["right"]

        # Draw top faces only on first row
        if i == 0:
            draw.polygon(top, fill=top_color)
            draw.line(top + [top[0]], fill=edge_color, width=1)
        # Draw right faces only on last column
        if j == n - 1:
            draw.polygon(right, fill=right_color)
            draw.line(right + [right[0]], fill=edge_color, width=1)

        # Front face always
        draw.polygon(front, fill=front_color)
        draw.line(front + [front[0]], fill=edge_color, width=1)

    image.save(out_path, "PNG")


def rgba_to_svg_fill(color: RGBA) -> str:
    r, g, b, a = color
    return f"rgb({r},{g},{b});fill-opacity:{a / 255.0:.4f}"


def rgba_to_svg_stroke(color: RGBA) -> str:
    r, g, b, a = color
    return f"rgb({r},{g},{b});stroke-opacity:{a / 255.0:.4f}"


def points_to_svg(points: List[Tuple[float, float]]) -> str:
    return " ".join([f"{x:.2f},{y:.2f}" for x, y in points])


def draw_cube_grid_svg(
    n: int,
    base_color: Tuple[int, int, int],
    out_path: str,
    cube_size: int,
    depth: int,
    line_alpha: int,
) -> None:
    items, width, height = prepare_featuremap_layout(n, cube_size, depth)

    front_color = shade(base_color, FACE_STYLE["front_factor"], 255)
    top_color = tint_to_white(base_color, FACE_STYLE["top_tint"], 255)
    right_color = shade(base_color, FACE_STYLE["right_factor"], 255)
    edge_color = shade((255, 255, 255), 1.0, line_alpha)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )

    for i, j, polys in items:
        front = polys["front"]
        top = polys["top"]
        right = polys["right"]

        if i == 0:
            lines.append(
                f'<polygon points="{points_to_svg(top)}" style="fill:{rgba_to_svg_fill(top_color)};'
                f'stroke:{rgba_to_svg_stroke(edge_color)};stroke-width:1" />'
            )
        if j == n - 1:
            lines.append(
                f'<polygon points="{points_to_svg(right)}" style="fill:{rgba_to_svg_fill(right_color)};'
                f'stroke:{rgba_to_svg_stroke(edge_color)};stroke-width:1" />'
            )
        lines.append(
            f'<polygon points="{points_to_svg(front)}" style="fill:{rgba_to_svg_fill(front_color)};'
            f'stroke:{rgba_to_svg_stroke(edge_color)};stroke-width:1" />'
        )

    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    blue = (52, 96, 190)
    green = (81, 174, 157)

    tasks = [
        (3, blue, "cube_3x3_blue.png"),
        (5, blue, "cube_5x5_blue.png"),
        (7, blue, "cube_7x7_blue.png"),
        (3, green, "cube_3x3_green.png"),
        (5, green, "cube_5x5_green.png"),
        (7, green, "cube_7x7_green.png"),
    ]

    for n, color, name in tasks:
        out_path = os.path.join(args.output_dir, name)
        draw_cube_grid(
            n=n,
            base_color=color,
            out_path=out_path,
            cube_size=args.cube_size,
            depth=args.depth,
            line_alpha=args.line_alpha,
        )
        print(f"Saved: {out_path}")
        if args.export_svg:
            svg_name = name.replace(".png", ".svg")
            svg_path = os.path.join(args.output_dir, svg_name)
            draw_cube_grid_svg(
                n=n,
                base_color=color,
                out_path=svg_path,
                cube_size=args.cube_size,
                depth=args.depth,
                line_alpha=args.line_alpha,
            )
            print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
