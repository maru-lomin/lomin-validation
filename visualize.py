"""
VIA JSON(VIA 형식 주석 파일, 예: result.json)의 KV(key/value) 영역을 이미지 위에 시각화합니다.
inference_demo.py → util.utils.visualize_kv_result 와 유사한 박스·라벨 스타일을 사용합니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.image as img
import matplotlib.patches as patches
import matplotlib.pyplot as plt

mpl.use("Agg")

COLOR_KEY = (0, 0, 255)
COLOR_VALUE = (255, 0, 0)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})

# 시각화 기본값 (키/값 박스 아래 라벨)
VIS_FONTSIZE = 14
VIS_FONT_MARGIN_RATIO = 1.5
VIS_MAX_WRAP_WIDTH_FACTOR = 1.7


def _configure_matplotlib_font() -> None:
    """한글 라벨용 폰트 (OS별로 실제 설치된 CJK 폰트를 선택)."""
    if sys.platform == "win32":
        candidates = (
            "Malgun Gothic",
            "맑은 고딕",
            "Gulim",
            "Batang",
            "Malgun Gothic Semilight",
            "Microsoft YaHei",
            "Noto Sans CJK KR",
            "Noto Sans CJK JP",
            "NanumGothic",
        )
    elif sys.platform == "darwin":
        candidates = (
            "Apple SD Gothic Neo",
            "AppleGothic",
            "Noto Sans CJK KR",
            "NanumGothic",
        )
    else:
        candidates = (
            "Noto Sans CJK KR",
            "Noto Sans CJK JP",
            "NanumGothic",
            "Noto Sans CJK",
        )

    for family in candidates:
        path = fm.findfont(fm.FontProperties(family=family))
        if path and "dejavu" not in path.lower():
            plt.rcParams["font.family"] = family
            break
    mpl.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_font()


def _norm_rgb(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(x / 255.0 for x in c)


def get_text_width(fig, renderer, text: str, fontsize: float) -> float:
    text_obj = plt.text(0, 0, text, fontsize=fontsize)
    bbox = text_obj.get_window_extent(renderer=renderer)
    text_obj.remove()
    return bbox.width


def split_text_by_bbox(
    text: str, fontsize: float, max_bbox_width: float, renderer, fig
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        w = get_text_width(fig, renderer, test_line, fontsize)
        if w <= max_bbox_width or not current_line:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    return lines


def load_via_json_by_filename(via_json_path: Path) -> dict[str, dict]:
    with via_json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_name: dict[str, dict] = {}
    for entry in data.values():
        fn = entry.get("filename")
        if not fn:
            continue
        by_name[fn] = entry
    return by_name


def list_kv_rect_regions(regions: list[dict]) -> list[tuple[dict, str]]:
    """key/value 서브클래스인 rect 영역만 (region, sub_class) 리스트로 반환."""
    out: list[tuple[dict, str]] = []
    for region in regions:
        ra = region.get("region_attributes") or {}
        sub = ra.get("sub_class")
        if sub not in ("key", "value"):
            continue
        shape = region.get("shape_attributes") or {}
        if shape.get("name") != "rect":
            continue
        out.append((region, sub))
    return out


def _value_display_text(ra: dict) -> str:
    if "value" in ra and ra.get("value") is not None:
        return str(ra["value"])
    return ra.get("text") or ""


def _label_for_kv_region(region: dict, sub: str) -> str:
    ra = region.get("region_attributes") or {}
    cls = ra.get("class") or ""
    if sub == "key":
        text_content = ra.get("text") or ""
        raw = f"{cls}: {text_content}".strip() if cls else text_content.strip()
    else:
        text_content = _value_display_text(ra)
        raw = f"{cls}: {text_content}".strip() if cls else text_content.strip()
    return raw.replace("☑", "<O>").replace("☐", "<X>").replace("$", r"\$")


def _add_kv_annotation(
    ax,
    region: dict,
    sub: str,
    fontsize: float,
    renderer,
    fig,
) -> None:
    shape = region.get("shape_attributes") or {}
    x = float(shape["x"])
    y = float(shape["y"])
    w = float(shape["width"])
    h = float(shape["height"])

    box_color = COLOR_KEY if sub == "key" else COLOR_VALUE
    edge = _norm_rgb(box_color)
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            edgecolor=edge,
            linewidth=2,
            fill=False,
        )
    )

    label = _label_for_kv_region(region, sub)
    max_wrap = w * VIS_MAX_WRAP_WIDTH_FACTOR
    lines = split_text_by_bbox(label, fontsize, max_wrap, renderer, fig)
    bottom = y + h
    for i, line in enumerate(lines):
        ax.text(
            x,
            bottom + (fontsize * VIS_FONT_MARGIN_RATIO * (i + 1)),
            line,
            fontsize=fontsize,
            fontweight="bold",
            color="blue",
            va="bottom",
            ha="left",
            bbox=dict(
                facecolor="lightgray",
                alpha=0.6,
                edgecolor="none",
                boxstyle="round,pad=0",
            ),
        )


def visualize_via_json_kv(
    image_path: Path,
    regions: list[dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    image = img.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.imshow(image)

    fontsize = VIS_FONTSIZE
    renderer = fig.canvas.get_renderer()

    for region, sub in list_kv_rect_regions(regions):
        _add_kv_annotation(ax, region, sub, fontsize, renderer, fig)

    plt.axis("scaled")
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        str(output_path),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def list_image_filenames(images_dir: Path) -> list[str]:
    return sorted(
        n
        for n in os.listdir(images_dir)
        if Path(n).suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="dataset_sampled/images 이미지에 VIA JSON의 KV 영역을 표시해 images_kv에 저장합니다."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=base / "dataset_sampled" / "images",
        help="입력 이미지 디렉터리",
    )
    parser.add_argument(
        "--via-json",
        type=Path,
        default=base / "dataset_sampled" / "gt.json",
        help="VIA 형식 JSON 경로 (예: dataset_sampled/gt.json, result/result.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "dataset_sampled" / "images_kv",
        help="시각화 결과 저장 디렉터리 (파일명은 원본과 동일)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="저장 PNG DPI",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    if not args.via_json.is_file():
        raise FileNotFoundError(f"VIA JSON not found: {args.via_json}")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {args.images_dir}")

    via_json_by_file = load_via_json_by_filename(args.via_json)
    names = list_image_filenames(args.images_dir)

    written = 0
    for name in names:
        in_path = args.images_dir / name
        out_path = args.output_dir / name
        entry = via_json_by_file.get(name)
        if not entry:
            logging.warning("VIA JSON에 없는 파일 — 건너뜀: %s", name)
            continue
        regions = entry.get("regions") or []
        if not list_kv_rect_regions(regions):
            logging.warning("KV(key/value) 영역 없음 — 건너뜀: %s", name)
            continue
        visualize_via_json_kv(in_path, regions, out_path, dpi=args.dpi)
        written += 1
        logging.info("저장: %s", out_path)

    logging.info("완료: %d개 파일 생성 (입력 이미지 %d개)", written, len(names))


if __name__ == "__main__":
    main()
