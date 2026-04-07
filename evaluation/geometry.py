from __future__ import annotations

from .types import BBox


def union_bbox_xywh(rects: list[BBox]) -> BBox:
    """여러 축 정렬 박스를 감싸는 최소 (x,y,w,h). 대각으로 가장 먼 점은 (min x,y)·(max x+w,y+h) 모서리."""
    if not rects:
        return (0.0, 0.0, 0.0, 0.0)
    x_min = min(x for x, _y, _w, _h in rects)
    y_min = min(y for _x, y, _w, _h in rects)
    x_max = max(x + w for x, _y, w, _h in rects)
    y_max = max(y + h for _x, y, _w, h in rects)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def format_xywh(box: BBox) -> str:
    x, y, w, h = box
    return f"x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}"


def iou_xywh(a: BBox, b: BBox) -> float:
    """축 정렬 사각형 IoU. (x, y, width, height). 면적 0이면 0."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
