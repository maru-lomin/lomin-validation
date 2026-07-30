from __future__ import annotations

import json
from pathlib import Path

from .geometry import union_bbox_xywh
from .types import BBox, ValueItem


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return float(s[m])
    return (s[m - 1] + s[m]) / 2.0


def _sort_value_items_reading_order(items: list[ValueItem]) -> list[ValueItem]:
    """같은 class의 여러 value 박스를 읽기 순으로 정렬.

    - 세로: 박스 높이 중앙값의 절반을 임계값으로, 중심 cy가 가까우면 같은 줄로 묶음.
    - 가로: 같은 줄 안에서는 중심 cx(왼쪽→오른쪽).
    - 줄끼리는 줄의 평균 cy가 위→아래.
    """
    if len(items) <= 1:
        return list(items)

    heights = [h for _, _, _, h, _ in items if h > 0]
    median_h = _median(heights) if heights else 1.0
    tau = max(0.5 * median_h, 1.0)

    def cy(it: ValueItem) -> float:
        _x, y, _w, h, _t = it
        return y + h / 2.0 if h > 0 else y

    def cx(it: ValueItem) -> float:
        x, _y, w, _h, _t = it
        return x + w / 2.0 if w > 0 else x

    by_cy = sorted(items, key=lambda it: (cy(it), cx(it)))

    lines: list[list[ValueItem]] = []
    for it in by_cy:
        c = cy(it)
        if not lines:
            lines.append([it])
            continue
        last = lines[-1]
        mean_cy = sum(cy(t) for t in last) / len(last)
        if abs(c - mean_cy) <= tau:
            last.append(it)
        else:
            lines.append([it])

    for line in lines:
        line.sort(key=cx)
    lines.sort(key=lambda ln: sum(cy(t) for t in ln) / len(ln))

    out: list[ValueItem] = []
    for line in lines:
        out.extend(line)
    return out


def _parse_value_region(region: dict, class_name: str) -> ValueItem | None:
    """sub_class==value 이고 class가 일치하면 (x,y,w,h,text), 아니면 None."""
    ra = region.get("region_attributes") or {}
    if ra.get("sub_class") != "value" or ra.get("class") != class_name:
        return None
    shape = region.get("shape_attributes") or {}
    x = float(shape.get("x", 0))
    y = float(shape.get("y", 0))
    w = float(shape.get("width", 0))
    h = float(shape.get("height", 0))
    if "value" in ra and ra.get("value") is not None:
        t = str(ra.get("value"))
    else:
        t = ra.get("text") or ""
    return (x, y, w, h, str(t))


def _value_items_for_class(entry: dict, class_name: str) -> list[ValueItem]:
    """(x,y,w,h,text) 목록을 읽기 순으로."""
    items: list[ValueItem] = []
    for region in entry.get("regions") or []:
        item = _parse_value_region(region, class_name)
        if item is not None:
            items.append(item)
    return _sort_value_items_reading_order(items)


def load_by_filename(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_name: dict[str, dict] = {}
    for entry in data.values():
        fn = entry.get("filename")
        if fn:
            by_name[fn] = entry
    return by_name


def value_rects_sorted_for_class(entry: dict, class_name: str) -> list[BBox]:
    """sub_class==value, class 일치 bbox(개별). CER 문자열과 동일한 읽기 순."""
    return [(x, y, w, h) for x, y, w, h, _ in _value_items_for_class(entry, class_name)]


def merged_value_bbox_for_class(entry: dict, class_name: str) -> BBox:
    """같은 class value 박스가 여러 개면 축 정렬 최소 외접 bbox 하나로 합침."""
    return union_bbox_xywh(value_rects_sorted_for_class(entry, class_name))


def first_nonempty_text_from_items(items: list[ValueItem]) -> str:
    """읽기 순으로 정렬된 value 항목 중 비어 있지 않은 첫 텍스트."""
    for _x, _y, _w, _h, text in items:
        if str(text).strip():
            return str(text)
    return str(items[0][4]) if items else ""


def value_text_concat_for_class(entry: dict, class_name: str) -> str:
    """sub_class==value 이고 class 일치: value 우선, 없으면 text.

    같은 class의 region이 여러 개인 경우, 읽기 순으로 비어 있지 않은
    첫 항목의 텍스트를 사용한다(이어붙이지 않음).
    """
    return first_nonempty_text_from_items(_value_items_for_class(entry, class_name))


UNVERIFIABLE_GT_VALUE = "확인불가"


def gt_is_unverifiable(entry: dict, class_name: str) -> bool:
    """GT value 전체가 '확인불가'이면 True (채점 제외 대상)."""
    return value_text_concat_for_class(entry, class_name).strip() == UNVERIFIABLE_GT_VALUE


def result_has_value_class(entry: dict, class_name: str) -> bool:
    """result(VIA)에 해당 class의 value region이 하나라도 있으면 True."""
    for region in entry.get("regions") or []:
        ra = region.get("region_attributes") or {}
        if ra.get("sub_class") == "value" and ra.get("class") == class_name:
            return True
    return False


def iter_value_classes_for_entry(entry: dict) -> list[str]:
    """VIA entry에 sub_class==value로 등장하는 class 이름 목록."""
    classes: set[str] = set()
    for region in entry.get("regions") or []:
        ra = region.get("region_attributes") or {}
        if ra.get("sub_class") != "value":
            continue
        c = ra.get("class")
        if c:
            classes.add(c)
    return sorted(classes)


def iter_file_class_pairs(gt_entry: dict) -> list[str]:
    """GT entry에 등장하는 value class 목록 (하위 호환용 이름)."""
    return iter_value_classes_for_entry(gt_entry)


def _short_list_preview(names: list[str], limit: int = 20) -> str:
    head = names[:limit]
    suffix = " …" if len(names) > limit else ""
    return ", ".join(head) + suffix


def assert_same_file_keys(gt_by: dict[str, dict], res_by: dict[str, dict]) -> list[str]:
    """gt·result·교집합의 키 집합이 동일한지 검사하고, 정렬된 공통 파일명 목록을 반환."""
    n_gt, n_res = len(gt_by), len(res_by)
    common = sorted(set(gt_by.keys()) & set(res_by.keys()))
    n_common = len(common)
    if n_gt == n_res == n_common:
        return common
    only_gt = sorted(set(gt_by.keys()) - set(res_by.keys()))
    only_res = sorted(set(res_by.keys()) - set(gt_by.keys()))
    raise ValueError(
        "gt.json·result.json·공통 파일 수가 모두 같아야 합니다. "
        f"gt={n_gt}, result={n_res}, 공통={n_common}. "
        f"gt에만 있음({len(only_gt)}): {_short_list_preview(only_gt)}"
        f"; result에만 있음({len(only_res)}): {_short_list_preview(only_res)}"
    )
