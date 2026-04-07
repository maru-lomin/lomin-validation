"""
gt.json과 result.json(VIA 형식, main.py 출력)을 비교해
파일·class별 문자 단위 오차(CER)와 value 영역 병합 bbox IoU(겹침)를 계산합니다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SEP = "-" * 80

BBox = tuple[float, float, float, float]
ValueItem = tuple[float, float, float, float, str]


@dataclass(frozen=True)
class FileClassMetrics:
    """파일 하나·class 하나에 대한 CER과 병합 bbox IoU."""

    filename: str
    class_name: str
    ref: str
    hyp: str
    cer: float
    edit_distance: int
    gt_rects: list[BBox]
    pred_rects: list[BBox]
    gt_merged: BBox
    pred_merged: BBox
    iou: float


def levenshtein(a: str, b: str) -> int:
    """편집 거리(삽입/삭제/치환)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[lb]


def cer_and_distance(ref: str, hyp: str) -> tuple[float, int]:
    """CER과 편집 거리를 한 번의 Levenshtein 계산으로 반환."""
    dist = levenshtein(ref, hyp)
    n = len(ref)
    if n == 0:
        return (0.0 if len(hyp) == 0 else 1.0), dist
    return dist / n, dist


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


def _value_items_for_class(entry: dict, class_name: str) -> list[ValueItem]:
    """(x,y,w,h,text) 목록을 읽기 순으로."""
    items: list[ValueItem] = []
    for region in entry.get("regions") or []:
        ra = region.get("region_attributes") or {}
        if ra.get("sub_class") != "value":
            continue
        if ra.get("class") != class_name:
            continue
        shape = region.get("shape_attributes") or {}
        x = float(shape.get("x", 0))
        y = float(shape.get("y", 0))
        w = float(shape.get("width", 0))
        h = float(shape.get("height", 0))
        if "value" in ra and ra.get("value") is not None:
            t = str(ra.get("value"))
        else:
            t = ra.get("text") or ""
        items.append((x, y, w, h, str(t)))
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


def value_rects_sorted_for_class(entry: dict, class_name: str) -> list[BBox]:
    """sub_class==value, class 일치 bbox(개별). CER 문자열과 동일한 읽기 순."""
    return [(x, y, w, h) for x, y, w, h, _ in _value_items_for_class(entry, class_name)]


def merged_value_bbox_for_class(entry: dict, class_name: str) -> BBox:
    """같은 class value 박스가 여러 개면 축 정렬 최소 외접 bbox 하나로 합침."""
    return union_bbox_xywh(value_rects_sorted_for_class(entry, class_name))


def value_text_concat_for_class(entry: dict, class_name: str) -> str:
    """sub_class==value 이고 class 일치: value 우선, 없으면 text. 박스 크기 반영 읽기 순."""
    items = _value_items_for_class(entry, class_name)
    return "".join(t for *_, t in items)


def _text_and_rects_from_items(items: list[ValueItem]) -> tuple[str, list[BBox]]:
    text = "".join(t for *_, t in items)
    rects = [(x, y, w, h) for x, y, w, h, _ in items]
    return text, rects


def compute_file_class_metrics(
    filename: str, gt_entry: dict, res_entry: dict, class_name: str
) -> FileClassMetrics:
    """gt/result 각각에 대해 value 항목을 한 번만 파싱해 CER·IoU를 계산."""
    gt_items = _value_items_for_class(gt_entry, class_name)
    res_items = _value_items_for_class(res_entry, class_name)
    ref, gt_rects = _text_and_rects_from_items(gt_items)
    hyp, pred_rects = _text_and_rects_from_items(res_items)

    c, dist = cer_and_distance(ref, hyp)
    gt_m = union_bbox_xywh(gt_rects)
    pred_m = union_bbox_xywh(pred_rects)
    iou_val = iou_xywh(gt_m, pred_m)

    return FileClassMetrics(
        filename=filename,
        class_name=class_name,
        ref=ref,
        hyp=hyp,
        cer=c,
        edit_distance=dist,
        gt_rects=gt_rects,
        pred_rects=pred_rects,
        gt_merged=gt_m,
        pred_merged=pred_m,
        iou=iou_val,
    )


def iter_file_class_pairs(gt_entry: dict) -> list[str]:
    classes: set[str] = set()
    for region in gt_entry.get("regions") or []:
        ra = region.get("region_attributes") or {}
        if ra.get("sub_class") != "value":
            continue
        c = ra.get("class")
        if c:
            classes.add(c)
    return sorted(classes)


def _short_list_preview(names: list[str], limit: int = 20) -> str:
    head = names[:limit]
    suffix = " …" if len(names) > limit else ""
    return f"{head}{suffix}"


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


def format_file_class_metrics(m: FileClassMetrics) -> str:
    n_gt, n_pr = len(m.gt_rects), len(m.pred_rects)
    return (
        f"file={m.filename}\n"
        f"  class={m.class_name}\n"
        f"  ref({len(m.ref)}): {m.ref!r}\n"
        f"  hyp({len(m.hyp)}): {m.hyp!r}\n"
        f"  CER={m.cer:.4f} (dist={m.edit_distance})\n"
        f"  bbox: GT 원본 {n_gt}개 → 합침 ({format_xywh(m.gt_merged)})\n"
        f"        pred 원본 {n_pr}개 → 합침 ({format_xywh(m.pred_merged)})\n"
        f"  IoU 비교: GT {format_xywh(m.gt_merged)}\n"
        f"            vs pred {format_xywh(m.pred_merged)}\n"
        f"  IoU={m.iou:.4f}\n\n"
    )


def print_file_class_metrics(m: FileClassMetrics) -> None:
    print(format_file_class_metrics(m), end="")


def evaluate_cer(
    common: list[str], gt_by: dict[str, dict], res_by: dict[str, dict]
) -> list[FileClassMetrics]:
    """파일·class별 메트릭 목록."""
    metrics: list[FileClassMetrics] = []

    for fn in common:
        gt_e = gt_by[fn]
        res_e = res_by[fn]
        for cls in iter_file_class_pairs(gt_e):
            m = compute_file_class_metrics(fn, gt_e, res_e, cls)
            metrics.append(m)
            print_file_class_metrics(m)

    return metrics


def aggregate_from_metrics(
    metrics: list[FileClassMetrics],
) -> tuple[list[float], int, int, list[float]]:
    all_cer = [m.cer for m in metrics]
    total_dist = sum(m.edit_distance for m in metrics)
    total_ref_len = sum(len(m.ref) for m in metrics)
    mean_ious = [m.iou for m in metrics]
    return all_cer, total_dist, total_ref_len, mean_ious


def print_summary(
    all_cer: list[float],
    total_dist: int,
    total_ref_len: int,
    mean_ious_per_pair: list[float],
) -> None:
    print(SEP)
    if all_cer:
        macro = sum(all_cer) / len(all_cer)
        print(f"Macro 평균 CER: {macro:.4f} (항목 수: {len(all_cer)})")
    if total_ref_len > 0:
        micro = total_dist / total_ref_len
        print(f"Micro 평균 CER: {micro:.4f} (총 ref 글자 수: {total_ref_len})")
    if mean_ious_per_pair:
        avg_iou = sum(mean_ious_per_pair) / len(mean_ious_per_pair)
        print(
            f"평균 IoU (병합 bbox): {avg_iou:.4f} "
            f"(파일·class 항목 수: {len(mean_ious_per_pair)})"
        )


def write_detail_txt(path: Path, metrics: list[FileClassMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metrics:
            f.write(format_file_class_metrics(m))


def _macro_micro_for_group(ms: list[FileClassMetrics]) -> tuple[float, float, int]:
    """한 그룹(같은 class 또는 같은 파일)의 Macro CER, Micro CER, 항목 수."""
    macro = sum(x.cer for x in ms) / len(ms)
    dist_sum = sum(x.edit_distance for x in ms)
    ref_sum = sum(len(x.ref) for x in ms)
    micro = (dist_sum / ref_sum) if ref_sum > 0 else 0.0
    return macro, micro, len(ms)


def _macro_micro_rows_by_key(
    metrics: list[FileClassMetrics],
    key: Callable[[FileClassMetrics], str],
) -> list[tuple[str, float, float, int]]:
    """metrics를 key로 묶어 각 그룹의 Macro/Micro CER 행을 만든다 (class별·파일별 공통)."""
    grouped: dict[str, list[FileClassMetrics]] = defaultdict(list)
    for m in metrics:
        grouped[key(m)].append(m)
    rows: list[tuple[str, float, float, int]] = []
    for k in sorted(grouped.keys()):
        macro, micro, n = _macro_micro_for_group(grouped[k])
        rows.append((k, macro, micro, n))
    return rows


def write_summary_txt(
    path: Path,
    metrics: list[FileClassMetrics],
    all_cer: list[float],
    total_dist: int,
    total_ref_len: int,
    mean_ious_per_pair: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("전체 요약")
    lines.append("-" * 40)
    if all_cer:
        macro = sum(all_cer) / len(all_cer)
        lines.append(f"Macro 평균 CER: {macro:.4f} (파일·class 항목 수: {len(all_cer)})")
    else:
        lines.append("Macro 평균 CER: (항목 없음)")
    if total_ref_len > 0:
        micro = total_dist / total_ref_len
        lines.append(f"Micro 평균 CER: {micro:.4f} (총 ref 글자 수: {total_ref_len})")
    else:
        lines.append("Micro 평균 CER: (ref 길이 합 0)")
    if mean_ious_per_pair:
        avg_iou = sum(mean_ious_per_pair) / len(mean_ious_per_pair)
        lines.append(
            f"평균 IoU (병합 bbox): {avg_iou:.4f} "
            f"(파일·class 항목 수: {len(mean_ious_per_pair)})"
        )
    else:
        lines.append("평균 IoU: (항목 없음)")

    lines.append("")
    lines.append("전체 파일에 대한 class별 CER")
    lines.append(
        "(Macro: 해당 class가 등장하는 파일·class 항목별 CER의 산술평균, "
        "Micro: 해당 class 항목들의 총 편집거리 / 총 ref 길이)"
    )
    lines.append("-" * 40)
    class_rows = _macro_micro_rows_by_key(metrics, lambda m: m.class_name)
    if class_rows:
        w_class = max(len(r[0]) for r in class_rows)
        lines.append(
            f"{'class':<{w_class}}  {'Macro_CER':>12}  {'Micro_CER':>12}  {'n':>4}"
        )
        for cls, macro_c, micro_c, n in class_rows:
            lines.append(
                f"{cls:<{w_class}}  {macro_c:12.4f}  {micro_c:12.4f}  {n:4d}"
            )
    else:
        lines.append("(항목 없음)")

    lines.append("")
    lines.append("전체 class에 대한 파일별 CER")
    lines.append(
        "(Macro: 해당 파일 내 class별 CER의 산술평균, "
        "Micro: 해당 파일 내 총 편집거리 / 총 ref 길이)"
    )
    lines.append("-" * 40)
    file_rows = _macro_micro_rows_by_key(metrics, lambda m: m.filename)
    if file_rows:
        w_file = max(len(r[0]) for r in file_rows)
        lines.append(
            f"{'filename':<{w_file}}  {'Macro_CER':>12}  {'Micro_CER':>12}  {'n_cls':>5}"
        )
        for fn, macro_c, micro_c, n in file_rows:
            lines.append(
                f"{fn:<{w_file}}  {macro_c:12.4f}  {micro_c:12.4f}  {n:5d}"
            )
    else:
        lines.append("(항목 없음)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(base: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gt.json vs result.json — 파일·class별 CER 및 value bbox IoU"
    )
    parser.add_argument(
        "--gt-json",
        type=Path,
        default=base / "dataset_sampled" / "gt.json",
        help="Ground truth (VIA)",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=base / "result" / "result.json",
        help="추론 결과 (gt.json과 동일 스키마)",
    )
    parser.add_argument(
        "--detail-txt",
        type=Path,
        default=base / "result" / "detail.txt",
        help="파일·class별 상세 결과 저장 경로",
    )
    parser.add_argument(
        "--summary-txt",
        type=Path,
        default=base / "result" / "summary.txt",
        help="요약(전체·class별·파일별 CER 등) 저장 경로",
    )
    return parser.parse_args()


def main() -> None:
    base = Path(__file__).resolve().parent
    args = parse_args(base)

    if not args.gt_json.is_file():
        raise FileNotFoundError(f"gt.json 없음: {args.gt_json}")
    if not args.result_json.is_file():
        raise FileNotFoundError(f"result.json 없음: {args.result_json}")

    gt_by = load_by_filename(args.gt_json)
    res_by = load_by_filename(args.result_json)
    common = assert_same_file_keys(gt_by, res_by)

    print(f"비교 대상 파일 수: {len(common)}")
    print(SEP)

    metrics = evaluate_cer(common, gt_by, res_by)
    all_cer, total_dist, total_ref_len, mean_ious = aggregate_from_metrics(metrics)

    write_detail_txt(args.detail_txt, metrics)
    write_summary_txt(
        args.summary_txt,
        metrics,
        all_cer,
        total_dist,
        total_ref_len,
        mean_ious,
    )
    print(f"상세 결과 저장: {args.detail_txt}")
    print(f"요약 저장: {args.summary_txt}")
    print()

    print_summary(all_cer, total_dist, total_ref_len, mean_ious)


if __name__ == "__main__":
    main()
