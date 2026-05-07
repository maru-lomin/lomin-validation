from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from .geometry import format_xywh
from .metrics import precision_recall_f1
from .types import BBox, CerAggregate, FileClassMetrics

SEP = "-" * 80


def format_file_class_metrics(m: FileClassMetrics) -> str:
    n_gt, n_pr = len(m.gt_rects), len(m.pred_rects)
    return (
        f"file={m.filename}\n"
        f"  class={m.class_name}\n"
        f"  ref({len(m.ref)}): {m.ref!r}\n"
        f"  hyp({len(m.hyp)}): {m.hyp!r}\n"
        f"  CER={m.cer:.4f} (dist={m.edit_distance})\n"
        f"  TP={m.tp:.4f}, FP={m.fp:.4f}, FN={m.fn:.4f}\n"
        f"  bbox: GT 원본 {n_gt}개 → 합침 ({format_xywh(m.gt_merged)})\n"
        f"        pred 원본 {n_pr}개 → 합침 ({format_xywh(m.pred_merged)})\n"
        f"  IoU 비교: GT {format_xywh(m.gt_merged)}\n"
        f"            vs pred {format_xywh(m.pred_merged)}\n"
        f"  IoU={m.iou:.4f}\n\n"
    )


def print_file_class_metrics(m: FileClassMetrics) -> None:
    print(format_file_class_metrics(m), end="")


def _pair_f1(m: FileClassMetrics) -> float:
    """파일·class 한 항목의 TP·FP·FN으로 F1."""
    _, _, f1 = precision_recall_f1(m.tp, m.fp, m.fn)
    return f1


def _append_global_summary_lines(
    lines: list[str], metrics: list[FileClassMetrics], agg: CerAggregate
) -> None:
    if metrics:
        tp_s = sum(m.tp for m in metrics)
        fp_s = sum(m.fp for m in metrics)
        fn_s = sum(m.fn for m in metrics)
        p, r, micro_f1 = precision_recall_f1(tp_s, fp_s, fn_s)
        macro_f1 = sum(_pair_f1(m) for m in metrics) / len(metrics)
        lines.append(
            f"전체 합산 TP={tp_s:.4f}, FP={fp_s:.4f}, FN={fn_s:.4f} "
            f"(파일·class 항목 수: {len(metrics)})"
        )
        lines.append(
            f"Macro F1 (항목별 F1 산술평균): {macro_f1:.4f}"
        )
        lines.append(
            f"Micro Precision: {p:.4f}, Recall: {r:.4f}, F1: {micro_f1:.4f} "
            f"(합산 TP·FP·FN 기준)"
        )
    else:
        lines.append("TP/FP/FN: (항목 없음)")
    if agg.mean_iou is not None:
        lines.append(
            f"평균 IoU (병합 bbox): {agg.mean_iou:.4f} "
            f"(파일·class 항목 수: {len(agg.per_pair_iou)})"
        )
    else:
        lines.append("평균 IoU: (항목 없음)")


def print_summary(metrics: list[FileClassMetrics], agg: CerAggregate) -> None:
    print(SEP)
    if metrics:
        tp_s = sum(m.tp for m in metrics)
        fp_s = sum(m.fp for m in metrics)
        fn_s = sum(m.fn for m in metrics)
        p, r, micro_f1 = precision_recall_f1(tp_s, fp_s, fn_s)
        macro_f1 = sum(_pair_f1(m) for m in metrics) / len(metrics)
        print(
            f"전체 합산 TP={tp_s:.4f}, FP={fp_s:.4f}, FN={fn_s:.4f} "
            f"(항목 수: {len(metrics)})"
        )
        print(f"Macro F1 (항목별 F1 산술평균): {macro_f1:.4f}")
        print(
            f"Micro Precision: {p:.4f}, Recall: {r:.4f}, F1: {micro_f1:.4f} "
            f"(합산 TP·FP·FN 기준)"
        )
    if agg.mean_iou is not None:
        print(
            f"평균 IoU (병합 bbox): {agg.mean_iou:.4f} "
            f"(파일·class 항목 수: {len(agg.per_pair_iou)})"
        )


def write_detail_txt(path: Path, metrics: list[FileClassMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metrics:
            f.write(format_file_class_metrics(m))


def _format_rects_list(rects: list[BBox]) -> str:
    """개별 bbox를 한 셀에 넣기 위한 문자열."""
    if not rects:
        return ""
    parts = [format_xywh(b) for b in rects]
    return " | ".join(parts)


def _apply_header_row(ws: Worksheet, headers: list[str], header_font: Font) -> None:
    for c in range(1, len(headers) + 1):
        ws.cell(row=1, column=c).font = header_font


def _autofit_columns(
    ws: Worksheet, headers: list[str], min_w: float, max_w: float
) -> None:
    for col_idx, h in enumerate(headers, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = min(max_w, max(min_w, len(str(h)) + 2))


def write_detail_xlsx(
    path: Path, metrics: list[FileClassMetrics], common: list[str]
) -> None:
    """파일별 넓은 표(값·CER·병합 bbox)와 파일·class별 긴 표(개별 bbox 포함)를 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str], FileClassMetrics] = {
        (m.filename, m.class_name): m for m in metrics
    }
    all_classes = sorted({m.class_name for m in metrics})

    wb = Workbook()

    ws_wide = wb.active
    ws_wide.title = "파일별"
    header_font = Font(bold=True)
    wide_headers: list[str] = ["파일명"]
    for cls in all_classes:
        wide_headers.extend(
            [
                f"{cls}_Pred",
                f"{cls}_GT",
                f"{cls}_CER",
                f"{cls}_TP",
                f"{cls}_FP",
                f"{cls}_FN",
                f"{cls}_Pred_bbox",
                f"{cls}_GT_bbox",
                f"{cls}_IoU",
            ]
        )
    ws_wide.append(wide_headers)
    _apply_header_row(ws_wide, wide_headers, header_font)

    for fn in common:
        row: list[str | float] = [fn]
        for cls in all_classes:
            m = by_key.get((fn, cls))
            if m is None:
                row.extend(["", "", "", "", "", "", "", "", "", ""])
            else:
                row.extend(
                    [
                        m.hyp,
                        m.ref,
                        m.cer,
                        m.tp,
                        m.fp,
                        m.fn,
                        format_xywh(m.pred_merged),
                        format_xywh(m.gt_merged),
                        m.iou,
                    ]
                )
        ws_wide.append(row)

    ws_wide.freeze_panes = "B2"
    _autofit_columns(ws_wide, wide_headers, min_w=12, max_w=48)

    ws_long = wb.create_sheet("파일·class별")
    long_headers = [
        "파일명",
        "KV class",
        "Pred",
        "GT",
        "CER",
        "TP",
        "FP",
        "FN",
        "edit_dist",
        "IoU",
        "Pred_bbox_merged",
        "GT_bbox_merged",
        "Pred_bbox_개별",
        "GT_bbox_개별",
    ]
    ws_long.append(long_headers)
    _apply_header_row(ws_long, long_headers, header_font)

    for m in metrics:
        ws_long.append(
            [
                m.filename,
                m.class_name,
                m.hyp,
                m.ref,
                m.cer,
                m.tp,
                m.fp,
                m.fn,
                m.edit_distance,
                m.iou,
                format_xywh(m.pred_merged),
                format_xywh(m.gt_merged),
                _format_rects_list(m.pred_rects),
                _format_rects_list(m.gt_rects),
            ]
        )

    ws_long.freeze_panes = "A2"
    _autofit_columns(ws_long, long_headers, min_w=14, max_w=56)

    wb.save(path)


def _tp_fp_fn_f1_micro_macro_for_group(
    ms: list[FileClassMetrics],
) -> tuple[float, float, float, float, float, int]:
    """한 그룹: TP·FP·FN 합, Micro F1(합산 기준), Macro F1(항목별 F1 평균), 항목 수."""
    tp = sum(x.tp for x in ms)
    fp = sum(x.fp for x in ms)
    fn = sum(x.fn for x in ms)
    _, _, micro_f1 = precision_recall_f1(tp, fp, fn)
    macro_f1 = sum(_pair_f1(x) for x in ms) / len(ms) if ms else 0.0
    return tp, fp, fn, micro_f1, macro_f1, len(ms)


def _aggregate_f1_rows_by_key(
    metrics: list[FileClassMetrics],
    key: Callable[[FileClassMetrics], str],
) -> list[tuple[str, float, float, float, float, float, int]]:
    """metrics를 key로 묶어 각 그룹의 TP/FP/FN 합·Macro/Micro F1 행을 만든다."""
    grouped: dict[str, list[FileClassMetrics]] = defaultdict(list)
    for m in metrics:
        grouped[key(m)].append(m)
    rows: list[tuple[str, float, float, float, float, float, int]] = []
    for k in sorted(grouped.keys()):
        tp, fp, fn, micro_f1, macro_f1, n = _tp_fp_fn_f1_micro_macro_for_group(
            grouped[k]
        )
        rows.append((k, tp, fp, fn, macro_f1, micro_f1, n))
    return rows


def write_summary_txt(
    path: Path,
    metrics: list[FileClassMetrics],
    agg: CerAggregate,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("전체 요약")
    lines.append("-" * 40)
    _append_global_summary_lines(lines, metrics, agg)

    lines.append("")
    lines.append(
        "전체 파일에 대한 class별 집계 "
        "(Macro F1: 항목별 F1 산술평균, Micro F1: 그룹 TP·FP·FN 합 기준)"
    )
    lines.append("-" * 40)
    class_rows = _aggregate_f1_rows_by_key(metrics, lambda m: m.class_name)
    if class_rows:
        w_class = max(len(r[0]) for r in class_rows)
        lines.append(
            f"{'class':<{w_class}}  {'Macro_F1':>10}  {'Micro_F1':>10}  "
            f"{'TP':>10}  {'FP':>10}  {'FN':>10}  {'n':>4}"
        )
        for cls, tp_g, fp_g, fn_g, macro_f1, micro_f1, n in class_rows:
            lines.append(
                f"{cls:<{w_class}}  {macro_f1:10.4f}  {micro_f1:10.4f}  "
                f"{tp_g:10.4f}  {fp_g:10.4f}  {fn_g:10.4f}  {n:4d}"
            )
    else:
        lines.append("(항목 없음)")

    lines.append("")
    lines.append(
        "전체 class에 대한 파일별 집계 "
        "(Macro F1: 항목별 F1 산술평균, Micro F1: 그룹 TP·FP·FN 합 기준)"
    )
    lines.append("-" * 40)
    file_rows = _aggregate_f1_rows_by_key(metrics, lambda m: m.filename)
    if file_rows:
        w_file = max(len(r[0]) for r in file_rows)
        lines.append(
            f"{'filename':<{w_file}}  {'Macro_F1':>10}  {'Micro_F1':>10}  "
            f"{'TP':>10}  {'FP':>10}  {'FN':>10}  {'n_cls':>5}"
        )
        for fn, tp_g, fp_g, fn_g, macro_f1, micro_f1, n in file_rows:
            lines.append(
                f"{fn:<{w_file}}  {macro_f1:10.4f}  {micro_f1:10.4f}  "
                f"{tp_g:10.4f}  {fp_g:10.4f}  {fn_g:10.4f}  {n:5d}"
            )
    else:
        lines.append("(항목 없음)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
