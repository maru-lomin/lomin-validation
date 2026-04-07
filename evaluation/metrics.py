from __future__ import annotations

from .edit_distance import cer_and_distance
from .geometry import iou_xywh, union_bbox_xywh
from .types import BBox, CerAggregate, FileClassMetrics, ValueItem
from .via import _value_items_for_class, iter_file_class_pairs


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


def build_cer_aggregate(metrics: list[FileClassMetrics]) -> CerAggregate:
    return CerAggregate(
        per_pair_cer=[m.cer for m in metrics],
        total_edit_distance=sum(m.edit_distance for m in metrics),
        total_ref_length=sum(len(m.ref) for m in metrics),
        per_pair_iou=[m.iou for m in metrics],
    )


def evaluate_cer(
    common: list[str], gt_by: dict[str, dict], res_by: dict[str, dict]
) -> list[FileClassMetrics]:
    """파일·class별 메트릭 목록."""
    from .reporting import print_file_class_metrics

    metrics: list[FileClassMetrics] = []

    for fn in common:
        gt_e = gt_by[fn]
        res_e = res_by[fn]
        for cls in iter_file_class_pairs(gt_e):
            m = compute_file_class_metrics(fn, gt_e, res_e, cls)
            metrics.append(m)
            print_file_class_metrics(m)

    return metrics
