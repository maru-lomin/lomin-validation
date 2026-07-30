"""gt.json과 result.json(VIA) 비교로 파일·class별 CER·WER와 병합 bbox IoU를 계산합니다."""

from __future__ import annotations

from .cli import main
from .metrics import (
    build_cer_aggregate,
    build_text_aggregate,
    compute_file_class_metrics,
    evaluate_cer,
)
from .reporting import SEP, print_summary
from .types import BBox, CerAggregate, FileClassMetrics, TextAggregate, ValueItem
from .via import (
    UNVERIFIABLE_GT_VALUE,
    assert_same_file_keys,
    gt_is_unverifiable,
    load_by_filename,
    merged_value_bbox_for_class,
    value_rects_sorted_for_class,
    value_text_concat_for_class,
)

__all__ = [
    "BBox",
    "CerAggregate",
    "FileClassMetrics",
    "SEP",
    "TextAggregate",
    "ValueItem",
    "UNVERIFIABLE_GT_VALUE",
    "assert_same_file_keys",
    "build_cer_aggregate",
    "build_text_aggregate",
    "compute_file_class_metrics",
    "evaluate_cer",
    "gt_is_unverifiable",
    "load_by_filename",
    "main",
    "merged_value_bbox_for_class",
    "print_summary",
    "value_rects_sorted_for_class",
    "value_text_concat_for_class",
]
