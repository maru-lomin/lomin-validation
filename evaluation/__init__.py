"""gt.json과 result.json(VIA) 비교로 파일·class별 CER과 병합 bbox IoU를 계산합니다."""

from __future__ import annotations

from .cli import main
from .metrics import (
    build_cer_aggregate,
    compute_file_class_metrics,
    evaluate_cer,
)
from .reporting import SEP, print_summary
from .types import BBox, CerAggregate, FileClassMetrics, ValueItem
from .via import (
    assert_same_file_keys,
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
    "ValueItem",
    "assert_same_file_keys",
    "build_cer_aggregate",
    "compute_file_class_metrics",
    "evaluate_cer",
    "load_by_filename",
    "main",
    "merged_value_bbox_for_class",
    "print_summary",
    "value_rects_sorted_for_class",
    "value_text_concat_for_class",
]
