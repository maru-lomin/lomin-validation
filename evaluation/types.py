from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]
ValueItem = tuple[float, float, float, float, str]


@dataclass(frozen=True)
class CerAggregate:
    """파일·class 항목 단위로 모은 CER·IoU 집계."""

    per_pair_cer: list[float]
    total_edit_distance: int
    total_ref_length: int
    per_pair_iou: list[float]

    @property
    def macro_cer(self) -> float | None:
        if not self.per_pair_cer:
            return None
        return sum(self.per_pair_cer) / len(self.per_pair_cer)

    @property
    def micro_cer(self) -> float | None:
        if self.total_ref_length <= 0:
            return None
        return self.total_edit_distance / self.total_ref_length

    @property
    def mean_iou(self) -> float | None:
        if not self.per_pair_iou:
            return None
        return sum(self.per_pair_iou) / len(self.per_pair_iou)


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
