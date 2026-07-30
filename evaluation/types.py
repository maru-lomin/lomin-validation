from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]
ValueItem = tuple[float, float, float, float, str]


@dataclass(frozen=True)
class TextAggregate:
    """파일·class 항목 단위로 모은 CER·WER·IoU·F1 집계."""

    per_pair_cer: list[float]
    per_pair_wer: list[float]
    total_char_edit_distance: int
    total_ref_chars: int
    total_word_edit_distance: int
    total_ref_words: int
    per_pair_iou: list[float]
    per_pair_f1: list[float] = field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0

    @property
    def macro_cer(self) -> float | None:
        if not self.per_pair_cer:
            return None
        return sum(self.per_pair_cer) / len(self.per_pair_cer)

    @property
    def micro_cer(self) -> float | None:
        if self.total_ref_chars <= 0:
            return None
        return self.total_char_edit_distance / self.total_ref_chars

    @property
    def macro_wer(self) -> float | None:
        if not self.per_pair_wer:
            return None
        return sum(self.per_pair_wer) / len(self.per_pair_wer)

    @property
    def micro_wer(self) -> float | None:
        if self.total_ref_words <= 0:
            return None
        return self.total_word_edit_distance / self.total_ref_words

    @property
    def mean_iou(self) -> float | None:
        if not self.per_pair_iou:
            return None
        return sum(self.per_pair_iou) / len(self.per_pair_iou)

    @property
    def macro_f1(self) -> float | None:
        pairs = self.per_pair_f1 or []
        if not pairs:
            return None
        return sum(pairs) / len(pairs)

    @property
    def micro_f1(self) -> float | None:
        tp, fp, fn = self.total_tp, self.total_fp, self.total_fn
        denom = 2 * tp + fp + fn
        if denom == 0:
            return None
        return (2 * tp) / denom


# 하위 호환 별칭
CerAggregate = TextAggregate


@dataclass(frozen=True)
class FileClassMetrics:
    """파일 하나·class 하나에 대한 CER·WER와 병합 bbox IoU.

    edit_distance: kv_scoring=edit_distance일 때 문자 Levenshtein 거리,
        char_multiset일 때 |GT|+|PRED|-2×문자 multiset 교집합.
    word_edit_distance: 공백 토큰 기준 Levenshtein 거리.
    char_intersection: 문자 multiset 교집합(각 문자 min(빈도)) 합. edit_distance 모드에서는 0.
    """

    filename: str
    class_name: str
    ref: str
    hyp: str
    cer: float
    wer: float
    edit_distance: int
    word_edit_distance: int
    char_intersection: int
    gt_rects: list[BBox]
    pred_rects: list[BBox]
    gt_merged: BBox
    pred_merged: BBox
    iou: float
    word_tp: int = 0
    word_fp: int = 0
    word_fn: int = 0
    word_f1: float = 0.0
