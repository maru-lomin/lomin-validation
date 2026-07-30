from __future__ import annotations

from collections import Counter

from .edit_distance import cer_and_distance, tokenize_words, wer_and_distance
from .geometry import iou_xywh, union_bbox_xywh
from .types import BBox, FileClassMetrics, TextAggregate, ValueItem
from .via import (
    UNVERIFIABLE_GT_VALUE,
    _value_items_for_class,
    first_nonempty_text_from_items,
    gt_is_unverifiable,
    iter_value_classes_for_entry,
    result_has_value_class,
)


def multiset_char_intersection(ref: str, hyp: str) -> int:
    """문자 단위 multiset 교집합: 각 문자 c에 대해 min(빈도_GT, 빈도_PRED)의 합."""
    if not ref or not hyp:
        return 0
    cr, ch = Counter(ref), Counter(hyp)
    return sum(min(cr[c], ch[c]) for c in cr.keys() | ch.keys())


def _text_and_rects_from_items(items: list[ValueItem]) -> tuple[str, list[BBox]]:
    """문자열은 읽기 순 첫 non-empty 항목 사용. bbox는 여전히 전부 합침."""
    rects = [(x, y, w, h) for x, y, w, h, _ in items]
    text = first_nonempty_text_from_items(items)
    return text, rects


def compute_file_class_metrics(
    filename: str,
    gt_entry: dict,
    res_entry: dict,
    class_name: str,
    *,
    kv_scoring: str = "edit_distance",
) -> FileClassMetrics:
    """gt/result 각각에 대해 value 항목을 한 번만 파싱해 CER·WER·IoU를 계산."""
    gt_items = _value_items_for_class(gt_entry, class_name)
    res_items = _value_items_for_class(res_entry, class_name)
    ref, gt_rects = _text_and_rects_from_items(gt_items)
    hyp, pred_rects = _text_and_rects_from_items(res_items)

    char_intersection = 0
    if kv_scoring == "char_multiset":
        inter = multiset_char_intersection(ref, hyp)
        char_intersection = inter
        symdiff = len(ref) + len(hyp) - 2 * inter
        if len(ref) == 0:
            c = 1.0 if len(hyp) > 0 else 0.0
        else:
            c = 1.0 - (inter / len(ref))
        dist = symdiff
    else:
        c, dist = cer_and_distance(ref, hyp)

    w, wdist = wer_and_distance(ref, hyp)

    # 빈 value region만 있는 경우: 오류로 처리
    if not ref and not hyp and result_has_value_class(res_entry, class_name):
        c, dist = 1.0, max(dist, 1)
        w, wdist = 1.0, max(wdist, 1)

    gt_m = union_bbox_xywh(gt_rects)
    pred_m = union_bbox_xywh(pred_rects)
    iou_val = iou_xywh(gt_m, pred_m)

    # Word-level F1 (strip → split으로 토큰화)
    ref_words = Counter(ref.strip().split())
    hyp_words = Counter(hyp.strip().split())
    tp = sum(min(ref_words[w], hyp_words[w]) for w in ref_words)
    fp = sum(hyp_words[w] for w in hyp_words if w not in ref_words) + sum(
        max(0, hyp_words[w] - ref_words[w]) for w in ref_words
    )
    fn = sum(ref_words[w] for w in ref_words if w not in hyp_words) + sum(
        max(0, ref_words[w] - hyp_words[w]) for w in ref_words if w in hyp_words
    )
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0

    return FileClassMetrics(
        filename=filename,
        class_name=class_name,
        ref=ref,
        hyp=hyp,
        cer=c,
        wer=w,
        edit_distance=dist,
        word_edit_distance=wdist,
        char_intersection=char_intersection,
        gt_rects=gt_rects,
        pred_rects=pred_rects,
        gt_merged=gt_m,
        pred_merged=pred_m,
        iou=iou_val,
        word_tp=tp,
        word_fp=fp,
        word_fn=fn,
        word_f1=f1,
    )


def build_text_aggregate(metrics: list[FileClassMetrics]) -> TextAggregate:
    return TextAggregate(
        per_pair_cer=[m.cer for m in metrics],
        per_pair_wer=[m.wer for m in metrics],
        total_char_edit_distance=sum(m.edit_distance for m in metrics),
        total_ref_chars=sum(len(m.ref) for m in metrics),
        total_word_edit_distance=sum(m.word_edit_distance for m in metrics),
        total_ref_words=sum(len(tokenize_words(m.ref)) for m in metrics),
        per_pair_iou=[m.iou for m in metrics],
        per_pair_f1=[m.word_f1 for m in metrics],
        total_tp=sum(m.word_tp for m in metrics),
        total_fp=sum(m.word_fp for m in metrics),
        total_fn=sum(m.word_fn for m in metrics),
    )


# 하위 호환 별칭
build_cer_aggregate = build_text_aggregate


def evaluate_cer(
    common: list[str],
    gt_by: dict[str, dict],
    res_by: dict[str, dict],
    *,
    kv_scoring: str = "edit_distance",
) -> tuple[list[FileClassMetrics], int]:
    """파일·class별 메트릭 목록. class는 GT·result value 클래스의 합집합."""
    metrics: list[FileClassMetrics] = []
    excluded_unverifiable = 0
    total_cases = 0

    for fn in common:
        gt_e = gt_by[fn]
        res_e = res_by[fn]
        classes = set(iter_value_classes_for_entry(gt_e)) | set(
            iter_value_classes_for_entry(res_e)
        )
        for cls in sorted(classes):
            if gt_is_unverifiable(gt_e, cls):
                excluded_unverifiable += 1
                continue
            m = compute_file_class_metrics(
                fn, gt_e, res_e, cls, kv_scoring=kv_scoring
            )
            metrics.append(m)
            total_cases += 1
            print(f"[{total_cases}/{len(common)}파일] {fn} / {cls} 처리 완료", end="\r", flush=True)

    print()  # 마지막 줄 개행
    print(f"채점 완료: 전체 {len(common)}개 파일, {total_cases}건 처리")
    if excluded_unverifiable:
        print(
            f"채점 제외(GT '{UNVERIFIABLE_GT_VALUE}'): {excluded_unverifiable}건"
        )

    return metrics, excluded_unverifiable
