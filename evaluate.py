"""
gt.json과 result.json(VIA 형식, main.py 출력)을 비교해
파일·class별 문자 단위 오차(CER)를 계산합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEP = "-" * 80


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


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate = 편집거리 / max(len(ref), 1)."""
    c, _ = cer_and_distance(ref, hyp)
    return c


def load_by_filename(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_name: dict[str, dict] = {}
    for entry in data.values():
        fn = entry.get("filename")
        if fn:
            by_name[fn] = entry
    return by_name


def value_text_concat_for_class(entry: dict, class_name: str) -> str:
    """sub_class==value 이고 class 일치하는 영역: value 우선, 없으면 text (GT 호환)."""
    rows: list[tuple[float, float, str]] = []
    for region in entry.get("regions") or []:
        ra = region.get("region_attributes") or {}
        if ra.get("sub_class") != "value":
            continue
        if ra.get("class") != class_name:
            continue
        shape = region.get("shape_attributes") or {}
        y = float(shape.get("y", 0))
        x = float(shape.get("x", 0))
        if "value" in ra and ra.get("value") is not None:
            t = str(ra.get("value"))
        else:
            t = ra.get("text") or ""
        rows.append((y, x, str(t)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return "".join(r[2] for r in rows)


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


def evaluate_cer(
    common: list[str], gt_by: dict[str, dict], res_by: dict[str, dict]
) -> tuple[list[float], int, int]:
    """파일·class별 CER을 계산하고 (per-item CER 목록, 총 편집거리, 총 ref 길이)를 반환."""
    all_cer: list[float] = []
    total_dist = 0
    total_ref_len = 0

    for fn in common:
        gt_e = gt_by[fn]
        res_e = res_by[fn]
        for cls in iter_file_class_pairs(gt_e):
            ref = value_text_concat_for_class(gt_e, cls)
            hyp = value_text_concat_for_class(res_e, cls)
            c, dist = cer_and_distance(ref, hyp)
            all_cer.append(c)
            total_dist += dist
            total_ref_len += len(ref)
            print(
                f"file={fn}\n"
                f"  class={cls}\n"
                f"  ref({len(ref)}): {ref!r}\n"
                f"  hyp({len(hyp)}): {hyp!r}\n"
                f"  CER={c:.4f} (dist={dist})"
            )

    return all_cer, total_dist, total_ref_len


def print_summary(all_cer: list[float], total_dist: int, total_ref_len: int) -> None:
    print(SEP)
    if all_cer:
        macro = sum(all_cer) / len(all_cer)
        print(f"Macro 평균 CER: {macro:.4f} (항목 수: {len(all_cer)})")
    if total_ref_len > 0:
        micro = total_dist / total_ref_len
        print(f"Micro 평균 CER: {micro:.4f} (총 ref 글자 수: {total_ref_len})")


def parse_args(base: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gt.json vs result.json — 파일·class별 CER"
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

    all_cer, total_dist, total_ref_len = evaluate_cer(common, gt_by, res_by)
    print_summary(all_cer, total_dist, total_ref_len)


if __name__ == "__main__":
    main()
