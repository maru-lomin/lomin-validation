from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import build_cer_aggregate, evaluate_cer
from .reporting import (
    SEP,
    print_summary,
    write_detail_txt,
    write_detail_xlsx,
    write_summary_txt,
)
from .via import assert_same_file_keys, load_by_filename


def parse_args(base: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gt.json vs result.json — 파일·class별 CER·TP/FP/FN·F1 및 value bbox IoU"
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
        help="요약(전체·class별·파일별 F1·TP/FP/FN 등) 저장 경로",
    )
    parser.add_argument(
        "--detail-xlsx",
        type=Path,
        default=base / "result" / "detail.xlsx",
        help="파일·class별 상세 결과 Excel (.xlsx)",
    )
    return parser.parse_args()


def main() -> None:
    base = Path(__file__).resolve().parent.parent
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
    agg = build_cer_aggregate(metrics)

    write_detail_txt(args.detail_txt, metrics)
    write_detail_xlsx(args.detail_xlsx, metrics, common)
    write_summary_txt(args.summary_txt, metrics, agg)
    print(f"상세 결과 저장: {args.detail_txt}")
    print(f"Excel 상세 저장: {args.detail_xlsx}")
    print(f"요약 저장: {args.summary_txt}")
    print()

    print_summary(metrics, agg)


if __name__ == "__main__":
    main()
