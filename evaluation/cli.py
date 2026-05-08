from __future__ import annotations

import argparse
from pathlib import Path

from .classification import evaluate_classification
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
        description=(
            "평가 실행기: kv(gt.json vs result.json) 또는 "
            "classification(gt.csv vs response.jsonl)"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["kv", "classification"],
        default="kv",
        help="평가 모드 선택",
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
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=base / "dataset_classification" / "gt.csv",
        help="classification GT CSV",
    )
    parser.add_argument(
        "--response-jsonl",
        type=Path,
        default=base / "result" / "response.jsonl",
        help="request_api.py 원본 응답 JSONL",
    )
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=base / "result" / "evaluation.csv",
        help="classification 파일별 평가 CSV",
    )
    parser.add_argument(
        "--confusion-matrix-png",
        type=Path,
        default=base / "result" / "confusion_matrix.png",
        help="classification confusion matrix PNG",
    )
    return parser.parse_args()


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    args = parse_args(base)

    if args.mode == "classification":
        evaluate_classification(
            gt_csv=args.gt_csv,
            response_jsonl=args.response_jsonl,
            evaluation_csv=args.evaluation_csv,
            confusion_matrix_png=args.confusion_matrix_png,
        )
        return

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
