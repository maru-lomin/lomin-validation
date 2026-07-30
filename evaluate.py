"""
실행 진입점. 로직은 `evaluation` 패키지에 있습니다.
`--mode kv`:
  gt.json과 result.json(VIA 형식)을 비교해 파일·class별 CER·WER와 value 병합 bbox IoU를 계산합니다.
  `--kv-scoring edit_distance`(기본): Levenshtein 기반 CER.
  `--kv-scoring char_multiset`: CER을 1-문자 multiset 재현율로 정의.
  WER(공백 토큰 Levenshtein)은 항상 원문 기준.
`--mode classification`:
  gt.csv와 response.jsonl을 비교해 evaluation.csv와 confusion_matrix.png를 생성합니다.
"""

from __future__ import annotations

from evaluation.cli import main

if __name__ == "__main__":
    main()
