"""
실행 진입점. 로직은 `evaluation` 패키지에 있습니다.
gt.json과 result.json(VIA 형식)을 비교해 파일·class별 CER과 value 병합 bbox IoU를 계산합니다.
"""

from __future__ import annotations

from evaluation.cli import main

if __name__ == "__main__":
    main()
