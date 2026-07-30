from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """편집 거리(삽입/삭제/치환). 문자열·토큰 시퀀스 공통."""
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
    """CER과 문자 편집 거리. CER은 [0, 1]로 제한."""
    dist = levenshtein(ref, hyp)
    n = len(ref)
    if n == 0:
        cer = 0.0 if len(hyp) == 0 else 1.0
    else:
        cer = dist / n
    return min(cer, 1.0), dist


def tokenize_words(text: str) -> list[str]:
    """공백 기준 단어 토큰화."""
    return text.split()


def wer_and_distance(ref: str, hyp: str) -> tuple[float, int]:
    """WER과 단어 편집 거리. WER은 [0, 1]로 제한."""
    ref_w = tokenize_words(ref)
    hyp_w = tokenize_words(hyp)
    dist = levenshtein(ref_w, hyp_w)
    n = len(ref_w)
    if n == 0:
        wer = 0.0 if len(hyp_w) == 0 else 1.0
    else:
        wer = dist / n
    return min(wer, 1.0), dist

