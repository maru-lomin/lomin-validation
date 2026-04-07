from __future__ import annotations


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
