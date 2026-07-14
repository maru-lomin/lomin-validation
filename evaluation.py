import argparse
import ast
import re
from collections.abc import Callable

import pandas as pd

SHEETS = ["kv", "tables"]


def normalize(val) -> str:
    return str(val).strip().strip('"').strip("'")


def exact_match(pred, gt) -> int:
    return int(normalize(pred) == normalize(gt))


def _edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else min(prev, dp[j], dp[j - 1]) + 1
    return dp[m]


def wer(pred, gt) -> float:
    ref, hyp = normalize(gt).split(), normalize(pred).split()
    denom = max(len(ref), len(hyp))
    if denom == 0:
        return 0.0
    return _edit_distance(hyp, ref) / denom


METRICS: dict[str, Callable] = {
    "exact_match": exact_match,
    "wer": wer,
}


def parse_items(val) -> list[str]:
    s = str(val)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return re.findall(r"'([^']*)'", s)


def apply_metric(fn: Callable, pred, gt, itemwise: bool) -> float:
    if not itemwise:
        return fn(pred, gt)
    preds, gts = parse_items(pred), parse_items(gt)
    n = max(len(preds), len(gts), 1)
    scores = [fn(preds[i] if i < len(preds) else "", gts[i] if i < len(gts) else "") for i in range(n)]
    return sum(scores) / n


def score(result: pd.DataFrame, answer: pd.DataFrame, metrics: list[str], itemwise: bool = False) -> pd.DataFrame:
    answer = answer.set_index("filename")
    cols: dict[str, pd.Series | list] = {"filename": result["filename"]}
    for col in result.columns.drop("filename"):
        pred, gt = result[col], result["filename"].map(answer[col])
        cols[f"{col}_PRED"] = pred
        cols[f"{col}_ANSWER"] = gt
        for m in metrics:
            fn = METRICS[m]
            cols[f"{col}_{m}"] = [apply_metric(fn, p, g, itemwise) for p, g in zip(pred, gt)]
    return pd.DataFrame(cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--answer_sheet", required=True)
    p.add_argument("--eval_result_path", required=True)
    p.add_argument("--metric", nargs="+", default=["exact_match"], choices=list(METRICS))
    args = p.parse_args()

    with pd.ExcelWriter(args.eval_result_path) as writer:
        for sheet in SHEETS:
            result = pd.read_excel(args.result, sheet_name=sheet)
            answer = pd.read_excel(args.answer_sheet, sheet_name=sheet)
            score(result, answer, args.metric, itemwise=(sheet == "tables")).to_excel(
                writer, sheet_name=sheet, index=False
            )


if __name__ == "__main__":
    main()
