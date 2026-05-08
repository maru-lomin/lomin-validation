from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_gt_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"].strip(): row["doc_type"].strip() for row in reader}


def load_pred_map(path: Path) -> dict[str, str]:
    pred_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            filename = str(item.get("file_name", "")).strip()
            pred = ""
            response_items = item.get("response", {}).get("response", [])
            if isinstance(response_items, list) and response_items:
                pred = (
                    response_items[0].get("inference_result", {}).get("doc_type", "")
                )
            pred_map[filename] = str(pred).strip()
    return pred_map


def write_evaluation_csv(
    gt_map: dict[str, str], pred_map: dict[str, str], out_path: Path
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    all_files = sorted(set(gt_map) | set(pred_map))
    for filename in all_files:
        gt = gt_map.get(filename, "")
        pred = pred_map.get(filename, "")
        if gt and pred:
            status = "correct" if gt == pred else "wrong"
        elif gt and not pred:
            status = "missing_prediction"
        elif not gt and pred:
            status = "missing_ground_truth"
        else:
            status = "empty"
        rows.append((filename, gt, pred, status))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt_doc_type", "pred_doc_type", "status"])
        writer.writerows(rows)
    return rows


def draw_confusion_matrix(rows: list[tuple[str, str, str, str]], out_path: Path) -> None:
    labels = sorted(
        {label for _, gt, pred, _ in rows for label in (gt, pred) if label}
    )
    if not labels:
        raise RuntimeError("No labels available to build confusion matrix.")

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    gt_label_counts: dict[str, int] = {label: 0 for label in labels}

    for _, gt, pred, _ in rows:
        if gt:
            gt_label_counts[gt] += 1
        if not gt or not pred:
            continue
        matrix[label_to_idx[gt]][label_to_idx[pred]] += 1

    normalized_matrix = [[0.0 for _ in labels] for _ in labels]
    for gt_label, gt_idx in label_to_idx.items():
        denom = gt_label_counts.get(gt_label, 0)
        if denom <= 0:
            continue
        for pred_idx in range(len(labels)):
            normalized_matrix[gt_idx][pred_idx] = matrix[gt_idx][pred_idx] / denom

    figsize = max(6, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(normalized_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = normalized_matrix[i][j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_classification(
    *, gt_csv: Path, response_jsonl: Path, evaluation_csv: Path, confusion_matrix_png: Path
) -> None:
    if not gt_csv.is_file():
        raise FileNotFoundError(f"gt.csv not found: {gt_csv}")
    if not response_jsonl.is_file():
        raise FileNotFoundError(f"response.jsonl not found: {response_jsonl}")

    gt_map = load_gt_map(gt_csv)
    pred_map = load_pred_map(response_jsonl)
    rows = write_evaluation_csv(gt_map, pred_map, evaluation_csv)
    draw_confusion_matrix(rows, confusion_matrix_png)

    correct = sum(1 for _, _, _, s in rows if s == "correct")
    wrong = sum(1 for _, _, _, s in rows if s == "wrong")
    print(f"[classification-eval] wrote: {evaluation_csv}")
    print(f"[classification-eval] wrote: {confusion_matrix_png}")
    print(f"[classification-eval] correct={correct}, wrong={wrong}, total={len(rows)}")
