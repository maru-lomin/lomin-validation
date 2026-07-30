from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEMO_RESULT_FILENAME = "final_response.json"
PDF_EXTENSIONS = {".pdf"}


def via_result_key(filename: str, file_size: int) -> str:
    return f"{filename}{file_size}"


def box_to_shape_attributes(four: list[float]) -> dict[str, Any]:
    a, b, c, d = four
    w, h = c, d
    if w <= 0 or h <= 0:
        w = max(0.0, c - a)
        h = max(0.0, d - b)
    return {"name": "rect", "x": a, "y": b, "width": w, "height": h}


def _make_value_region(
    shape: dict[str, Any], class_name: str, text: str, value: str
) -> dict[str, Any]:
    return {
        "shape_attributes": shape,
        "region_attributes": {
            "sub_class": "value",
            "class": class_name,
            "text": text,
            "value": value,
        },
    }


def build_regions_from_kv(kv: dict[str, Any]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for class_name, block in sorted(kv.items()):
        if not isinstance(block, dict):
            continue
        texts = block.get("text") or [""]
        values = block.get("value") or [""]
        boxes = block.get("box") or [[0.0, 0.0, 0.0, 0.0]]
        if not isinstance(texts, list):
            texts = [str(texts)]
        if not isinstance(values, list):
            values = [str(values)]
        if not isinstance(boxes, list):
            boxes = [[0.0, 0.0, 0.0, 0.0]]
        n = max(len(boxes), len(texts), len(values), 1)
        for i in range(n):
            box = boxes[i] if i < len(boxes) else boxes[0]
            t = str(texts[i] if i < len(texts) else "")
            v = str(values[i] if i < len(values) else "")
            regions.append(
                _make_value_region(box_to_shape_attributes(box), class_name, t, v)
            )
    return regions


def build_via_entry(filename: str, file_size: int, kv: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": filename,
        "size": file_size,
        "regions": build_regions_from_kv(kv),
    }


def _bbox_xyxy_to_xywh(bbox: list[float]) -> list[float]:
    if len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = (float(bbox[i]) for i in range(4))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _field_value(block: dict[str, Any]) -> str | None:
    raw = block.get("normalized_value")
    if raw is None:
        raw = block.get("value")
    if raw is None:
        return None
    return str(raw)


def _normalize_kv_boxes_xywh(kv: dict[str, Any]) -> dict[str, Any]:
    """inference-pipeline final_response.kv (box=xyxy) → build_regions_from_kv용 xywh."""
    normalized: dict[str, Any] = {}
    for class_name, block in kv.items():
        if not isinstance(block, dict):
            continue
        boxes = block.get("box") or []
        if not isinstance(boxes, list):
            boxes = [[0.0, 0.0, 0.0, 0.0]]
        normalized[class_name] = {
            **block,
            "box": [
                _bbox_xyxy_to_xywh(box)
                if isinstance(box, list)
                else [0.0, 0.0, 0.0, 0.0]
                for box in boxes
            ],
        }
    return normalized


def final_response_to_kv(final: dict[str, Any]) -> dict[str, Any] | None:
    """final_response.json (results 또는 kv 형식) → VIA 변환용 kv."""
    results = final.get("results")
    if isinstance(results, dict):
        return demo_results_to_kv(results)

    kv = final.get("kv")
    if isinstance(kv, dict):
        return _normalize_kv_boxes_xywh(kv)

    return None


def demo_results_to_kv(results: dict[str, Any]) -> dict[str, Any]:
    """개발서버 final_response.results → request_api inference_result.kv 형식."""
    kv: dict[str, Any] = {}
    for class_name in sorted(results.keys()):
        block = results[class_name]
        if not isinstance(block, dict):
            continue
        value_str = _field_value(block)
        if value_str is None:
            continue

        evidence = block.get("evidence") or []
        if not evidence:
            kv[class_name] = {
                "text": [""],
                "value": [value_str],
                "box": [[0.0, 0.0, 0.0, 0.0]],
            }
            continue

        texts: list[str] = []
        values: list[str] = []
        boxes: list[list[float]] = []
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            texts.append(str(ev.get("text") or ""))
            values.append(value_str)
            boxes.append(_bbox_xyxy_to_xywh(ev.get("bbox") or []))

        if not boxes:
            kv[class_name] = {
                "text": [""],
                "value": [value_str],
                "box": [[0.0, 0.0, 0.0, 0.0]],
            }
        else:
            kv[class_name] = {"text": texts, "value": values, "box": boxes}
    return kv


def resolve_pdf_path(dataset_dir: Path, document_id: str) -> Path | None:
    """document_id(폴더명)에 대응하는 PDF 경로."""
    exact = dataset_dir / f"{document_id}.pdf"
    if exact.is_file():
        return exact
    prefix = document_id.split("_", 1)[0] + "_"
    candidates = sorted(
        p
        for p in dataset_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in PDF_EXTENSIONS
        and p.stem.startswith(prefix)
    )
    if len(candidates) == 1:
        return candidates[0]
    for p in candidates:
        if p.stem == document_id or document_id in p.stem:
            return p
    return None


def load_final_response(demo_doc_dir: Path) -> dict[str, Any] | None:
    path = demo_doc_dir / DEMO_RESULT_FILENAME
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def convert_demo_result_dir(
    demo_result_dir: Path,
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """demo_result 하위 문서별 final_response → VIA 루트 dict."""
    if not demo_result_dir.is_dir():
        raise FileNotFoundError(f"demo_result 디렉터리 없음: {demo_result_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset 디렉터리 없음: {dataset_dir}")

    root: dict[str, Any] = {}
    warnings: list[str] = []

    subdirs = sorted(p for p in demo_result_dir.iterdir() if p.is_dir())
    if not subdirs:
        warnings.append(f"하위 문서 폴더 없음: {demo_result_dir}")

    for doc_dir in subdirs:
        final = load_final_response(doc_dir)
        if final is None:
            warnings.append(f"{doc_dir.name}: {DEMO_RESULT_FILENAME} 없음 — 건너뜀")
            continue

        document_id = str(final.get("document_id") or doc_dir.name)
        pdf_path = resolve_pdf_path(dataset_dir, document_id)
        if pdf_path is None:
            pdf_path = resolve_pdf_path(dataset_dir, doc_dir.name)
        if pdf_path is None:
            warnings.append(
                f"{document_id}: dataset에서 PDF를 찾지 못함 — 건너뜀"
            )
            continue

        filename = pdf_path.name
        file_size = pdf_path.stat().st_size
        kv = final_response_to_kv(final)
        if kv is None:
            warnings.append(f"{document_id}: results/kv 필드 없음 — 건너뜀")
            continue
        entry = build_via_entry(filename, file_size, kv)
        root[via_result_key(filename, file_size)] = entry

    return root, warnings


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "개발서버 파이프라인 결과(demo_result)를 "
            "채점용 VIA result.json 형식으로 변환합니다. "
            "final_response.json은 standalone API(results)와 "
            "inference-pipeline(kv) 형식을 모두 지원합니다."
        )
    )
    p.add_argument(
        "--demo-result-dir",
        type=Path,
        default=base / "convert_sample" / "demo_result",
        help="문서별 final_response.json이 있는 디렉터리",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=base / "convert_sample" / "dataset",
        help="원본 PDF가 있는 디렉터리 (filename·size 매핑용)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 gt.json 경로 (기본: demo-result-dir의 상위/gt.json)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    demo_result_dir = args.demo_result_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output = args.output
    if output is None:
        output = demo_result_dir.parent / "gt.json"
    else:
        output = output.resolve()

    root, warnings = convert_demo_result_dir(demo_result_dir, dataset_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=4)

    print(f"[완료] {len(root)}개 문서 → {output}")
    for w in warnings:
        print(f"[경고] {w}")


if __name__ == "__main__":
    main()
