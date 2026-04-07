from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# --- paths & API defaults (script entrypoint uses these) ---

# IMG_DIR = Path("./dataset_sampled/images/")
# RESULT_DIR = Path("./result/")
WORKFLOW_URL = "https://beta.zixy.io/cognition-api/api/v1/workflows/api/apis"
HTTP_TIMEOUT = 300.0

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def workflow_payload() -> dict[str, Any]:
    return {
        "params": '{"api_key": "03e57dac1847ddfa296b8813f17c21c21de945e330f39b7"}',
        "api_option": {
            "async_mode": False,
            "timeout": HTTP_TIMEOUT,
        },
    }


def content_type_for(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    return CONTENT_TYPES.get(ext, "application/octet-stream")


def parse_response_body(response: requests.Response) -> Any:
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        return response.text


def extract_inference_result(body: Any) -> dict | None:
    if not isinstance(body, dict):
        return None
    resp = body.get("response")
    if not resp or not isinstance(resp, list) or not resp:
        return None
    inf = resp[0].get("inference_result")
    return inf if isinstance(inf, dict) else None


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return ["" if x is None else str(x) for x in v]
    return [str(v)]


def _normalize_boxes(box_raw: Any) -> list[list[float]]:
    """kv block의 box: [[x,y,w,h], ...] 또는 [x,y,w,h]."""
    if not box_raw:
        return [[0.0, 0.0, 0.0, 0.0]]
    if isinstance(box_raw, list) and len(box_raw) >= 1:
        first = box_raw[0]
        if isinstance(first, (list, tuple)):
            out = []
            for b in box_raw:
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    out.append([float(x) for x in b[:4]])
            return out if out else [[0.0, 0.0, 0.0, 0.0]]
        if len(box_raw) >= 4:
            return [[float(x) for x in box_raw[:4]]]
    return [[0.0, 0.0, 0.0, 0.0]]


def box_to_shape_attributes(four: list[float]) -> dict[str, Any]:
    """4실수: 기본은 x, y, width, height. width/height가 0 이하면 (x1,y1,x2,y2)로 해석 시도."""
    a, b, c, d = four
    w, h = c, d
    if w <= 0 or h <= 0:
        w = max(0.0, c - a)
        h = max(0.0, d - b)
    return {
        "name": "rect",
        "x": a,
        "y": b,
        "width": w,
        "height": h,
    }


def _make_value_region(
    shape: dict[str, Any], class_name: str, text: str, value: str
) -> dict[str, Any]:
    """API kv의 text·value를 모두 region_attributes에 보존."""
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
    """kv의 class, text, value, box만 사용해 VIA regions 생성 (value만, API에 key 없음)."""
    regions: list[dict[str, Any]] = []
    items = sorted(
        kv.items(),
        key=lambda it: (it[1].get("order", 0) if isinstance(it[1], dict) else 0, it[0]),
    )
    for class_name, block in items:
        if not isinstance(block, dict):
            continue
        texts = _as_str_list(block.get("text"))
        values = _as_str_list(block.get("value"))
        boxes = _normalize_boxes(block.get("box"))
        n = max(len(boxes), len(texts), len(values), 1)
        for i in range(n):
            box = boxes[i] if i < len(boxes) else boxes[0]
            t = texts[i] if i < len(texts) else ""
            v = values[i] if i < len(values) else ""
            shape = box_to_shape_attributes(box)
            regions.append(_make_value_region(shape, class_name, t, v))
    return regions


def build_via_entry(filename: str, file_size: int, inference_result: dict) -> dict[str, Any]:
    kv = inference_result.get("kv")
    if not isinstance(kv, dict):
        kv = {}
    regions = build_regions_from_kv(kv)
    return {
        "filename": filename,
        "size": file_size,
        "regions": regions,
    }


def via_result_key(filename: str, file_size: int) -> str:
    return f"{filename}{file_size}"


def list_image_filenames(img_dir: Path) -> list[str]:
    return sorted(
        f.name
        for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def post_workflow_for_image(
    img_path: Path,
    file_name: str,
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> requests.Response:
    ct = content_type_for(img_path)
    with img_path.open("rb") as f:
        files = [("file", (file_name, f, ct))]
        return requests.request("POST", url, data=payload, files=files, timeout=timeout)


@dataclass(frozen=True)
class RunPaths:
    img_dir: Path
    result_dir: Path

    @property
    def output_json(self) -> Path:
        return self.result_dir / "result.json"

    @property
    def response_jsonl(self) -> Path:
        return self.result_dir / "response.jsonl"


def run(paths: RunPaths, *, url: str = WORKFLOW_URL, payload: dict[str, Any] | None = None) -> None:
    payload = payload or workflow_payload()
    if not paths.img_dir.is_dir():
        print(f"[ERROR] 이미지 디렉터리가 없습니다: {paths.img_dir}")
        return

    paths.result_dir.mkdir(parents=True, exist_ok=True)

    names = list_image_filenames(paths.img_dir)
    total = len(names)
    print(f"[시작] 대상 이미지: {total}개 ({paths.img_dir})")
    print(f"[시작] 결과 파일 (API만으로 VIA 형식): {paths.output_json}")
    print(f"[시작] 원본 응답 (JSONL): {paths.response_jsonl}")

    result_root: dict[str, Any] = {}

    with paths.response_jsonl.open("w", encoding="utf-8") as jsonl_out:
        for i, file_name in enumerate(names, start=1):
            img_path = paths.img_dir / file_name
            file_size = img_path.stat().st_size
            print(f"[{i}/{total}] 요청 중: {file_name}")

            t0 = time.perf_counter()
            try:
                response = post_workflow_for_image(
                    img_path,
                    file_name,
                    url=url,
                    payload=payload,
                    timeout=HTTP_TIMEOUT,
                )
            except OSError as e:
                print(f"    -> 실패 (파일): {e}")
                line = {
                    "file_name": file_name,
                    "error": f"file_read_error: {e}",
                }
                jsonl_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                continue

            elapsed_sec = time.perf_counter() - t0
            body = parse_response_body(response)
            jsonl_out.write(
                json.dumps(
                    {
                        "file_name": file_name,
                        "status_code": response.status_code,
                        "response": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if response.status_code != 200:
                print(
                    f"    -> 실패 HTTP {response.status_code} "
                    f"(응답까지 {elapsed_sec:.2f}s)"
                )
                continue

            inference_result = extract_inference_result(body)
            if not inference_result:
                print(
                    "    -> 경고: inference_result 없음 — result.json에는 미반영 "
                    f"(응답까지 {elapsed_sec:.2f}s)"
                )
                continue

            entry = build_via_entry(file_name, file_size, inference_result)
            result_root[via_result_key(file_name, file_size)] = entry

            print(
                f"    -> 완료 HTTP {response.status_code} "
                f"regions={len(entry['regions'])} "
                f"({len(response.content)} bytes, 응답까지 {elapsed_sec:.2f}s)"
            )

    with paths.output_json.open("w", encoding="utf-8") as f:
        json.dump(result_root, f, ensure_ascii=False, indent=4)

    print(f"[완료] {len(result_root)}개 항목 저장: {paths.output_json}")
    print(f"[완료] 원본 응답 저장: {paths.response_jsonl}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="워크플로 API로 이미지 일괄 추론 후 VIA/JSONL 저장")
    p.add_argument(
        "--img-dir",
        type=Path,
        default="./dataset_sampled/images/",
        help=f"입력 이미지 디렉터리",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default="./result/",
        help=f"결과 저장 디렉터리",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(RunPaths(img_dir=args.img_dir, result_dir=args.result_dir))


if __name__ == "__main__":
    main()
