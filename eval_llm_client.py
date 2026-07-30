"""OpenAI 호환 API: /v1/models 목록, /v1/chat/completions (vLLM 등)."""

from __future__ import annotations

from typing import Any

import requests


def get_model_catalog(base_url: str, *, timeout: float) -> tuple[list[str], Any]:
    """GET /v1/models 한 번 호출 후 (id 목록, 원본 JSON).

    id는 서버가 준 `data` 배열 순서를 유지합니다.
    """
    url = base_url.rstrip("/") + "/v1/models"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    mids: list[str] = []
    if isinstance(data, dict):
        for item in data.get("data") or []:
            mid = item.get("id")
            if isinstance(mid, str) and mid.strip():
                mids.append(mid.strip())
    return mids, data


def first_model_id_or_raise(base_url: str, *, timeout: float) -> str:
    """목록의 첫 model id. 없으면 RuntimeError."""
    mids, data = get_model_catalog(base_url, timeout=timeout)
    if not mids:
        raise RuntimeError(f"/v1/models 에서 model id를 찾지 못함: {data!r}")
    return mids[0]


def chat_completion(
    *,
    base_url: str,
    model: str,
    user_content: str,
    timeout: float,
    max_tokens: int = 2048,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Qwen3 / vLLM: 템플릿 인자로만 인식됨 (최상위 enable_thinking 무시)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    resp = r.json()
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices in API response: {resp!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    return content
