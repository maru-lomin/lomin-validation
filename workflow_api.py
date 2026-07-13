from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

WORKFLOW_URL = ""
API_KEY = ""
DATA_PATH = ""
AUTH_EMAIL = ""
AUTH_PASSWORD = ""
AUTH_BASE_URL = ""
HTTP_TIMEOUT = 300.0

LOGIN_API_PATH = "/cognition-api/api/v2/account/login"

CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def base_url_from_workflow_url(workflow_url: str) -> str:
    parsed = urlparse(workflow_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"workflow URL must be absolute with scheme and host: {workflow_url!r}"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def login_authorization_header(
    *,
    base_url: str,
    email: str,
    password: str,
    timeout: float,
) -> str:
    url = f"{base_url.rstrip('/')}{LOGIN_API_PATH}"
    response = requests.post(
        url,
        json={"email": email, "password": password},
        headers={"accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("login response is not a JSON object")
    inner = body.get("response")
    if not isinstance(inner, dict):
        raise ValueError("login JSON missing 'response' object")
    token_type = str(inner.get("type", "Bearer")).strip()
    token = inner.get("token")
    if not token:
        raise ValueError("login response missing token")
    return f"{token_type} {token}"


def workflow_payload(*, api_key: str) -> dict[str, str]:
    return {
        "params": json.dumps(
            {
                "api_key": api_key,
                "api_option": {
                    "async_mode": False,
                    "timeout": HTTP_TIMEOUT,
                },
            },
            ensure_ascii=False,
        ),
    }


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def post_workflow(
    data_path: Path,
    *,
    url: str,
    api_key: str,
    authorization: str | None,
) -> requests.Response:
    payload = workflow_payload(api_key=api_key)
    headers = {"Authorization": authorization} if authorization else None
    with data_path.open("rb") as f:
        files = [("file", (data_path.name, f, content_type_for(data_path)))]
        return requests.post(
            url,
            data=payload,
            files=files,
            timeout=HTTP_TIMEOUT,
            headers=headers,
        )


def main() -> None:
    if not WORKFLOW_URL.strip():
        raise SystemExit("WORKFLOW_URL is required")
    if not API_KEY.strip():
        raise SystemExit("API_KEY is required")
    if not DATA_PATH.strip():
        raise SystemExit("DATA_PATH is required")

    data_path = Path(DATA_PATH)
    if not data_path.is_file():
        raise SystemExit(f"DATA_PATH is not a file: {data_path}")

    authorization: str | None = None
    email = AUTH_EMAIL.strip()
    password = AUTH_PASSWORD
    if email or password:
        if not (email and password):
            raise SystemExit("AUTH_EMAIL and AUTH_PASSWORD must both be set")
        auth_base = AUTH_BASE_URL.strip() or base_url_from_workflow_url(WORKFLOW_URL)
        login_timeout = min(60.0, HTTP_TIMEOUT)
        authorization = login_authorization_header(
            base_url=auth_base,
            email=email,
            password=password,
            timeout=login_timeout,
        )

    response = post_workflow(
        data_path,
        url=WORKFLOW_URL.strip(),
        api_key=API_KEY.strip(),
        authorization=authorization,
    )

    try:
        body: Any = response.json()
        print(json.dumps(body, ensure_ascii=False))
    except (ValueError, json.JSONDecodeError):
        sys.stdout.write(response.text)
        if response.text and not response.text.endswith("\n"):
            sys.stdout.write("\n")

    raise SystemExit(0 if response.ok else 1)


if __name__ == "__main__":
    main()
