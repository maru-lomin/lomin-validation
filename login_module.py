"""Optional Lomin account login (sync `requests`) for Bearer-style workflow calls.

Patterns from `reference/lomin_client.py` / `reference/lomin_schema.py`, without
internal `koreanre` package dependencies.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

LOGIN_API_PATH = "/cognition-api/api/v2/account/login"


def base_url_from_workflow_url(workflow_url: str) -> str:
    """Origin (scheme + host[:port]) of the workflow URL for login on the same host."""
    parsed = urlparse(workflow_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"workflow URL must be absolute with scheme and host: {workflow_url!r}"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def _parse_login_authorization(data: dict[str, Any]) -> str:
    inner = data.get("response")
    if not isinstance(inner, dict):
        raise ValueError("login JSON missing 'response' object")
    token_type = str(inner.get("type", "Bearer")).strip()
    token = inner.get("token")
    if not token:
        raise ValueError("login response missing token")
    return f"{token_type} {token}"


def login_authorization_header(
    *,
    base_url: str,
    email: str,
    password: str,
    timeout: float,
    session: requests.Session | None = None,
) -> str:
    """POST login; return value suitable for ``Authorization`` header."""
    url = f"{base_url.rstrip('/')}{LOGIN_API_PATH}"
    sess = session or requests.Session()
    close_sess = session is None
    try:
        response = sess.post(
            url,
            json={"email": email, "password": password},
            headers={"accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("login response is not a JSON object")
        return _parse_login_authorization(body)
    finally:
        if close_sess:
            sess.close()


class CachedLominAuth:
    """Caches one ``Authorization`` value; call :meth:`invalidate` after HTTP 401."""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        timeout: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._email = email
        self._password = password
        self._timeout = timeout
        self._authorization: str | None = None

    def invalidate(self) -> None:
        self._authorization = None

    def authorization_header_value(self) -> str:
        if self._authorization is None:
            self._authorization = login_authorization_header(
                base_url=self._base_url,
                email=self._email,
                password=self._password,
                timeout=self._timeout,
            )
        return self._authorization
