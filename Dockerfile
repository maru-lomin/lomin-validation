# syntax=docker/dockerfile:1
# 빌드 시에만 외부 네트워크(Python 패키지·uv 바이너리)가 필요합니다.
# 완성된 이미지는 폐쇄망에서 docker load 후 오프라인 실행 가능합니다.
# 소스 코드는 이미지에 넣지 않고, 실행 시 호스트 프로젝트를 /app 에 마운트합니다.
FROM python:3.13-slim-bookworm

# Matplotlib 등 바이너리 휠이 기대하는 최소 런타임 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libfreetype6 \
        libpng16-16 \
        fontconfig \
        fonts-nanum \
        fonts-noto-cjk \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 비root(--user) 실행 시 HOME 미설정이면 Matplotlib이 /.config 에 쓰려 Permission denied 발생
ENV MPLCONFIGDIR=/tmp/mplconfig \
    XDG_CONFIG_HOME=/tmp/.config

# ghcr.io/astral-sh/uv 이미지에서 COPY하는 방식도 가능하나, 방화벽 환경에서는 pip가 더 통과하기 쉬움
RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 의존성만 이미지에 고정. 런타임 코드는 호스트 → /app 마운트
WORKDIR /install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

ENV PATH="/install/.venv/bin:$PATH" \
    VIRTUAL_ENV="/install/.venv"

WORKDIR /data
# 인자 없이 실행 시 기본 동작; 실제 워크플로는 docker_run.sh 사용
CMD ["python", "--version"]
