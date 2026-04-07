#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-evaluator:latest}"

docker build \
  -t "${IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

echo "Built ${IMAGE}"
echo "폐쇄망 전달 예: docker save ${IMAGE} | gzip > evaluator-image.tar.gz"
