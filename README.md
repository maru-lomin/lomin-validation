Lomin workflow api의 **추론 결과(VIA 형식 JSON)**와 **정답(gt.json)**을 맞춰, **CER(Character Error Rate)**과 KV 영역의 **value 병합 bbox IoU**를 계산하는 도구 모음입니다.

## 구성

| 구성 요소 | 설명 |
|-----------|------|
| `request_api.py` | 워크플로 API에 이미지를 일괄 전송하고, 응답을 VIA 형식 `result.json`·원본 `response.jsonl`로 저장합니다. |
| `evaluate.py` | `gt.json`과 `result.json`을 비교해 지표를 산출하고, 상세 텍스트·Excel·요약 텍스트를 씁니다. |
| `visualize.py` | VIA JSON의 KV(key/value) 영역을 이미지 위에 그려 `images_kv` 등에 PNG로 저장합니다. |
| `evaluation/` | 메트릭·VIA 로딩·리포트 작성 로직 |

데이터셋은 보통 `{데이터셋명}/gt.json`, `{데이터셋명}/images/` 구조를 가정합니다.

## 요구 사항

- **Python** 3.13 이상
- **패키지 관리**: [uv](https://docs.astral.sh/uv/) 권장 (`pyproject.toml` / `uv.lock` 기준 동일 환경)
- **Docker** (선택): 폐쇄망 배포용 이미지 빌드·실행 시

## 설치

프로젝트 루트에서:

```bash
uv sync
```

가상환경 없이 한 번 실행할 때는 `uv run <스크립트>` 형태를 쓰면 됩니다.

## 실행 (로컬)

프로젝트 루트를 현재 디렉터리로 두고 경로를 맞춥니다.

**1) API 추론 → `result/`**

```bash
uv run request_api.py --img-dir dataset/images --result-dir result
```

**2) 평가**

```bash
uv run evaluate.py \
  --gt-json dataset/gt.json \
  --result-json result/result.json \
  --detail-txt result/detail.txt \
  --detail-xlsx result/detail.xlsx \
  --summary-txt result/summary.txt
```

**3) 시각화 (선택)**

```bash
uv run visualize.py \
  --via-json result/result.json \
  --images-dir dataset/images \
  --output-dir dataset/images_kv
```

`evaluate.py` / `visualize.py`의 인자 기본값은 코드 및 `evaluation/cli.py`에 정의되어 있으므로, 데이터셋 이름에 맞게 `--gt-json`, `--images-dir` 등을 바꿉니다.

### API 설정

`request_api.py` 상단의 `WORKFLOW_URL`과 `workflow_payload()` 안의 `params`(API 키 등)는 사용 중인 환경에 맞게 수정해야 합니다.

## Docker

**빌드** (`evaluator:latest` 태그):

```bash
./docker_build.sh
```

**실행**: 프로젝트 루트에서 `./docker_run.sh`를 실행합니다. 현재 디렉터리가 호스트 프로젝트 루트로 컨테이너의 `/data`에 마운트되고, 비 root 사용자(`--user`로 호스트 UID/GID)로 동작합니다.

- 사용할 데이터셋 상위 디렉터리는 환경 변수 `DATASET_DIR`으로 지정합니다 (기본값은 스크립트 내 설정).

```bash
DATASET_DIR=./dataset ./docker_run.sh
```

스크립트는 `request_api.py`와 `evaluate.py`를 순서대로 호출합니다. 시각화 단계는 주석 처리되어 있으며, 필요하면 `docker_run.sh` 안의 `visualize.py` 호출을 풀고 경로를 `/data/...` 형태로 맞춥니다.

**경로 주의**: 컨테이너 안에서는 `uv run --directory /app` 때문에 작업 디렉터리가 `/app`이 됩니다. 결과를 호스트 프로젝트에 쓰려면 **`--result-dir` 등은 `/data/result`처럼 절대 경로**로 두는 것이 안전합니다 (`docker_run.sh`에 반영됨).

오프라인 전달 예:

```bash
docker save evaluator:latest | gzip > evaluator-image.tar.gz
```

## 산출물

- `result/result.json`: VIA 형식 추론 결과
- `result/response.jsonl`: API 원본 응답(줄 단위 JSON)
- `result/detail.txt`, `result/detail.xlsx`, `result/summary.txt`: 평가 상세·요약
