# 내부망 설치

## uv

`packages/uv`의 Windows용 zip을 압축 해제한 뒤 `uv.exe`를 PATH에 추가합니다.

```powershell
Expand-Archive -Path packages\uv\uv-x86_64-pc-windows-msvc.zip -DestinationPath packages\uv\extract -Force
$env:Path = "$PWD\packages\uv\extract;" + $env:Path
```

## Python (uv-python)

`packages/uv-python`의 Windows용 standalone Python tarball을 압축 해제한 뒤 `UV_PYTHON`으로 지정합니다.

```powershell
New-Item -ItemType Directory -Force -Path packages\uv-python\extract | Out-Null
tar -xzf packages/uv-python/cpython-3.13.14-x86_64-pc-windows-msvc-install_only.tar.gz -C packages/uv-python/extract
$env:UV_PYTHON = "$PWD\packages\uv-python\extract\python\python.exe"
```

## Python 패키지 (wheelhouse)

`packages/wheelhouse`의 wheel 파일만 사용해 오프라인 설치합니다.

```powershell
uv venv
uv pip install --no-index --find-links packages/wheelhouse requests pandas notebook jupyter openpyxl
```
