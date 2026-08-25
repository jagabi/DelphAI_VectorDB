#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""현재 환경에 설치된 버전으로 requirements 를 찍어낸다.

pip freeze 는 환경의 모든 패키지를 쏟아내서 그대로 쓰기 어렵다.
이 스크립트는 이 프로젝트가 실제로 쓰는 것만, 설치된 버전으로 출력한다.

    python scripts/freeze.py                 # 화면에 출력
    python scripts/freeze.py > requirements.txt
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

GROUPS: list[tuple[str, list[str]]] = [
    ("공통", ["numpy", "tqdm", "orjson"]),
    ("전처리", ["pyarrow"]),
    ("임베딩 / 리랭커", [
        "torch",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "langchain-huggingface",
        "langchain-core",
    ]),
    ("Qdrant", ["langchain-qdrant", "qdrant-client", "pydantic"]),
    ("API", ["fastapi", "uvicorn", "starlette"]),
]

HEADER = """\
# 이 파일은 scripts/freeze.py 로 생성되었습니다.
# python >= {py}
#
# GPU 머신이면 torch 를 먼저 CUDA 빌드로 설치한 뒤 이 파일을 적용하세요.
# 그냥 설치하면 CPU 전용 휠이 잡힐 수 있습니다.
#   https://pytorch.org/get-started/locally/
"""


def main() -> int:
    missing: list[str] = []
    print(HEADER.format(py=".".join(map(str, sys.version_info[:3]))))

    for title, packages in GROUPS:
        print(f"# ---------------------------------------------------- [{title}]")
        for name in packages:
            try:
                print(f"{name}=={version(name)}")
            except PackageNotFoundError:
                missing.append(name)
                print(f"# {name}  (미설치)")
        print()

    if missing:
        print("# 미설치:", ", ".join(missing), file=sys.stderr)

    _warn_if_cpu_torch()
    return 0


def _warn_if_cpu_torch() -> None:
    """CPU 전용 torch 가 섞여 들어가면 여기서 걸러낸다.

    pip 가 pytorch 인덱스 대신 PyPI 에서 CPU 휠을 집어가는 일이 흔하다.
    그대로 requirements 에 박히면 다음 설치 때도 CPU 로 굳는다.
    """
    try:
        import torch
    except ImportError:
        return
    if torch.version.cuda:
        return
    print("", file=sys.stderr)
    print(f"# [warn] torch={torch.__version__} 는 CPU 전용 빌드입니다.",
          file=sys.stderr)
    print("#        GPU 머신이라면 다시 설치한 뒤 freeze 를 다시 돌리세요:", file=sys.stderr)
    print("#          pip install --force-reinstall torch "
          "--index-url https://download.pytorch.org/whl/cu124", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
