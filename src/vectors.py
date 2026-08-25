#!/usr/bin/env python
# -*- coding: utf-8 -*-
""".f16 파일 입출력.

row 는 전부 같은 크기(DIM x itemsize)라 색인이 필요 없다.
N번째 row 의 위치는 그냥 N * ROW_BYTES 다.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np

from . import config as C


def open_reader(path: Path, start_row: int) -> BinaryIO:
    handle = path.open("rb")
    handle.seek(start_row * C.ROW_BYTES)
    return handle


def read_rows(handle: BinaryIO, count: int) -> np.ndarray:
    """열려 있는 .f16 핸들에서 count 개 row 를 읽어 float32 배열로."""
    raw = handle.read(count * C.ROW_BYTES)
    got = len(raw) // C.ROW_BYTES
    if got < count:
        msg = f"벡터 부족: {count}개 요청, {got}개만 남음"
        raise EOFError(msg)
    return np.frombuffer(raw, dtype=C.DTYPE).reshape(count, C.DIM).astype(np.float32)


def append_block(path: Path, block: np.ndarray) -> None:
    """배치 하나를 파일 끝에 덧붙인다. 파일 크기가 곧 체크포인트가 된다."""
    if block.dtype != C.DTYPE or block.shape[1] != C.DIM:
        msg = f"블록 형식 불일치: {block.dtype} {block.shape}"
        raise ValueError(msg)
    with path.open("ab") as handle:
        handle.write(block.tobytes(order="C"))
        handle.flush()


def zero_block(rows: int) -> np.ndarray:
    """임베딩 대상이 아닌 줄의 자리를 채우는 영벡터 블록."""
    return np.zeros((rows, C.DIM), dtype=C.DTYPE)


def find_zero_row(block: np.ndarray) -> int | None:
    """영벡터인 첫 row 의 인덱스. 없으면 None.

    텍스트가 있는 줄인데 벡터가 비었다면 .f16 과 jsonl 이 어긋난 것이다.
    코사인 컬렉션은 영벡터를 거부하므로 그냥 두면 뒤에서 더 헷갈리게 터진다.
    """
    if not len(block):
        return None
    nonzero = block.any(axis=1)
    if nonzero.all():
        return None
    return int((~nonzero).argmax())
