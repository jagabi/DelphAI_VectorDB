#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""jsonl 한 줄 -> (row, text) / point id / qdrant payload.

preprocess 와 post 가 "이 줄이 임베딩 대상인가"를 똑같이 판단해야 .f16 과 jsonl 의
줄 정렬이 유지된다. 그래서 판단 로직을 이 파일 하나에만 둔다.
"""

from __future__ import annotations

import json
import uuid

from . import config as C

# orjson 이 있으면 쓴다. 9천만 줄을 파싱하므로 2~5배 차이가 난다.
try:
    import orjson

    def _loads(text: str):
        return orjson.loads(text)
except ImportError:
    def _loads(text: str):
        return json.loads(text)


UUID_NAMESPACE = uuid.NAMESPACE_URL


def parse_line(line: str) -> tuple[dict | None, str]:
    """
    (row, text) 반환.

    text 가 빈 문자열이면 '임베딩 대상 아님' 이라는 뜻이고,
    preprocess 는 영벡터를, post 는 스킵을 선택한다.
    """
    line = line.strip()
    if not line:
        return None, ""
    try:
        row = _loads(line)
    except Exception:
        return None, ""
    if not row.get("id"):
        return None, ""

    text = (row.get(C.TEXT_FIELD) or "").strip()
    if not text and C.FALLBACK_TO_TITLE:
        text = (row.get("title") or "").strip()
    return row, text


def point_id(openalex_id: str) -> str:
    """결정적 id. 몇 번을 다시 올려도 같은 점을 덮어쓴다."""
    return str(uuid.uuid5(UUID_NAMESPACE, openalex_id))


def to_payload(row: dict) -> dict:
    """raw json -> qdrant payload. 'id' -> 'openalex_id', None/빈값 제거.

    인덱스 유무와 무관하게 모든 필드가 저장된다. Qdrant 는 payload 에 스키마가 없다.
    """
    return {
        C.KEY_MAP.get(key, key): value
        for key, value in row.items()
        if value is not None and value != ""
    }
