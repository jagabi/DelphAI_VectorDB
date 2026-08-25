#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""중복 문서 제거.

OpenAlex 에는 같은 논문이 여러 레코드로 들어있다. 출판사 원본과 기관 저장소 사본이
따로 등재되는 식이라, **DOI 도 openalex_id 도 서로 다르다.** 예:

    Constructing a cohesive pattern ...   PeerJ Computer Science   10.7717/peerj-cs.626
    Constructing a cohesive pattern ...   Greater South Info Sys   10.60692/wx650-y1416

그래서 id 기반 중복 제거는 통하지 않는다. 제목을 정규화한 것을 키로 쓴다.
제목이 없으면 초록 앞부분으로 대신한다.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Callable, Iterable

_NON_WORD = re.compile(r"[^0-9a-z가-힣]+")


def normalize(text: str | None) -> str:
    """대소문자, 구두점, 공백, HTML 엔티티, 유니코드 표기 차이를 지운다."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).lower()
    return _NON_WORD.sub(" ", text).strip()


def key_of(item: dict, *, title_key: str = "title",
           abstract_key: str = "abstract") -> str:
    """중복 판정 키. 제목 우선, 없으면 초록 앞 200자."""
    title = normalize(item.get(title_key))
    if len(title) >= 12:            # 너무 짧은 제목은 우연히 겹칠 수 있다
        return "t:" + title
    abstract = normalize(item.get(abstract_key))
    if abstract:
        return "a:" + abstract[:200]
    return "i:" + str(item.get("openalex_id") or id(item))


def dedupe(items: Iterable[dict], *, key: Callable[[dict], str] = key_of,
           count_field: str = "duplicates") -> list[dict]:
    """먼저 나온 것을 남긴다. 호출 전에 원하는 순서로 정렬해 둘 것.

    남은 항목에는 함께 묶인 사본 수를 count_field 로 붙인다 (1 이면 중복 없음).
    """
    seen: dict[str, dict] = {}
    out: list[dict] = []
    for item in items:
        k = key(item)
        first = seen.get(k)
        if first is None:
            seen[k] = item
            item[count_field] = 1
            out.append(item)
        else:
            first[count_field] += 1
    return out
