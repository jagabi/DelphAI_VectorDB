#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""요청/응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src import config as C


class Filters(BaseModel):
    """payload 인덱스가 걸린 필드만 담는다 (src/config.py 참고).

    인덱스 없는 필드로 걸러도 동작은 하지만 전수 스캔이라 매우 느리다.
    """

    year_from: int | None = None
    year_to: int | None = None
    field: str | None = Field(None, description='분야, 예: "Computer Science"')
    subfield: str | None = Field(None, description='예: "Artificial Intelligence"')
    domain: str | None = Field(None, description='예: "Physical Sciences"')
    topic: str | None = Field(None, description="세부 토픽 이름")
    source: str | None = Field(None, description="저널/저장소 이름")
    type: str | None = Field(None, description="예: article, preprint")
    country: str | None = Field(None, description="예: KR, US")


class SearchRequest(Filters):
    """의미(벡터) 검색.

    벡터로 candidates 개를 뽑고 -> 중복 제거 -> 리랭커로 다시 정렬 -> limit 개 반환.
    """

    query: str = Field(..., min_length=1, max_length=4000)
    limit: int = Field(C.SEARCH_LIMIT, ge=1, le=200, description="최종 반환 개수")
    candidates: int = Field(C.SEARCH_CANDIDATES, ge=1, le=1000,
                            description="리랭킹 전에 벡터로 뽑을 후보 수")

    rerank: bool = Field(True, description="cross-encoder 재정렬. 서버에 모델이 없으면 무시")
    dedupe: bool = Field(True, description="같은 논문의 중복 레코드를 합침")

    hnsw_ef: int = Field(C.SEARCH_HNSW_EF, ge=16, le=1024,
                         description="클수록 정확하고 느림")
    rescore: bool = Field(C.SEARCH_RESCORE,
                          description="원본 float32 로 후보를 다시 잼. "
                                      "리랭커를 쓰면 불필요하고 느리기만 하다")
    oversampling: float = Field(2.0, ge=1.0, le=10.0)


class KeywordRequest(Filters):
    """제목 키워드 검색. 벡터를 쓰지 않는다.

    title 에 걸린 전문 인덱스로 단어(또는 구문)를 포함하는 문서를 찾는다.
    """

    query: str = Field(..., min_length=1, max_length=500, description="제목에 포함될 단어")
    limit: int = Field(C.KEYWORD_LIMIT, ge=1, le=200)
    phrase: bool = Field(False, description="단어 나열이 아니라 구문 전체로 일치")
    dedupe: bool = Field(True)
    rerank: bool = Field(False, description="찾은 것들을 질의 기준으로 재정렬")


class Hit(BaseModel):
    rank: int
    score: float = Field(..., description="리랭킹했으면 리랭커 점수, 아니면 벡터 유사도")
    vector_score: float | None = Field(None, description="벡터 검색 코사인 점수")
    rerank_score: float | None = None
    duplicates: int = Field(1, description="합쳐진 동일 논문 레코드 수 (1이면 중복 없음)")

    openalex_id: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    source: str | None = None
    topic: str | None = None
    field: str | None = None
    country: str | None = None
    type: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str = Field(..., description="vector | keyword")
    count: int
    took_ms: float
    candidates: int = Field(0, description="후보로 뽑았던 개수")
    reranked: bool = False
    deduped: int = Field(0, description="중복으로 제거된 개수")
    hits: list[Hit]


class Health(BaseModel):
    status: str
    points: int
    indexed: int
    segments: int
    device: str
    embedding_model: str
    reranker_model: str | None = None
