#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""읽기 전용 검색 API.

Qdrant 를 직접 노출하지 않는다. Qdrant REST 에는 컬렉션 삭제 API 가 그대로
들어 있어서, 그걸 터널로 열면 며칠짜리 적재를 한 번의 DELETE 로 잃을 수 있다.
여기는 검색 세 개와 /health 뿐이고 쓰기 경로가 아예 없다.

  POST /search        의미(벡터) 검색 -> 중복 제거 -> 리랭커 -> top N
  POST /search/batch  질의 여러 개를 한 번에 (임베딩/검색/리랭킹 모두 배치)
  POST /keyword       제목 키워드 검색 (벡터 미사용)

검색은 qdrant-client 를 직접 쓴다. langchain 의 similarity_search 는 서버측
timeout 을 넘길 방법이 없어 60초 기본값에 걸리고, 배치 질의도 지원하지 않는다.
(적재 경로인 post.py 는 langchain 을 그대로 쓴다.)

임베딩과 리랭킹 모두 서버 GPU 에서 하므로 클라이언트는 아무 의존성도 필요 없다.
"""

from __future__ import annotations

import html
import secrets
import threading
import time
import traceback

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client import models

from src import config as C
from src import dedupe as dd
from src import embedding, gpu

from .models import (BatchSearchRequest, BatchSearchResponse, Filters, Health,
                     Hit, KeywordRequest, SearchRequest, SearchResponse)

STATE: dict = {}
GPU_LOCK = threading.Lock()   # 임베딩과 리랭킹이 GPU 를 서로 밟지 않도록

TIMEOUT_HINT = (
    "Qdrant 검색이 제한시간 안에 끝나지 않았습니다. 보통 컬렉션 세그먼트가 많거나 "
    "스토리지가 느릴 때 납니다. rescore=false, hnsw_ef 낮추기, candidates 줄이기 "
    "순으로 시도해 보세요."
)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def clean(text: str | None) -> str | None:
    """OpenAlex 초록에는 &#13; &gt; 같은 HTML 엔티티와 줄바꿈이 섞여 있다."""
    if not text:
        return None
    return " ".join(html.unescape(text).split()) or None


def require_key(x_api_key: str = Header(default="")) -> None:
    if not secrets.compare_digest(x_api_key, STATE["api_key"]):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def to_filter(req: Filters, extra: list | None = None) -> models.Filter | None:
    must: list = list(extra or [])

    def match(key: str, value):
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    if req.year_from or req.year_to:
        must.append(models.FieldCondition(
            key="publication_year",
            range=models.Range(gte=req.year_from, lte=req.year_to)))
    if req.field:
        match("primary_topic__field__display_name", req.field)
    if req.subfield:
        match("primary_topic__subfield__display_name", req.subfield)
    if req.domain:
        match("primary_topic__domain__display_name", req.domain)
    if req.topic:
        match("primary_topic__display_name", req.topic)
    if req.source:
        match("primary_location__source__display_name", req.source)
    if req.type:
        match("type", req.type)
    if req.country:
        match("country_code", req.country)
    return models.Filter(must=must) if must else None


def to_item(payload: dict, *, vector_score: float | None = None) -> dict:
    """qdrant payload -> 내부 표현. 여기서 텍스트를 한 번만 정리한다."""
    return {
        "_payload": payload,
        "openalex_id": payload.get("openalex_id"),
        "doi": payload.get("doi"),
        "title": clean(payload.get("title")),
        "abstract": clean(payload.get(C.TEXT_FIELD)),
        "publication_year": payload.get("publication_year"),
        "source": payload.get("primary_location__source__display_name"),
        "topic": payload.get("primary_topic__display_name"),
        "field": payload.get("primary_topic__field__display_name"),
        "country": payload.get("country_code"),
        "type": payload.get("type"),
        "vector_score": vector_score,
    }


def rerank_text(item: dict) -> str:
    """리랭커에 넣을 문서 표현. 제목 + 초록."""
    title = item.get("title") or ""
    abstract = item.get("abstract") or ""
    return f"{title}\n{abstract}".strip()


def to_hits(items: list[dict], *, raw: bool = False) -> list[Hit]:
    hits = []
    for rank, item in enumerate(items, start=1):
        score = item.get("rerank_score")
        if score is None:
            score = item.get("vector_score")
        hits.append(Hit(
            rank=rank,
            score=None if score is None else float(score),
            vector_score=item.get("vector_score"),
            rerank_score=item.get("rerank_score"),
            duplicates=item.get("duplicates", 1),
            openalex_id=item.get("openalex_id"),
            doi=item.get("doi"),
            title=item.get("title"),
            abstract=item.get("abstract"),
            publication_year=item.get("publication_year"),
            source=item.get("source"),
            topic=item.get("topic"),
            field=item.get("field"),
            country=item.get("country"),
            type=item.get("type"),
            payload=item.get("_payload") if raw else None,
        ))
    return hits


# ---------------------------------------------------------------------------
# 검색 코어
# ---------------------------------------------------------------------------

def embed_queries(queries: list[str]) -> list[list[float]]:
    """질의 여러 개를 한 번의 forward 로 임베딩한다."""
    with GPU_LOCK:
        return STATE["embeddings"].embed_documents(queries)


def vector_search(queries: list[str], req: SearchRequest, fetch: int
                  ) -> list[list[dict]]:
    """질의별 후보 목록. 한 번의 배치 요청으로 처리한다."""
    vectors = embed_queries(queries)
    params = models.SearchParams(
        # ef 는 탐색 중 들고 다니는 후보 목록의 크기다. 뽑으려는 개수보다
        # 작으면 애초에 그만큼 채울 수가 없으므로 최소한 fetch 만큼 확보한다.
        hnsw_ef=max(req.hnsw_ef, fetch),
        quantization=models.QuantizationSearchParams(
            rescore=req.rescore, oversampling=req.oversampling),
    )
    query_filter = to_filter(req)

    responses = STATE["client"].query_batch_points(
        collection_name=C.COLLECTION,
        requests=[
            models.QueryRequest(
                query=vector, limit=fetch, filter=query_filter,
                params=params, with_payload=True, with_vector=False,
            )
            for vector in vectors
        ],
        timeout=C.SEARCH_TIMEOUT,      # langchain 경로에선 못 넘기던 값
    )
    return [
        [to_item(point.payload or {}, vector_score=float(point.score))
         for point in response.points]
        for response in responses
    ]


def refine(queries: list[str], candidate_lists: list[list[dict]],
           *, rerank: bool, dedupe: bool, limit: int
           ) -> tuple[list[list[dict]], list[int], bool]:
    """중복 제거 -> 리랭킹 -> 자르기. (결과, 제거된 수, 리랭킹 여부)"""
    removed = []
    for index, items in enumerate(candidate_lists):
        before = len(items)
        if dedupe:
            # 벡터 점수 순이므로 가장 좋은 것이 남는다
            candidate_lists[index] = dd.dedupe(items)
        removed.append(before - len(candidate_lists[index]))

    reranker = STATE.get("reranker")
    if not (rerank and reranker):
        for items in candidate_lists:
            del items[limit:]
        return candidate_lists, removed, False

    with GPU_LOCK:
        ranked = reranker.rerank_many(queries, candidate_lists,
                                      text_of=rerank_text, top_k=limit)
    return ranked, removed, True


def respond(query: str, mode: str, items: list[dict], *, began: float,
            found: int, removed: int, reranked: bool, raw: bool) -> SearchResponse:
    took = (time.perf_counter() - began) * 1000
    return SearchResponse(query=query, mode=mode, count=len(items),
                          took_ms=round(took, 1), candidates=found,
                          reranked=reranked, deduped=removed,
                          hits=to_hits(items, raw=raw))


# 서로 다른 분야를 훑어 HNSW 그래프의 여러 영역을 건드리게 고른 질의들
WARMUP_TEXTS = [
    "machine learning model evaluation",
    "cancer treatment clinical trial",
    "climate change carbon emissions",
    "quantum computing algorithm",
    "protein structure prediction",
    "semiconductor device fabrication",
]


def warmup(count: int) -> None:
    """캐시를 데운다. 백그라운드 스레드에서 부른다.

    콜드 상태에서는 세그먼트와 벡터를 디스크에서 읽느라 첫 질의가 수십 초 걸린다.
    미리 훑어 두면 실제 사용자의 첫 질의가 그 대가를 치르지 않는다.
    """
    texts = WARMUP_TEXTS[:max(0, count)]
    if not texts:
        return
    began = time.perf_counter()
    print(f"[warmup] 캐시를 데우는 중 ({len(texts)}개 질의, 백그라운드)")
    for text in texts:
        try:
            request = SearchRequest(query=text, limit=5, candidates=20, rerank=False)
            vector_search([text], request, 20)
        except Exception as exc:                  # 워밍업 실패는 치명적이지 않다
            print(f"[warmup] 건너뜀: {type(exc).__name__}: {str(exc)[:120]}")
            return
    print(f"[warmup] 완료 ({time.perf_counter() - began:.0f}초). "
          f"이제 첫 질의가 빠릅니다")


# ---------------------------------------------------------------------------
# 앱
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenAlex Vector Search",
    version="1.2",
    description="읽기 전용 의미 검색 + 제목 키워드 검색. 쓰기/삭제 기능은 없습니다.",
)


@app.exception_handler(Exception)
def on_error(request: Request, exc: Exception) -> JSONResponse:
    """맨 500 대신 실제 예외를 돌려준다.

    API 키로 보호되는 내부용이라 예외 내용을 숨길 이유가 없고,
    터널 너머에서 디버깅하려면 이게 훨씬 빠르다.
    """
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    detail = str(exc)[:2000]
    body = {"error": type(exc).__name__, "detail": detail, "path": request.url.path}
    if "timed out" in detail.lower() or "timeout" in detail.lower():
        body["hint"] = TIMEOUT_HINT
    return JSONResponse(status_code=500, content=body)


@app.get("/health", response_model=Health)
def health() -> Health:
    info = STATE["client"].get_collection(C.COLLECTION)
    reranker = STATE.get("reranker")
    return Health(
        status=str(info.status),
        points=info.points_count or 0,
        indexed=info.indexed_vectors_count or 0,
        segments=info.segments_count or 0,
        device=STATE["device"],
        embedding_model=C.MODEL_NAME,
        reranker_model=reranker.model_name if reranker else None,
    )


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_key)])
def search(req: SearchRequest) -> SearchResponse:
    began = time.perf_counter()
    fetch = max(req.candidates, req.limit) if req.rerank else req.limit

    candidates = vector_search([req.query], req, fetch)
    found = len(candidates[0])
    results, removed, reranked = refine(
        [req.query], candidates, rerank=req.rerank, dedupe=req.dedupe,
        limit=req.limit)

    if C.RELEASE_CACHE:
        gpu.release(STATE["device"])   # 옆에서 도는 모델에 VRAM 을 돌려준다
    return respond(req.query, "vector", results[0], began=began, found=found,
                   removed=removed[0], reranked=reranked, raw=req.raw)


@app.post("/search/batch", response_model=BatchSearchResponse,
          dependencies=[Depends(require_key)])
def search_batch(req: BatchSearchRequest) -> BatchSearchResponse:
    """질의 여러 개를 한 번에.

    각 단계가 전부 배치로 묶인다.
      임베딩  질의 N개를 한 번의 forward 로
      검색    query_batch_points 로 왕복 1회. 서버가 내부적으로 병렬 처리하고
              세그먼트와 캐시를 재사용하므로 N번 따로 부르는 것보다 훨씬 빠르다
      리랭킹  모든 (질의, 문서) 쌍을 한 번의 forward 로
    """
    began = time.perf_counter()
    fetch = max(req.candidates, req.limit) if req.rerank else req.limit

    candidates = vector_search(req.queries, req, fetch)
    found = [len(items) for items in candidates]
    results, removed, reranked = refine(
        req.queries, candidates, rerank=req.rerank, dedupe=req.dedupe,
        limit=req.limit)

    if C.RELEASE_CACHE:
        gpu.release(STATE["device"])

    took = round((time.perf_counter() - began) * 1000, 1)
    return BatchSearchResponse(
        count=len(req.queries),
        took_ms=took,
        results=[
            SearchResponse(query=query, mode="vector", count=len(items),
                           took_ms=took, candidates=found[i],
                           reranked=reranked, deduped=removed[i],
                           hits=to_hits(items, raw=req.raw))
            for i, (query, items) in enumerate(zip(req.queries, results))
        ],
    )


@app.post("/keyword", response_model=SearchResponse, dependencies=[Depends(require_key)])
def keyword(req: KeywordRequest) -> SearchResponse:
    """제목에 특정 단어/구문이 든 문서를 찾는다. 벡터를 쓰지 않는다.

    payload 인덱스 조회라 벡터 검색보다 훨씬 빠르다. 대신 의미가 아니라
    표기가 기준이다.
    """
    began = time.perf_counter()
    condition = models.FieldCondition(key="title",
                                      match=_text_match(req.query, req.phrase))
    # 중복/재정렬을 감안해 넉넉히 훑는다 (인덱스 조회라 비용이 작다)
    fetch = min(req.limit * 10, 500) if (req.dedupe or req.rerank) else req.limit

    points, _ = STATE["client"].scroll(
        collection_name=C.COLLECTION,
        scroll_filter=to_filter(req, extra=[condition]),
        limit=fetch, with_payload=True, with_vectors=False)

    items = [to_item(point.payload or {}) for point in points]
    found = len(items)
    results, removed, reranked = refine(
        [req.query], [items], rerank=req.rerank, dedupe=req.dedupe,
        limit=req.limit)

    if C.RELEASE_CACHE and reranked:
        gpu.release(STATE["device"])
    return respond(req.query, "keyword", results[0], began=began, found=found,
                   removed=removed[0], reranked=reranked, raw=req.raw)


def _text_match(text: str, phrase: bool):
    """구문 검색은 비교적 최신 qdrant-client/서버에서만 지원한다."""
    if phrase and hasattr(models, "MatchPhrase"):
        return models.MatchPhrase(phrase=text)
    return models.MatchText(text=text)


# ---------------------------------------------------------------------------
# 기동
# ---------------------------------------------------------------------------

def build(device: str = "auto", api_key: str = "", *, with_reranker: bool = True,
          reranker_device: str | None = None, tf32: bool = True,
          memory_gib: float | None = None) -> str:
    """모델과 Qdrant 연결을 올리고 API 키를 확정한다. 키를 반환."""
    from src import collection as coll
    from src.store import check_collection

    key = api_key or C.API_KEY or secrets.token_urlsafe(24)
    STATE["api_key"] = key

    embeddings, resolved = embedding.build(device, batch_size=8, tf32=tf32,
                                           memory_gib=memory_gib)
    STATE["embeddings"] = embeddings
    STATE["device"] = resolved
    print(f"[model] embedding {C.MODEL_NAME} on {resolved}")

    client = coll.client(timeout=C.SEARCH_TIMEOUT)
    check_collection(client)
    STATE["client"] = client

    if with_reranker:
        from src.reranker import Reranker
        model = Reranker(device=reranker_device or device, tf32=tf32,
                         memory_gib=memory_gib)
        STATE["reranker"] = model
        print(f"[model] reranker  {model.model_name} on {model.device} "
              f"(max_len={model.max_length}, batch={model.batch_size})")
    else:
        STATE["reranker"] = None
        print("[model] reranker  비활성")

    return key
