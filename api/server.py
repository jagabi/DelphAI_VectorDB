#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""읽기 전용 검색 API.

Qdrant 를 직접 노출하지 않는다. Qdrant REST 에는 컬렉션 삭제 API 가 그대로
들어 있어서, 그걸 터널로 열면 며칠짜리 적재를 한 번의 DELETE 로 잃을 수 있다.
여기는 검색 두 개와 /health 뿐이고 쓰기 경로가 아예 없다.

  POST /search    의미(벡터) 검색 -> 중복 제거 -> 리랭커 -> top N
  POST /keyword   제목 키워드 검색 (벡터 미사용)

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
from src import embedding, gpu, store

from .models import (Filters, Health, Hit, KeywordRequest, SearchRequest,
                     SearchResponse)

STATE: dict = {}
GPU_LOCK = threading.Lock()   # 임베딩과 리랭킹이 GPU 를 서로 밟지 않도록


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
    return f"{title}\n{abstract}".strip() or title or abstract


def to_hits(items: list[dict]) -> list[Hit]:
    hits = []
    for rank, item in enumerate(items, start=1):
        score = item.get("rerank_score")
        if score is None:
            score = item.get("vector_score") or 0.0
        hits.append(Hit(
            rank=rank,
            score=float(score),
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
        ))
    return hits


def maybe_rerank(query: str, items: list[dict], *, want: bool, top_k: int) -> bool:
    """리랭킹을 수행했으면 True. 모델이 없으면 조용히 건너뛴다."""
    reranker = STATE.get("reranker")
    if not (want and reranker and items):
        del items[top_k:]
        return False
    with GPU_LOCK:
        ranked = reranker.rerank(query, items, text_of=rerank_text, top_k=top_k)
    items[:] = ranked
    return True


# ---------------------------------------------------------------------------
# 앱
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenAlex Vector Search",
    version="1.1",
    description="읽기 전용 의미 검색 + 제목 키워드 검색. 쓰기/삭제 기능은 없습니다.",
)


TIMEOUT_HINT = (
    "Qdrant 검색이 제한시간 안에 끝나지 않았습니다. 보통 컬렉션 세그먼트가 많거나 "
    "스토리지가 느릴 때 납니다. rescore=false, hnsw_ef 낮추기, candidates 줄이기 "
    "순으로 시도해 보세요."
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
    info = STATE["store"].client.get_collection(C.COLLECTION)
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
    params = models.SearchParams(
        hnsw_ef=req.hnsw_ef,
        quantization=models.QuantizationSearchParams(
            rescore=req.rescore, oversampling=req.oversampling),
    )
    # 리랭킹을 할 거면 후보를 넉넉히, 아니면 필요한 만큼만
    fetch = max(req.candidates, req.limit) if req.rerank else req.limit

    began = time.perf_counter()
    with GPU_LOCK:
        results = STATE["store"].similarity_search_with_score(
            req.query, k=fetch, filter=to_filter(req), search_params=params)

    items = [to_item(doc.metadata | {C.TEXT_FIELD: doc.page_content},
                     vector_score=float(score))
             for doc, score in results]

    found = len(items)
    if req.dedupe:
        items = dd.dedupe(items)          # 벡터 점수 순이므로 가장 좋은 것이 남는다
    removed = found - len(items)

    reranked = maybe_rerank(req.query, items, want=req.rerank, top_k=req.limit)
    took = (time.perf_counter() - began) * 1000
    if C.RELEASE_CACHE:
        gpu.release(STATE["device"])   # 옆에서 도는 모델에 VRAM 을 돌려준다

    return SearchResponse(query=req.query, mode="vector", count=len(items),
                          took_ms=round(took, 1), candidates=found,
                          reranked=reranked, deduped=removed, hits=to_hits(items))


@app.post("/keyword", response_model=SearchResponse, dependencies=[Depends(require_key)])
def keyword(req: KeywordRequest) -> SearchResponse:
    """제목에 특정 단어/구문이 든 문서를 찾는다. 벡터를 쓰지 않는다."""
    match = _text_match(req.query, req.phrase)
    condition = models.FieldCondition(key="title", match=match)
    query_filter = to_filter(req, extra=[condition])

    # 중복/재정렬을 감안해 넉넉히 훑는다 (인덱스 조회라 비용이 작다)
    fetch = min(req.limit * 10, 500) if (req.dedupe or req.rerank) else req.limit

    began = time.perf_counter()
    points, _ = STATE["store"].client.scroll(
        collection_name=C.COLLECTION, scroll_filter=query_filter,
        limit=fetch, with_payload=True, with_vectors=False)

    items = [to_item(p.payload or {}) for p in points]
    found = len(items)
    if req.dedupe:
        items = dd.dedupe(items)
    removed = found - len(items)

    reranked = maybe_rerank(req.query, items, want=req.rerank, top_k=req.limit)
    if not reranked:
        del items[req.limit:]
    took = (time.perf_counter() - began) * 1000
    if C.RELEASE_CACHE:
        gpu.release(STATE["device"])

    return SearchResponse(query=req.query, mode="keyword", count=len(items),
                          took_ms=round(took, 1), candidates=found,
                          reranked=reranked, deduped=removed, hits=to_hits(items))


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
    """모델과 벡터스토어를 올리고 API 키를 확정한다. 키를 반환."""
    key = api_key or C.API_KEY or secrets.token_urlsafe(24)
    STATE["api_key"] = key

    # 질의는 한 번에 하나라 배치를 크게 잡을 이유가 없다
    embeddings, resolved = embedding.build(device, batch_size=8, tf32=tf32,
                                           memory_gib=memory_gib)
    STATE["device"] = resolved
    STATE["store"] = store.build_search_store(
        embeddings, timeout=C.SEARCH_TIMEOUT)
    print(f"[model] embedding {C.MODEL_NAME} on {resolved}")

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
