#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangChain QdrantVectorStore 연결.

QdrantVectorStore 기본 _build_payloads 는 payload 를
    {"page_content": <text>, "metadata": {...}}
로 중첩 저장한다. 그런데 이 컬렉션은 title / publication_year / openalex_id 처럼
최상위(flat) 필드에 인덱스가 걸려 있어서 중첩하면 인덱스가 전혀 안 먹는다.
그래서 쓰기(_build_payloads) / 읽기(_document_from_point) 두 메서드만 flat 으로
바꾸고 나머지(배치 생성, upsert, 검색)는 기본 구현을 그대로 쓴다.
"""

from __future__ import annotations

import threading

from . import config as C

_FLAT_CLS = None


def flat_store_cls():
    """langchain_qdrant 를 지연 임포트해서 flat-payload 서브클래스를 만든다."""
    global _FLAT_CLS
    if _FLAT_CLS is not None:
        return _FLAT_CLS

    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore

    class FlatQdrantVectorStore(QdrantVectorStore):
        """metadata 를 payload 최상위에 펼쳐서 저장/조회한다."""

        @staticmethod
        def _build_payloads(texts, metadatas, content_key, metadata_key):  # type: ignore[override]
            payloads = []
            for i, text in enumerate(texts):
                if text is None:
                    msg = "At least one of the texts is None."
                    raise ValueError(msg)
                payload = dict(metadatas[i]) if metadatas is not None else {}
                payload[content_key] = text
                payloads.append(payload)
            return payloads

        @classmethod
        def _document_from_point(cls, point, collection, content_key, metadata_key):  # type: ignore[override]
            payload = dict(point.payload or {})
            content = payload.pop(content_key, "")
            payload["_id"] = point.id
            payload["_collection_name"] = collection
            return Document(page_content=content, metadata=payload)

    _FLAT_CLS = FlatQdrantVectorStore
    return _FLAT_CLS


class PrecomputedEmbeddings:
    """.f16 에서 읽은 벡터를 그대로 돌려주는 Embeddings 구현.

    적재할 때는 모델을 띄울 필요가 없다. 여러 스레드가 동시에 add_texts 를
    호출할 수 있으므로 슬롯은 스레드 로컬로 둔다.

    LangChain 의 Embeddings 를 상속하려면 langchain_core 임포트가 필요한데,
    적재 경로에서만 쓰이므로 덕 타이핑으로 충분하다. (아래 build_upload_store
    에서 validate_embeddings=False 로 검증을 끄기 때문에 isinstance 검사도 없다.)
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def feed(self, vectors: list[list[float]]) -> None:
        self._local.pending = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = getattr(self._local, "pending", None)
        self._local.pending = None
        if vectors is None or len(vectors) != len(texts):
            got = 0 if vectors is None else len(vectors)
            msg = f"배치 불일치: texts={len(texts)}, vectors={got}"
            raise RuntimeError(msg)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        msg = "적재 전용 Embeddings 입니다. 검색에는 쓸 수 없습니다."
        raise NotImplementedError(msg)


def _connect(embedding, *, url, api_key, timeout, prefer_grpc, validate):
    return flat_store_cls().from_existing_collection(
        collection_name=C.COLLECTION,
        embedding=embedding,
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=prefer_grpc,
        vector_name=C.VECTOR_NAME,         # 이 컬렉션은 "" (langchain 기본값)
        content_payload_key=C.TEXT_FIELD,  # page_content <-> abstract
        validate_embeddings=validate,
        validate_collection_config=validate,
    )


def check_collection(client, collection: str | None = None) -> None:
    """벡터 이름 / 차원이 .f16 과 맞는지 확인한다."""
    name = collection or C.COLLECTION
    info = client.get_collection(name)
    vectors = info.config.params.vectors

    if isinstance(vectors, dict):
        if C.VECTOR_NAME not in vectors:
            msg = (f"컬렉션에 {C.VECTOR_NAME!r} 벡터가 없습니다. "
                   f"있는 것: {list(vectors)}")
            raise SystemExit(msg)
        params = vectors[C.VECTOR_NAME]
    else:
        params = vectors   # 이름 없는 기본 벡터

    if params.size != C.DIM:
        msg = f"컬렉션 차원 {params.size} != 임베딩 차원 {C.DIM}"
        raise SystemExit(msg)

    print(f"[qdrant] {name} | dim={params.size} {params.distance} "
          f"| points={info.points_count:,} | status={info.status}")


def build_upload_store(*, url=None, api_key=None, timeout=300, prefer_grpc=False):
    """적재용. 모델을 띄우지 않는다."""
    embeddings = PrecomputedEmbeddings()
    # LangChain 의 차원 검증은 embed_documents(["dummy_text"]) 를 호출하는데
    # 우리 Embeddings 는 미리 먹여둔 벡터만 돌려주므로 걸린다. 직접 검증한다.
    store = _connect(
        embeddings,
        url=url or C.QDRANT_URL, api_key=api_key or C.QDRANT_API_KEY,
        timeout=timeout, prefer_grpc=prefer_grpc, validate=False,
    )
    check_collection(store.client)
    return store, embeddings


def build_search_store(embeddings, *, url=None, api_key=None, timeout=120):
    """검색용. 진짜 임베딩 모델이 필요하다."""
    return _connect(
        embeddings,
        url=url or C.QDRANT_URL, api_key=api_key or C.QDRANT_API_KEY,
        timeout=timeout, prefer_grpc=False, validate=True,
    )
