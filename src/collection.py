#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""컬렉션 생성 / payload 인덱스 관리 / 대량 적재 모드 토글."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from . import config as C


def client(url: str | None = None, api_key: str | None = None, timeout: int = 300):
    return QdrantClient(url=url or C.QDRANT_URL,
                        api_key=api_key or C.QDRANT_API_KEY, timeout=timeout)


# ---------------------------------------------------------------------------
# payload 인덱스
# ---------------------------------------------------------------------------

def _has_field(model_cls, name: str) -> bool:
    fields = getattr(model_cls, "model_fields", None) or getattr(model_cls, "__fields__", {})
    return name in fields


def _text_params() -> models.TextIndexParams:
    kwargs = {
        "type": models.TextIndexType.TEXT,
        "tokenizer": models.TokenizerType.WORD,
        "lowercase": True,
    }
    # phrase_matching 은 비교적 최신 qdrant-client/서버에서만 지원
    if _has_field(models.TextIndexParams, "phrase_matching"):
        kwargs["phrase_matching"] = True
    return models.TextIndexParams(**kwargs)


def payload_indexes() -> list[tuple[str, object]]:
    """(필드명, 스키마) 목록. config.py 의 목록에서 만든다."""
    out: list[tuple[str, object]] = []
    for field in C.KEYWORD_INDEXES:
        out.append((field, models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)))
    for field in C.TEXT_INDEXES:
        out.append((field, _text_params()))
    for field in C.INTEGER_INDEXES:
        out.append((field, models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER, lookup=True, range=True)))
    for field in C.DATETIME_INDEXES:
        out.append((field, models.DatetimeIndexParams(type=models.DatetimeIndexType.DATETIME)))
    return out


def create_indexes(qdrant, keep: set[str] | None = None, wait: bool = False) -> int:
    fields = [(f, s) for f, s in payload_indexes() if not keep or f in keep]
    for field, schema in fields:
        qdrant.create_payload_index(collection_name=C.COLLECTION, field_name=field,
                                    field_schema=schema, wait=wait)
        print(f"  [add] {field}")
    return len(fields)


def drop_indexes(qdrant, keep: set[str] | None = None) -> int:
    dropped = 0
    for field, _ in payload_indexes():
        if keep and field in keep:
            print(f"  [keep] {field}")
            continue
        try:
            qdrant.delete_payload_index(collection_name=C.COLLECTION,
                                        field_name=field, wait=True)
            print(f"  [del]  {field}")
            dropped += 1
        except Exception as exc:
            print(f"  [skip] {field}: {type(exc).__name__}")
    return dropped


# ---------------------------------------------------------------------------
# 컬렉션
# ---------------------------------------------------------------------------

def vectors_config(vectors_in_ram: bool = False) -> dict[str, models.VectorParams]:
    return {
        C.VECTOR_NAME: models.VectorParams(
            size=C.DIM,
            distance=models.Distance.COSINE,
            datatype=models.Datatype.FLOAT32,
            on_disk=not vectors_in_ram,     # 원본은 디스크 mmap, 검색은 양자화본으로
            hnsw_config=models.HnswConfigDiff(
                m=C.HNSW_M,
                ef_construct=C.HNSW_EF_CONSTRUCT,
                payload_m=C.HNSW_PAYLOAD_M,
                on_disk=False,              # 그래프는 램 (검색 지연에 직결)
            ),
        )
    }


def quantization_config(enabled: bool = True):
    if not enabled:
        return None
    # int8 은 원본 대비 1/4. always_ram 이면 검색 경로가 전부 램에서 끝난다.
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True)
    )


def create(qdrant, *, segments: int | None = None, quantize: bool = True,
           vectors_in_ram: bool = False, indexing_threshold: int | None = None) -> None:
    qdrant.create_collection(
        collection_name=C.COLLECTION,
        vectors_config=vectors_config(vectors_in_ram),
        quantization_config=quantization_config(quantize),
        on_disk_payload=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=(C.INDEXING_THRESHOLD if indexing_threshold is None
                                else indexing_threshold),
            # 세그먼트마다 id tracker / HNSW 그래프 / payload 인덱스를 따로 들고 있어서
            # 개수가 곧 램 오버헤드의 곱셈 계수가 된다. 자동(0)으로 두면 쓰기가 몰릴 때
            # 100개 넘게 쌓이므로 명시적으로 적게 잡는다.
            default_segment_number=C.SEGMENT_NUMBER if segments is None else segments,
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),   # 컬렉션 기본값
    )
    print(f"[ok] 컬렉션 생성: {C.COLLECTION}")
    count = create_indexes(qdrant, wait=True)
    print(f"[ok] payload 인덱스 {count}개 생성 완료")


def set_bulk_mode(qdrant, enabled: bool) -> None:
    """적재 중 HNSW 빌드를 멈추거나 재개한다. payload 인덱스는 영향받지 않는다."""
    threshold = 0 if enabled else C.INDEXING_THRESHOLD
    qdrant.update_collection(
        collection_name=C.COLLECTION,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
    )


def describe(qdrant) -> None:
    info = qdrant.get_collection(C.COLLECTION)
    params = info.config.params
    vec = params.vectors
    vec_params = next(iter(vec.values())) if isinstance(vec, dict) else vec

    print(f"\n[collection] {C.COLLECTION}")
    print(f"  status          : {info.status}")
    print(f"  optimizer       : {info.optimizer_status}")
    print(f"  points          : {info.points_count:,}")
    print(f"  indexed vectors : {info.indexed_vectors_count:,}")
    print(f"  segments        : {info.segments_count}")
    print(f"  vector          : {vec_params.size} / {vec_params.distance}")
    print(f"  vectors.on_disk : {vec_params.on_disk}")
    print(f"  quantization    : {info.config.quantization_config}")
    print(f"  on_disk_payload : {params.on_disk_payload}")
    print(f"  payload indexes : {len(info.payload_schema)}개")
    # 필터를 거는 필드에 인덱스가 없으면 Qdrant 는 후보마다 payload 본문을 읽어야
    # 한다. on_disk_payload=True 면 그게 전부 디스크 접근이라 검색이 멈춘 것처럼
    # 느려진다. 그래서 이름과 타입, 색인된 점 수까지 확인할 수 있게 찍는다.
    for name, schema in sorted(info.payload_schema.items()):
        kind = getattr(schema, "data_type", schema)
        points = getattr(schema, "points", None)
        counted = f"{points:,}" if isinstance(points, int) else "-"
        print(f"      {name:<50} {str(kind):<12} points={counted}")
