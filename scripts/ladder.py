#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""변수를 하나씩만 바꿔가며 검색 시간을 재서 무엇이 느린지 가른다.

지금까지 관찰:
    난수 벡터 / limit 1 / ef 16 / payload 안 받음   -> 0.05 ~ 0.8초
    진짜 벡터 / limit 50 / ef 64 / payload 받음     -> 300초 타임아웃

이 사이에 변수가 네 개(벡터 종류, limit, ef, payload) 걸쳐 있어서 그대로는
무엇 때문인지 알 수 없다. 하나씩만 움직이며 재면 범인이 드러난다.

각 항목은 --timeout 초에서 끊으므로 최악의 경우에도 (항목 수 x timeout) 이다.

    python scripts/ladder.py
    python scripts/ladder.py --timeout 60
    python scripts/ladder.py --no-model      # 모델 없이 난수만 (30초 절약)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402

QUERIES = [
    "graph neural networks for drug discovery",
    "self-supervised learning for medical imaging",
    "quantum error correction surface codes",
]

ARTICLE = models.Filter(must=[models.FieldCondition(
    key="type", match=models.MatchValue(value="article"))])


def random_vector(rng) -> list[float]:
    vec = rng.standard_normal(C.DIM, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def run(client, vectors, *, limit, ef, payload, filt=None, timeout):
    """벡터가 1개면 단일 질의, 여러 개면 배치로 던진다."""
    params = models.SearchParams(
        hnsw_ef=max(ef, limit),
        quantization=models.QuantizationSearchParams(rescore=False))
    if len(vectors) == 1:
        return client.query_points(
            collection_name=C.COLLECTION, query=vectors[0], limit=limit,
            query_filter=filt, search_params=params,
            with_payload=payload, with_vector=False, timeout=timeout)
    return client.query_batch_points(
        collection_name=C.COLLECTION,
        requests=[models.QueryRequest(
            query=vector, limit=limit, filter=filt, params=params,
            with_payload=payload, with_vector=False) for vector in vectors],
        timeout=timeout)


def measure(client, label: str, vectors, *, timeout, **kwargs) -> float | None:
    print(f"  {label:<46}", end="", flush=True)
    began = time.perf_counter()
    try:
        run(client, vectors, timeout=timeout, **kwargs)
        elapsed = time.perf_counter() - began
        print(f"{elapsed:9.2f}s")
        return elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - began
        name = type(exc).__name__
        print(f"{elapsed:9.2f}s   <- {name}")
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="검색 변수를 하나씩 바꿔가며 측정",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--timeout", type=int, default=45,
                   help="항목별 제한시간. 짧게 잡아야 전체가 빨리 끝난다")
    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    p.add_argument("--no-model", action="store_true",
                   help="임베딩 모델을 올리지 않고 난수 벡터만 쓴다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(0)
    client = QdrantClient(url=args.qdrant_url, api_key=C.QDRANT_API_KEY,
                          timeout=args.timeout)

    print(f"[ladder] {args.qdrant_url} / {C.COLLECTION}  "
          f"(항목별 {args.timeout}초에서 끊음)")

    noise = [random_vector(rng)]
    real = noise
    if not args.no_model:
        from src import embedding
        print("\n  임베딩 모델 적재 중...", end="", flush=True)
        began = time.perf_counter()
        model, device = embedding.build(
            device="auto", memory_gib=C.GPU_MEMORY_GIB or None)
        real = model.embed_documents(QUERIES)
        print(f" {time.perf_counter() - began:.1f}s ({device})")

    one = [real[0]]
    kind = "난수" if args.no_model else "진짜"

    print("\n[A] 벡터 종류만 다르게  (limit 1 / ef 16 / payload 안 받음)")
    measure(client, "난수 벡터", noise,
            limit=1, ef=16, payload=False, timeout=args.timeout)
    measure(client, f"{kind} 질의 벡터", one,
            limit=1, ef=16, payload=False, timeout=args.timeout)

    print("\n[B] limit 과 ef 만 올림  (질의 벡터 / payload 안 받음)")
    for limit, ef in ((1, 16), (10, 64), (50, 64), (50, 128)):
        measure(client, f"limit {limit} / ef {ef}", one,
                limit=limit, ef=ef, payload=False, timeout=args.timeout)

    print("\n[C] payload 만 켬  (질의 벡터 / limit 50 / ef 64)")
    measure(client, "payload 안 받음", one,
            limit=50, ef=64, payload=False, timeout=args.timeout)
    measure(client, "payload 받음  <- 여기서 터지면 payload 가 원인", one,
            limit=50, ef=64, payload=True, timeout=args.timeout)

    print("\n[D] 필터만 켬  (질의 벡터 / limit 50 / ef 64 / payload 받음)")
    measure(client, "필터 없음", one,
            limit=50, ef=64, payload=True, timeout=args.timeout)
    measure(client, "type=article", one,
            limit=50, ef=64, payload=True, filt=ARTICLE, timeout=args.timeout)

    print("\n[E] 질의 개수만 늘림  (limit 50 / ef 64 / payload 받음)")
    measure(client, "질의 1개", one,
            limit=50, ef=64, payload=True, timeout=args.timeout)
    measure(client, "질의 3개 배치", real,
            limit=50, ef=64, payload=True, timeout=args.timeout)

    print("\n[해석]")
    print("  [A] 에서 갈리면      -> 질의가 데이터 밀집 영역을 향할 때만 느린 것")
    print("  [B] 에서 갈리면      -> 그래프 탐색 폭(ef/limit)에 비례해 느려지는 것")
    print("  [C] 에서 갈리면      -> payload 를 디스크에서 읽는 것이 원인")
    print("  [D] 에서 갈리면      -> 필터 경로가 원인")
    print("  [E] 에서만 갈리면    -> 동시 처리(배치)가 원인")
    print("  전부 비슷하면        -> 특정 조건이 아니라 간헐적으로 멈추는 것")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
