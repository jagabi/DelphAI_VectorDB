#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qdrant 자체를 직접 재본다. API 도 모델도 거치지 않는다.

느린 게 Qdrant 인지 그 위 계층인지 가리기 위한 것이다.
임베딩 없이 난수 벡터를 쓰므로 결과의 의미는 없지만, 걸리는 시간은 실제와 같다.

    python scripts/probe.py
    python scripts/probe.py --timeout 600      # 더 오래 기다려보기
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

# scripts/ 에서 실행해도 레포 루트의 src 를 찾을 수 있게
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402


def timed(label: str, fn):
    began = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - began
        print(f"  {label:44s} {elapsed:8.2f}s")
        return result, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - began
        print(f"  {label:44s} {elapsed:8.2f}s  <- {type(exc).__name__}")
        return None, elapsed


def random_vector(rng) -> list[float]:
    vec = rng.standard_normal(C.DIM, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Qdrant 직접 측정")
    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    p.add_argument("--api-key", default=C.QDRANT_API_KEY)
    p.add_argument("--collection", default=C.COLLECTION)
    p.add_argument("--timeout", type=int, default=300)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = QdrantClient(url=args.qdrant_url, api_key=args.api_key,
                          timeout=args.timeout)
    rng = np.random.default_rng(0)
    name = args.collection

    print(f"[probe] {args.qdrant_url} / {name}  (timeout {args.timeout}s)\n")

    info, _ = timed("get_collection (메타데이터만)",
                    lambda: client.get_collection(name))
    if info is not None:
        print(f"       status={info.status} points={info.points_count:,} "
              f"segments={info.segments_count} "
              f"indexed={info.indexed_vectors_count:,}")
    print()

    # payload 조회만. 벡터 연산 없음.
    timed("scroll 10건 (payload 읽기만)",
          lambda: client.scroll(collection_name=name, limit=10,
                                with_payload=True, with_vectors=False))

    # 가장 가벼운 벡터 검색부터 단계적으로 올린다.
    for label, limit, ef in (("limit 1  / ef 16", 1, 16),
                             ("limit 10 / ef 64", 10, 64),
                             ("limit 50 / ef 128", 50, 128)):
        timed(f"검색 {label}", lambda l=limit, e=ef: client.query_points(
            collection_name=name, query=random_vector(rng), limit=l,
            search_params=models.SearchParams(
                hnsw_ef=e,
                quantization=models.QuantizationSearchParams(rescore=False)),
            with_payload=False, timeout=args.timeout))

    # payload 까지 받아오면 얼마나 더 드는지
    timed("검색 limit 50 + payload", lambda: client.query_points(
        collection_name=name, query=random_vector(rng), limit=50,
        search_params=models.SearchParams(
            hnsw_ef=128,
            quantization=models.QuantizationSearchParams(rescore=False)),
        with_payload=True, timeout=args.timeout))

    # 필터를 걸면 (payload 인덱스 경로)
    timed("검색 limit 50 + type=article 필터", lambda: client.query_points(
        collection_name=name, query=random_vector(rng), limit=50,
        query_filter=models.Filter(must=[models.FieldCondition(
            key="type", match=models.MatchValue(value="article"))]),
        search_params=models.SearchParams(
            hnsw_ef=128,
            quantization=models.QuantizationSearchParams(rescore=False)),
        with_payload=True, timeout=args.timeout))

    # 같은 검색을 한 번 더 (캐시가 데워졌는지)
    timed("검색 limit 10 재실행 (캐시 확인)", lambda: client.query_points(
        collection_name=name, query=random_vector(rng), limit=10,
        search_params=models.SearchParams(
            hnsw_ef=64,
            quantization=models.QuantizationSearchParams(rescore=False)),
        with_payload=False, timeout=args.timeout))

    print("\n[해석]")
    print("  get_collection 은 빠른데 검색만 느리다  -> Qdrant 의 검색 경로 문제")
    print("  limit 1 / ef 16 조차 수십 초           -> 스토리지 I/O 가 병목")
    print("  필터 있을 때만 느리다                  -> payload 인덱스 문제")
    print("  전부 빠르다                            -> Qdrant 는 정상, 상위 계층 문제")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
