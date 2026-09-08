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
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

# scripts/ 에서 실행해도 레포 루트의 src 를 찾을 수 있게
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402


def sample_container(name: str) -> str | None:
    """검색이 도는 동안 컨테이너가 CPU 를 쓰는지 본다.

    CPU 가 붙어 있으면 계산 중(인덱스를 못 쓰고 전수 스캔 등),
    0 에 가까우면 I/O 나 락을 기다리는 중이다.
    """
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
             "{{.CPUPerc}} {{.MemUsage}}", name],
            capture_output=True, text=True, timeout=60, shell=True)
    except Exception:
        return None
    return out.stdout.strip() or None


def watch_while(fn, container: str, interval: float = 10.0):
    """fn 을 돌리면서 컨테이너 상태를 주기적으로 찍는다."""
    result: dict = {}

    def run():
        try:
            result["value"] = fn()
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    began = time.perf_counter()
    worker.start()
    while worker.is_alive():
        worker.join(timeout=interval)
        if not worker.is_alive():
            break
        stats = sample_container(container)
        if stats:
            print(f"      +{time.perf_counter() - began:5.0f}s  {stats}")
    if "error" in result:
        raise result["error"]
    return result.get("value")


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
    p.add_argument("--container", default="kisti_openalex_vdb",
                   help="검색 중 CPU 를 관찰할 컨테이너 이름")
    p.add_argument("--watch-only", action="store_true",
                   help="가장 가벼운 검색 하나만, 컨테이너를 관찰하며 실행")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = QdrantClient(url=args.qdrant_url, api_key=args.api_key,
                          timeout=args.timeout)
    rng = np.random.default_rng(0)
    name = args.collection

    print(f"[probe] {args.qdrant_url} / {name}  (timeout {args.timeout}s)\n")

    if args.watch_only:
        print("  가장 가벼운 검색(limit 1 / ef 16)을 돌리며 컨테이너를 관찰합니다.")
        print(f"  기준선: {sample_container(args.container)}\n")
        timed("검색 limit 1 / ef 16 (감시)", lambda: watch_while(
            lambda: client.query_points(
                collection_name=name, query=random_vector(rng), limit=1,
                search_params=models.SearchParams(
                    hnsw_ef=16,
                    quantization=models.QuantizationSearchParams(rescore=False)),
                with_payload=False, timeout=args.timeout),
            args.container))
        print("\n[해석]")
        print("  CPU 가 100% 근처   -> 인덱스를 못 쓰고 전수 스캔 중일 가능성")
        print("  CPU 가 0 근처      -> I/O 대기 또는 락. 계산은 안 하고 있다")
        print("  MEM 이 오르내림    -> 캐시가 밀려나며 다시 읽는 중 (스래싱)")
        return 0

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
