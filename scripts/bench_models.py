#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""임베딩 모델과 리랭커만 따로 재본다. Qdrant 에는 연결하지 않는다.

검색이 느릴 때 원인이 모델 쪽인지 가리기 위한 것이다. API 서버가 하는 것과
같은 방식으로(같은 device, 같은 배치, 같은 메모리 상한) 올린 뒤 시간을 잰다.

    python scripts/bench_models.py
    python scripts/bench_models.py --queries 3 --docs 50
    python scripts/bench_models.py --skip-reranker
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src import config as C  # noqa: E402
from src import embedding, gpu  # noqa: E402

QUERIES = [
    "transformer architectures for long document retrieval",
    "graph neural networks applied to molecular property prediction",
    "self-supervised pretraining for medical image segmentation",
    "quantum error correction with surface codes",
    "large language model alignment from human feedback",
]

# 실제 초록 정도의 길이. 토크나이저가 자르는 지점이 비슷해야 시간이 의미가 있다.
DOC = (
    "We present a method for learning representations that transfer across "
    "domains without labelled supervision. Our approach combines contrastive "
    "objectives with a masked reconstruction term, and we show that the two "
    "are complementary. Experiments on eleven benchmarks demonstrate "
    "consistent improvements over strong baselines, with the largest gains in "
    "the low-resource regime. We further analyse which components matter "
    "through a series of ablations, and release code and pretrained weights. "
) * 3


def report_torch() -> str:
    print("[torch]")
    print(f"  version        : {torch.__version__}")
    print(f"  cuda build     : {torch.version.cuda}")
    print(f"  cuda available : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n  [!] GPU 를 못 씁니다. 아래를 확인하세요.")
        gpu.diagnose_cpu_fallback()
        return "cpu"
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    print(f"  device         : {props.name}")
    print(f"  총 VRAM        : {props.total_memory / (1 << 30):.1f} GiB")
    if C.GPU_MEMORY_GIB:
        print(f"  상한(config)   : {C.GPU_MEMORY_GIB:.1f} GiB")
    return "cuda"


def timed(label: str, fn):
    began = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - began
    print(f"  {label:38s} {elapsed * 1000:9.1f} ms")
    return result, elapsed


def bench_embedding(args, device: str):
    print("\n[임베딩] bge-m3 적재")
    cap = C.GPU_MEMORY_GIB or None
    (model, resolved), load = timed("모델 로드", lambda: embedding.build(
        device=device, memory_gib=cap))
    print(f"  device={resolved}")

    queries = (QUERIES * 4)[:args.queries]
    print(f"\n[임베딩] 질의 {len(queries)}개")
    timed("1회차 (커널 컴파일 포함)",
          lambda: embedding.encode(model, queries))
    _, warm = timed("2회차", lambda: embedding.encode(model, queries))
    timed("3회차", lambda: embedding.encode(model, queries))

    per = warm / max(1, len(queries)) * 1000
    print(f"\n  질의 1개당 {per:.1f} ms  (로드 {load:.1f}s)")
    return warm


def bench_reranker(args, device: str):
    from src.reranker import Reranker

    print("\n[리랭커] bge-reranker-v2-m3 적재")
    cap = C.GPU_MEMORY_GIB or None
    model, load = timed("모델 로드",
                        lambda: Reranker(device=device, memory_gib=cap))
    print(f"  device={model.device} batch={model.batch_size} "
          f"max_length={model.max_length}")

    docs = [DOC] * args.docs
    queries = (QUERIES * 4)[:args.queries]
    lists = [[{"text": d} for d in docs] for _ in queries]

    print(f"\n[리랭커] 질의 {len(queries)}개 x 문서 {args.docs}개 "
          f"= 쌍 {len(queries) * args.docs}개")
    timed("1회차 (커널 컴파일 포함)", lambda: model.rerank_many(
        queries, lists, text_of=lambda i: i["text"], top_k=10))
    _, warm = timed("2회차", lambda: model.rerank_many(
        queries, lists, text_of=lambda i: i["text"], top_k=10))
    timed("3회차", lambda: model.rerank_many(
        queries, lists, text_of=lambda i: i["text"], top_k=10))
    print(f"\n  (로드 {load:.1f}s)")
    return warm


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="임베딩/리랭커만 측정 (Qdrant 미사용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--queries", type=int, default=3)
    p.add_argument("--docs", type=int, default=C.SEARCH_CANDIDATES)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-reranker", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = report_torch() if args.device == "auto" else args.device

    embed_ms = bench_embedding(args, device)
    gpu.report_memory(device, "임베딩 후")

    rerank_ms = 0.0
    if not args.skip_reranker:
        rerank_ms = bench_reranker(args, device)
        gpu.report_memory(device, "리랭커 후")

    total = embed_ms + rerank_ms
    print("\n[정리]")
    print(f"  모델이 쓰는 시간 합계          {total * 1000:9.1f} ms")
    print("\n  API 한 번의 총 시간에서 이만큼을 빼면 나머지가 Qdrant 몫이다.")
    if total < 2.0:
        print("  -> 모델은 정상. 느리다면 원인은 Qdrant 쪽이다.")
    else:
        print("  -> 모델이 느리다. GPU 를 못 쓰고 있는지 위 [torch] 를 보라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
