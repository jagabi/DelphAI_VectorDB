#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""질의 -> 임베딩 -> 벡터 검색 top N -> 중복 제거 -> 리랭킹 -> top K.

API 서버도 터널도 거치지 않고 이 프로세스 안에서 전 과정을 돈다.
단계마다 걸린 시간을 따로 찍으므로 어디가 느린지 바로 보인다.
쓰는 코드는 api/server.py 와 같은 것이라 결과도 같다.

    python scripts/ask.py "graph neural networks for drug discovery"
    python scripts/ask.py "q1" "q2" "q3" --type article
    python scripts/ask.py "..." --candidates 50 --limit 10 --full
    python scripts/ask.py "..." --no-rerank --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models  # noqa: E402

from api.server import rerank_text, to_item  # noqa: E402
from src import config as C  # noqa: E402
from src import dedupe as dd  # noqa: E402
from src import embedding  # noqa: E402

DOTS = 32


def force_utf8() -> None:
    """Windows 콘솔 기본 인코딩(cp949)에서 특수문자가 깨지는 것을 막는다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class Stage:
    """단계별 소요 시간을 재서 그 자리에서 찍는다."""

    def __init__(self, total: int):
        self.total = total
        self.index = 0
        self.spans: dict[str, float] = {}

    def run(self, label: str, fn):
        self.index += 1
        print(f"[{self.index}/{self.total}] {label}".ljust(DOTS, "."),
              end="", flush=True)
        began = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            print(f" {time.perf_counter() - began:8.2f}s"
                  f"   <- {type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - began
        self.spans[label] = elapsed
        print(f" {elapsed:8.2f}s")
        return result

    def note(self, text: str) -> None:
        print(f"{' ' * (DOTS + 10)}   {text}")


def build_filter(args) -> models.Filter | None:
    """인덱스가 걸려 있는 필드만 조건으로 쓴다."""
    must: list = []
    for key, value in (("type", args.type),
                       ("country_code", args.country),
                       ("primary_topic__display_name", args.topic),
                       ("primary_topic__field__display_name", args.field),
                       ("primary_location__source__display_name", args.source)):
        if value:
            must.append(models.FieldCondition(
                key=key, match=models.MatchValue(value=value)))
    if args.year:
        must.append(models.FieldCondition(
            key="publication_year", match=models.MatchValue(value=args.year)))
    elif args.year_from or args.year_to:
        must.append(models.FieldCondition(
            key="publication_year",
            range=models.Range(gte=args.year_from, lte=args.year_to)))
    return models.Filter(must=must) if must else None


def wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def show(queries: list[str], results: list[list[dict]], args) -> None:
    for query, items in zip(queries, results):
        print()
        print("=" * 78)
        print(f"  {query}")
        print("=" * 78)
        if not items:
            print("  (결과 없음)")
            continue
        for rank, item in enumerate(items, start=1):
            score = item.get("rerank_score")
            label = "rerank" if score is not None else "vector"
            if score is None:
                score = item.get("vector_score") or 0.0
            copies = item.get("duplicates", 1)
            merged = f"   (중복 {copies}건 병합)" if copies > 1 else ""
            title = item.get("title") or "(제목 없음)"
            print()
            print(f"{rank:2d}. [{label} {score:+.3f}]  {title}{merged}")

            meta = " · ".join(str(v) for v in (
                item.get("publication_year"), item.get("source"),
                item.get("type"), item.get("country"), item.get("doi"),
            ) if v)
            if meta:
                print(f"    {meta}")
            if item.get("topic"):
                print(f"    분야: {item['topic']}")

            abstract = item.get("abstract") or ""
            if abstract and not args.no_abstract:
                if not args.full and len(abstract) > args.chars:
                    abstract = abstract[:args.chars].rstrip() + " …"
                for line in wrap(abstract, 74):
                    print(f"    {line}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="임베딩 -> 벡터검색 -> 리랭킹 전 과정을 로컬에서 실행",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("queries", nargs="+", help="질의. 여러 개 주면 배치로 처리")

    p.add_argument("--candidates", type=int, default=C.SEARCH_CANDIDATES,
                   help="벡터 검색으로 뽑을 후보 수")
    p.add_argument("--limit", type=int, default=C.SEARCH_LIMIT,
                   help="리랭킹 후 최종 개수")
    p.add_argument("--hnsw-ef", type=int, default=C.SEARCH_HNSW_EF)
    p.add_argument("--rescore", action="store_true",
                   help="양자화 점수를 원본 벡터로 재채점. 정확하지만 느리다")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--timeout", type=int, default=C.SEARCH_TIMEOUT)
    p.add_argument("--retries", type=int, default=0,
                   help="검색이 실패하면 이만큼 다시 던진다")

    p.add_argument("--type", help="예: article, preprint")
    p.add_argument("--country")
    p.add_argument("--topic")
    p.add_argument("--field")
    p.add_argument("--source")
    p.add_argument("--year", type=int)
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)

    p.add_argument("--full", action="store_true", help="초록 전문 출력")
    p.add_argument("--chars", type=int, default=300, help="초록 잘라낼 길이")
    p.add_argument("--no-abstract", action="store_true")
    p.add_argument("--json", help="결과를 이 경로에 JSON 으로 저장")
    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    return p.parse_args(argv)


def main(argv=None) -> int:
    force_utf8()
    args = parse_args(argv)
    queries = args.queries
    steps = 4 if args.no_rerank else 5
    ef = max(args.hnsw_ef, args.candidates)

    print(f"[ask] 질의 {len(queries)}개 / 후보 {args.candidates} -> "
          f"최종 {args.limit} / ef {ef}")
    print(f"      {args.qdrant_url} / {C.COLLECTION}")
    print()

    stage = Stage(steps)
    cap = C.GPU_MEMORY_GIB or None

    model, device = stage.run(
        "임베딩 모델 적재",
        lambda: embedding.build(device="auto", memory_gib=cap))
    stage.note(f"device={device}")

    reranker = None
    if not args.no_rerank:
        from src.reranker import Reranker
        reranker = stage.run(
            "리랭커 적재", lambda: Reranker(device="auto", memory_gib=cap))
        stage.note(f"device={reranker.device} batch={reranker.batch_size}")

    vectors = stage.run("질의 임베딩", lambda: model.embed_documents(queries))

    client = QdrantClient(url=args.qdrant_url, api_key=C.QDRANT_API_KEY,
                          timeout=args.timeout)
    params = models.SearchParams(
        # ef 는 탐색 중 들고 다니는 후보 목록의 크기다. 뽑으려는 개수보다 작으면
        # 애초에 그만큼 채울 수 없으므로 최소한 candidates 만큼은 확보한다.
        hnsw_ef=ef,
        quantization=models.QuantizationSearchParams(rescore=args.rescore))
    query_filter = build_filter(args)

    def fetch():
        attempt = 0
        while True:
            try:
                return client.query_batch_points(
                    collection_name=C.COLLECTION,
                    requests=[models.QueryRequest(
                        query=vector, limit=args.candidates,
                        filter=query_filter, params=params,
                        with_payload=True, with_vector=False)
                        for vector in vectors],
                    timeout=args.timeout)
            except Exception as exc:
                if attempt >= args.retries:
                    raise
                attempt += 1
                print(f"\n      [retry {attempt}/{args.retries}] "
                      f"{type(exc).__name__} -> 다시 던집니다", flush=True)

    responses = stage.run("Qdrant 벡터 검색", fetch)
    lists = [[to_item(point.payload or {}, vector_score=float(point.score))
              for point in response.points] for response in responses]
    stage.note(f"후보 {sum(len(items) for items in lists)}개")

    removed = 0
    if not args.no_dedupe:
        before = sum(len(items) for items in lists)
        # 벡터 점수 순으로 와 있으므로 각 중복 묶음에서 가장 좋은 것이 남는다
        lists = [dd.dedupe(items) for items in lists]
        removed = before - sum(len(items) for items in lists)

    if reranker is None:
        for items in lists:
            del items[args.limit:]
        results = lists
        stage.index += 1
        print(f"[{stage.index}/{steps}] 중복 제거".ljust(DOTS, ".")
              + f" {'':8}")
        stage.note(f"{removed}건 제거 (리랭킹 안 함)")
    else:
        results = stage.run(
            "중복 제거 + 리랭킹",
            lambda: reranker.rerank_many(queries, lists, text_of=rerank_text,
                                         top_k=args.limit))
        stage.note(f"중복 {removed}건 제거")

    total = sum(stage.spans.values())
    load = sum(span for name, span in stage.spans.items() if "적재" in name)
    print()
    print(f"  전체 {total:.2f}s   (모델 적재 {load:.2f}s 빼면 {total - load:.2f}s)")

    show(queries, results, args)

    if args.json:
        payload = [
            {"query": query,
             "hits": [{k: v for k, v in item.items() if k != "_payload"}
                      for item in items]}
            for query, items in zip(queries, results)]
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
