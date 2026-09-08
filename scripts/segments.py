#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""세그먼트별 인덱스 상태와 검색 경로 카운터를 뽑는다.

컬렉션 통계(`indexed_vectors_count`)는 "인덱싱이 끝났다"고 말해도, 실제 검색이
그 인덱스를 타는지는 별개다. Qdrant 는 검색을 처리한 경로를 따로 세는데,
`hnsw` 계열은 그래프를 걸은 것이고 `plain`/`exact` 계열은 전수 스캔이다.
후자에 숫자가 쌓여 있으면 인덱스가 있어도 안 쓰이고 있다는 뜻이다.

telemetry 의 모양은 버전마다 다르다. 찾지 못하면 키 뼈대를 찍어준다.

    python scripts/segments.py
    python scripts/segments.py --tree            # 구조만 보기
    python scripts/segments.py --raw tel.json    # 원본 저장
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402

SEGMENT_HINTS = ("segment_type", "num_points", "num_indexed_vectors")
COUNTER_HINTS = ("hnsw", "plain", "exact", "cardinality", "sparse")


def fetch(url: str, api_key: str, timeout: int) -> dict:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("api-key", api_key)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def find(node, predicate):
    """중첩 구조 어디에 있든 조건에 맞는 dict 를 전부 찾는다."""
    if isinstance(node, dict):
        if predicate(node):
            yield node
        for value in node.values():
            yield from find(value, predicate)
    elif isinstance(node, list):
        for item in node:
            yield from find(item, predicate)


def tree(node, depth: int, limit: int, prefix: str = "") -> None:
    """키 뼈대만 찍는다. 값이 큰 리스트면 길이와 첫 원소만."""
    if depth <= 0:
        return
    if isinstance(node, dict):
        for key, value in list(node.items())[:limit]:
            kind = type(value).__name__
            if isinstance(value, (dict, list)):
                size = len(value)
                print(f"{prefix}{key}  ({kind}, {size})")
                tree(value, depth - 1, limit, prefix + "  ")
            else:
                print(f"{prefix}{key} = {value!r}"[:110])
    elif isinstance(node, list) and node:
        print(f"{prefix}[0]")
        tree(node[0], depth - 1, limit, prefix + "  ")


def count_of(value) -> int:
    if isinstance(value, dict):
        return int(value.get("count") or 0)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def avg_ms(value) -> float | None:
    if not isinstance(value, dict):
        return None
    for key in ("avg_duration_micros", "avg"):
        if isinstance(value.get(key), (int, float)):
            return float(value[key]) / 1000
    inner = value.get("total_duration_micros")
    if isinstance(inner, dict):
        return avg_ms(inner)
    return None


def show_segments(telemetry: dict) -> bool:
    def is_segment(node: dict) -> bool:
        return any(hint in node for hint in SEGMENT_HINTS)

    segments = list(find(telemetry, is_segment))
    if not segments:
        return False

    kinds: Counter = Counter()
    points: Counter = Counter()
    indexed: Counter = Counter()
    for info in segments:
        kind = str(info.get("segment_type", "?"))
        kinds[kind] += 1
        points[kind] += int(info.get("num_points") or 0)
        indexed[kind] += int(info.get("num_indexed_vectors") or 0)

    print(f"  세그먼트 {len(segments)}개")
    print()
    print(f"    {'타입':<14}{'개수':>8}{'points':>16}{'indexed':>16}")
    for kind, number in kinds.most_common():
        print(f"    {kind:<14}{number:>8}{points[kind]:>16,}{indexed[kind]:>16,}")

    total = sum(points.values())
    bare = sum(n for k, n in points.items() if k.lower().startswith("plain"))
    if bare and total:
        print()
        print(f"  [!] plain 세그먼트에 {bare:,}건 ({bare / total:.1%}). "
              "여기는 HNSW 가 없어 전수 스캔한다.")
    return True


def show_search_paths(telemetry: dict) -> bool:
    def is_counter(node: dict) -> bool:
        keys = " ".join(node.keys()).lower()
        return sum(hint in keys for hint in COUNTER_HINTS) >= 2

    totals: Counter = Counter()
    spans: dict[str, list[float]] = {}
    found = False

    for node in find(telemetry, is_counter):
        found = True
        for name, value in node.items():
            if not any(hint in name.lower() for hint in COUNTER_HINTS):
                continue
            totals[name] += count_of(value)
            ms = avg_ms(value)
            if ms:
                spans.setdefault(name, []).append(ms)

    if not found:
        return False

    if not sum(totals.values()):
        print("  집계된 검색이 없습니다. 검색을 한 번 돌린 뒤 다시 보세요:")
        print("    python scripts/probe.py --timeout 1800")
        return True

    print(f"    {'경로':<32}{'횟수':>10}{'평균':>14}")
    for name, number in totals.most_common():
        if not number:
            continue
        pool = spans.get(name)
        shown = f"{sum(pool) / len(pool):,.0f} ms" if pool else "-"
        print(f"    {name:<32}{number:>10,}{shown:>14}")

    graph = sum(n for k, n in totals.items() if "hnsw" in k.lower()
                or "large_cardinality" in k.lower())
    scan = sum(n for k, n in totals.items()
               if "plain" in k.lower() or "exact" in k.lower())
    print()
    if scan and scan >= graph:
        print(f"  [!] 전수 스캔 {scan:,}회 vs HNSW {graph:,}회.")
        print("      인덱스를 타지 않고 있다. 이것이 느린 원인이다.")
    elif graph:
        print(f"  HNSW {graph:,}회 / 전수 스캔 {scan:,}회. 인덱스는 쓰이고 있다.")
        print("      -> 원인은 인덱스가 아니라 그 아래(세그먼트 수/스토리지)다.")
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="세그먼트 인덱스 상태 점검")
    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    p.add_argument("--api-key", default=C.QDRANT_API_KEY)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--raw", help="telemetry 원본 JSON 을 저장할 경로")
    p.add_argument("--tree", action="store_true", help="키 뼈대만 출력")
    p.add_argument("--depth", type=int, default=7)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    url = f"{args.qdrant_url.rstrip('/')}/telemetry?details_level=3"
    print(f"[telemetry] {url}")
    print()

    try:
        payload = fetch(url, args.api_key, args.timeout)
    except urllib.error.URLError as exc:
        print(f"[error] 가져오지 못했습니다: {exc}")
        return 2

    telemetry = payload.get("result", payload)

    if args.raw:
        Path(args.raw).write_text(json.dumps(telemetry, indent=2),
                                  encoding="utf-8")
        print(f"[raw] {args.raw} 에 저장했습니다")
        print()

    if args.tree:
        print("[구조]")
        tree(telemetry, args.depth, 40)
        return 0

    print("[세그먼트]")
    ok_segments = show_segments(telemetry)
    if not ok_segments:
        print("  찾지 못했습니다.")
    print()
    print("[검색 경로]")
    ok_paths = show_search_paths(telemetry)
    if not ok_paths:
        print("  찾지 못했습니다.")

    if not (ok_segments and ok_paths):
        print()
        print("[구조] telemetry 모양이 예상과 달라 키 뼈대를 찍습니다.")
        tree(telemetry, 6, 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
