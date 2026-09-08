#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""세그먼트별 인덱스 상태와 검색 경로 카운터를 뽑는다.

컬렉션 통계(`indexed_vectors_count`)는 "인덱싱이 끝났다"고 말해도, 실제 검색이
그 인덱스를 타는지는 별개다. Qdrant 는 검색을 처리한 경로를 따로 세는데,

    unfiltered_hnsw   HNSW 그래프를 걸어서 처리 (정상, 빠름)
    unfiltered_plain  인덱스 없이 전수 스캔 (느림)

`plain` 쪽에 숫자가 쌓여 있으면 인덱스가 있어도 안 쓰이고 있다는 뜻이다.
세그먼트 타입도 같이 본다. `plain` 세그먼트는 HNSW 자체가 없다.

    python scripts/segments.py
    python scripts/segments.py --raw telemetry.json   # 원본도 저장
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

# 검색 경로 카운터 이름. 앞의 둘이 핵심이다.
SEARCH_PATHS = ("unfiltered_hnsw", "unfiltered_plain", "unfiltered_exact",
                "filtered_plain", "filtered_small_cardinality",
                "filtered_large_cardinality", "filtered_exact",
                "unfiltered_sparse", "filtered_sparse")


def fetch(url: str, api_key: str, timeout: int) -> dict:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("api-key", api_key)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def walk(node, want: str):
    """중첩 구조 어디에 있든 `want` 키를 가진 dict 를 전부 찾는다.

    telemetry 의 모양은 버전마다 조금씩 달라서 경로를 고정하지 않는다.
    """
    if isinstance(node, dict):
        if want in node:
            yield node
        for value in node.values():
            yield from walk(value, want)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item, want)


def count_of(value) -> int:
    """카운터가 숫자로도, 통계 dict 로도 온다. 호출 횟수만 꺼낸다."""
    if isinstance(value, dict):
        return int(value.get("count") or 0)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def duration_of(value) -> float | None:
    """평균 소요 시간(ms). 통계 dict 일 때만 있다."""
    if not isinstance(value, dict):
        return None
    stats = value.get("total_duration_micros") or value
    avg = stats.get("avg_duration_micros") if isinstance(stats, dict) else None
    if avg is None:
        avg = value.get("avg_duration_micros")
    return float(avg) / 1000 if avg else None


def show_segments(telemetry: dict) -> None:
    segments = list(walk(telemetry, "segment_type"))
    if not segments:
        print("  [warn] 세그먼트 정보를 찾지 못했습니다 (telemetry 모양이 다름)")
        return

    kinds = Counter()
    points = Counter()
    indexed = Counter()
    for info in segments:
        kind = str(info.get("segment_type"))
        kinds[kind] += 1
        points[kind] += int(info.get("num_points") or 0)
        indexed[kind] += int(info.get("num_indexed_vectors") or 0)

    print(f"  세그먼트 {len(segments)}개\n")
    print(f"    {'타입':<12}{'개수':>8}{'points':>16}{'indexed':>16}")
    for kind, number in kinds.most_common():
        print(f"    {kind:<12}{number:>8}{points[kind]:>16,}{indexed[kind]:>16,}")

    plain = points.get("plain", 0)
    total = sum(points.values())
    if plain and total:
        print(f"\n  [!] plain 세그먼트에 {plain:,}건 ({plain / total:.1%}). "
              "여기는 HNSW 가 없어 전수 스캔한다.")


def show_search_paths(telemetry: dict) -> None:
    totals = Counter()
    durations: dict[str, list[float]] = {}
    found = False

    for node in walk(telemetry, "unfiltered_hnsw"):
        found = True
        for name in SEARCH_PATHS:
            if name not in node:
                continue
            totals[name] += count_of(node[name])
            ms = duration_of(node[name])
            if ms:
                durations.setdefault(name, []).append(ms)

    if not found:
        print("  [warn] 검색 경로 카운터를 찾지 못했습니다.")
        print("         (한 번도 검색하지 않았거나 telemetry 모양이 다름)")
        return

    if not sum(totals.values()):
        print("  아직 집계된 검색이 없습니다. 검색을 한 번 돌린 뒤 다시 보세요:")
        print("    python scripts/probe.py --timeout 1800")
        return

    print(f"    {'경로':<28}{'횟수':>10}{'평균':>12}")
    for name in SEARCH_PATHS:
        if not totals[name]:
            continue
        pool = durations.get(name)
        avg = f"{sum(pool) / len(pool):,.0f} ms" if pool else "-"
        print(f"    {name:<28}{totals[name]:>10,}{avg:>12}")

    hnsw = totals["unfiltered_hnsw"] + totals["filtered_large_cardinality"]
    scan = (totals["unfiltered_plain"] + totals["unfiltered_exact"]
            + totals["filtered_plain"] + totals["filtered_exact"])
    print()
    if scan and scan >= hnsw:
        print(f"  [!] 전수 스캔 {scan:,}회 vs HNSW {hnsw:,}회.")
        print("      인덱스를 타지 않고 있다. 이것이 느린 원인이다.")
    elif hnsw:
        print(f"  HNSW {hnsw:,}회 / 전수 스캔 {scan:,}회. 인덱스는 쓰이고 있다.")
        print("      -> 느린 원인은 인덱스가 아니라 그 아래(스토리지/세그먼트 수)다.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="세그먼트 인덱스 상태 점검")
    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    p.add_argument("--api-key", default=C.QDRANT_API_KEY)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--raw", help="telemetry 원본 JSON 을 저장할 경로")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    url = f"{args.qdrant_url.rstrip('/')}/telemetry?details_level=3"
    print(f"[telemetry] {url}\n")

    try:
        payload = fetch(url, args.api_key, args.timeout)
    except urllib.error.URLError as exc:
        print(f"[error] 가져오지 못했습니다: {exc}")
        return 2

    telemetry = payload.get("result", payload)

    if args.raw:
        Path(args.raw).write_text(json.dumps(telemetry, indent=2),
                                  encoding="utf-8")
        print(f"[raw] {args.raw} 에 저장했습니다\n")

    print("[세그먼트]")
    show_segments(telemetry)
    print("\n[검색 경로]")
    show_search_paths(telemetry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
