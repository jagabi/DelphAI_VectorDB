#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""검색이 도는 동안 Qdrant 와 VM 의 상태를 주기적으로 찍는다.

probe.py 를 다른 창에서 돌려놓고 이걸 같이 띄운다. 느린 검색이 진행되는 중에
무엇이 변하는지 보기 위한 것이다.

    python scripts/vitals.py
    python scripts/vitals.py --interval 10 --count 60

읽는 값:
    opt     옵티마이저가 처리한 점 수. 늘고 있으면 인덱스를 다시 만드는 중이다.
    queue   쓰기 대기열. 0 이 아니면 뭔가 쓰고 있다.
    cpu     이 컬렉션이 쓴 누적 CPU. 구간별 증가량으로 코어 몇 개분인지 나온다.
    free    VM 의 남은 물리 메모리. 0 으로 떨어지면 메모리를 밀어내는 중이다.
    cache   파일 캐시. 줄어들면 밀려나고 있다는 뜻.
    mapped  mmap 으로 올라온 파일 페이지. HNSW 링크가 여기 산다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402

GIB = 1 << 20   # /proc/meminfo 는 kB 단위라 이걸로 나누면 GiB


def meminfo(container: str) -> dict[str, int]:
    """컨테이너 안에서 /proc/meminfo 를 읽는다. VM 전체의 값이 나온다."""
    try:
        out = subprocess.run(["docker", "exec", container, "cat", "/proc/meminfo"],
                             capture_output=True, text=True, timeout=30, shell=True)
    except Exception:
        return {}
    values: dict[str, int] = {}
    for line in out.stdout.splitlines():
        name, _, rest = line.partition(":")
        digits = rest.strip().split(" ")[0]
        if digits.isdigit():
            values[name] = int(digits)
    return values


def thread_states(container: str) -> dict[str, int]:
    """Qdrant 프로세스의 스레드들이 각각 어떤 상태인지 센다.

    D = uninterruptible sleep. 커널이 I/O 를 기다리는 중이고 깨울 수도 없는 상태다.
        멈춘 동안 D 가 있으면 원인은 파일시스템이다.
    R = 실행 중, S = 그냥 대기(락 포함).
    멈췄는데 D 가 하나도 없고 전부 S 면 I/O 가 아니라 락이다.
    """
    try:
        out = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             "cat /proc/1/task/*/stat"],
            capture_output=True, text=True, timeout=30, shell=True)
    except Exception:
        return {}
    states: dict[str, int] = {}
    for line in out.stdout.splitlines():
        # "pid (comm) S ..." 인데 comm 안에 공백이 있을 수 있어 ')' 뒤에서 자른다
        _, _, rest = line.partition(") ")
        if rest:
            states[rest[0]] = states.get(rest[0], 0) + 1
    return states


def telemetry(url: str, api_key: str) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/telemetry?details_level=3")
    if api_key:
        request.add_header("api-key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8")).get("result", {})
    except Exception:
        return {}


def dig(node, *path, default=None):
    for part in path:
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            return default
    return node


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="검색 중 Qdrant/VM 상태 관찰")
    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    p.add_argument("--api-key", default=C.QDRANT_API_KEY)
    p.add_argument("--container", default="kisti_openalex_vdb")
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--count", type=int, default=0, help="0 이면 Ctrl+C 까지")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[vitals] {args.qdrant_url} / {args.container}  "
          f"{args.interval:.0f}초 간격  (Ctrl+C 로 종료)")
    print()
    header = (f"{'시각':>8} {'상태':>7} {'opt':>12} {'queue':>6} "
              f"{'실행R':>6} {'IO대기D':>8} {'대기S':>6} {'스레드':>6} "
              f"{'free':>7} {'cache':>7} {'mapped':>7} {'cpu델타':>10}")
    print(header)
    print("-" * len(header))

    previous_cpu: float | None = None
    previous_at: float | None = None
    turn = 0

    while not args.count or turn < args.count:
        turn += 1
        now = time.perf_counter()
        tel = telemetry(args.qdrant_url, args.api_key)
        mem = meminfo(args.container)
        states = thread_states(args.container)

        shard = dig(tel, "collections", "collections", 0, "shards", 0, "local",
                    default={})
        status = str(dig(shard, "status", default="?"))[:7]
        optimized = int(dig(shard, "total_optimized_points", default=0) or 0)
        queue = int(dig(shard, "update_queue", "length", default=0) or 0)

        cpu_now = dig(tel, "hardware", "collection_data", C.COLLECTION, "cpu",
                      default=None)
        # Qdrant 의 cpu 카운터는 단위가 명시돼 있지 않다. 절대값은 믿지 말고
        # "움직이는가 / 멈춰 있는가" 만 본다.
        delta = "-"
        if cpu_now is not None:
            if previous_cpu is not None and previous_at is not None:
                span = now - previous_at
                delta = f"{(cpu_now - previous_cpu) / span:,.0f}" if span else "-"
            previous_cpu, previous_at = float(cpu_now), now

        def gib(key: str) -> str:
            return f"{mem[key] / GIB:.1f}" if key in mem else "-"

        running = states.get("R", 0)
        blocked = states.get("D", 0)
        sleeping = states.get("S", 0)
        total = sum(states.values())

        print(f"{time.strftime('%H:%M:%S'):>8} {status:>7} {optimized:>12,} "
              f"{queue:>6} {running:>6} {blocked:>8} {sleeping:>6} {total:>6} "
              f"{gib('MemFree'):>7} {gib('Cached'):>7} {gib('Mapped'):>7} "
              f"{delta:>10}")

        if args.count and turn >= args.count:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[vitals] 중단")
