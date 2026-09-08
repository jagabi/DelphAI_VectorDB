#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qdrant 스토리지를 순차 읽기로 OS 캐시에 올린다.

검색이 느려지는 건 대개 페이지 캐시가 비었을 때다. 그 상태에서 검색을 던지면
HNSW 가 그래프를 무작위로 점프하며 4KB 씩 수천 번 폴트를 내는데, 이 계층에서
랜덤 읽기는 초당 몇 건 수준이라 사실상 안 끝난다.

같은 데이터를 순차로 미리 읽으면 수백 MB/s 가 나온다. 그래서 검색이 필요로 하는
파일을 통째로 한 번 훑어 캐시에 올려두는 것이 훨씬 빠르다.

Windows 호스트에서 직접 읽는다. Docker 바인드 마운트는 결국 이 파일을 읽으므로
호스트 캐시가 데워지면 컨테이너 쪽 읽기도 같이 빨라진다.

    python scripts/warm_storage.py
    python scripts/warm_storage.py --path G:/OpenAlex_VectorDB/qdrant
    python scripts/warm_storage.py --all        # payload/원본벡터까지 전부
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# 검색 경로가 실제로 만지는 파일들. 이 둘만 데워도 대부분 해결된다.
SEARCH_FILES = ("quantized.data", "links_compressed.bin")

# --all 일 때 추가로 읽을 것. 원본 벡터(rescore)와 payload 본문.
EXTRA_FILES = ("matrix.dat", "postings.dat", "point_to_values.bin",
               "tracker.dat", "page_0.dat")

CHUNK = 8 << 20   # 8 MiB


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


def collect(root: Path, names: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for name in names:
        found.extend(root.rglob(name))
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def warm(files: list[Path]) -> tuple[int, float]:
    """파일들을 순차로 읽는다. 내용은 버리고 캐시에 올리는 것이 목적."""
    total = sum(f.stat().st_size for f in files)
    print(f"[warm] {len(files)}개 파일 / {human(total)}\n")

    read = 0
    began = time.perf_counter()
    for index, path in enumerate(files, start=1):
        size = path.stat().st_size
        started = time.perf_counter()
        try:
            with path.open("rb", buffering=0) as handle:
                while handle.readinto(bytearray(CHUNK)):
                    pass
        except OSError as exc:
            print(f"  [skip] {path.name}: {exc}")
            continue
        read += size
        elapsed = time.perf_counter() - started
        rate = size / elapsed / (1 << 20) if elapsed else 0
        done = time.perf_counter() - began
        print(f"  [{index:3d}/{len(files)}] {human(size):>10} "
              f"{rate:6.0f} MiB/s   누적 {human(read)} / {done:5.0f}s")

    return read, time.perf_counter() - began


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Qdrant 스토리지를 순차 읽기로 캐시에 올린다",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path", default="G:/OpenAlex_VectorDB/qdrant",
                   help="Qdrant 스토리지 경로 (Windows 쪽 경로)")
    p.add_argument("--all", action="store_true",
                   help="원본 벡터와 payload 까지 전부. 훨씬 오래 걸린다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.path)
    if not root.is_dir():
        print(f"[error] 경로 없음: {root}")
        return 2

    names = SEARCH_FILES + (EXTRA_FILES if args.all else ())
    files = collect(root, names)
    if not files:
        print(f"[error] {root} 아래에서 {names} 를 찾지 못했습니다")
        return 2

    read, elapsed = warm(files)
    rate = read / elapsed / (1 << 20) if elapsed else 0
    print(f"\n[done] {human(read)} 를 {elapsed:.0f}초에 읽음 ({rate:.0f} MiB/s)")
    print("[next] python scripts/probe.py  로 검색이 돌아왔는지 확인하세요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
