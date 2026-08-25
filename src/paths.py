#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""파트 파일(part_XXXX.jsonl)과 거기 딸린 산출물의 경로를 다룬다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import config as C


@dataclass(frozen=True)
class Part:
    """part_XXXX.jsonl 하나와, 같은 디렉토리에 같은 이름으로 딸린 산출물들."""

    jsonl: Path

    @property
    def vec(self) -> Path:
        return self.jsonl.with_suffix(".f16")

    @property
    def posted_marker(self) -> Path:
        return self.jsonl.with_suffix(".posted")

    @property
    def year(self) -> str:
        return self.jsonl.parent.name

    @property
    def label(self) -> str:
        return f"{self.year}/{self.jsonl.name}"

    # -- 임베딩 진행 위치 ---------------------------------------------------

    def embedded_rows(self) -> int:
        """.f16 에 기록된 온전한 row 수. 그대로 '이미 처리한 줄 수'다."""
        if not self.vec.exists():
            return 0
        return self.vec.stat().st_size // C.ROW_BYTES

    def trim_partial_row(self) -> None:
        """중단으로 잘린 마지막 row 를 제거해 파일을 row 경계에 맞춘다."""
        if not self.vec.exists():
            return
        size = self.vec.stat().st_size
        remainder = size % C.ROW_BYTES
        if remainder:
            with self.vec.open("r+b") as handle:
                handle.truncate(size - remainder)

    # -- 적재 진행 위치 -----------------------------------------------------

    def posted_rows(self) -> int:
        try:
            return int(self.posted_marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    def set_posted_rows(self, count: int) -> None:
        tmp = self.jsonl.with_suffix(".posted.tmp")
        tmp.write_text(f"{count}\n", encoding="utf-8")
        tmp.replace(self.posted_marker)   # 원자적 교체


def discover_parts(root: Path, years: list[str] | None = None) -> list[Part]:
    """연도 디렉토리를 순회하며 Part 목록을 만든다."""
    parts: list[Part] = []
    for year in years or C.YEARS:
        year_dir = root / year
        if not year_dir.is_dir():
            continue
        for jsonl in sorted(year_dir.glob("*.jsonl"), key=lambda p: p.name):
            parts.append(Part(jsonl=jsonl))
    return parts


def count_lines(path: Path, chunk: int = 8 << 20) -> int:
    """2GB 파일도 몇 초: 바이너리 청크로 개행만 센다."""
    total, last = 0, b"\n"
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            total += data.count(b"\n")
            last = data[-1:]
    return total + (1 if last != b"\n" else 0)


def iter_lines_from(path: Path, skip: int) -> Iterator[tuple[int, str]]:
    """skip 줄을 건너뛴 뒤 (줄번호, 줄) 을 스트리밍. 파일을 통째로 올리지 않는다."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no > skip:
                yield line_no, line


def human_bytes(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
