#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""파트 단위 임베딩 루프.

resume 은 .f16 파일 크기가 곧 진행 위치라 별도 상태 파일이 없다.
같은 명령을 다시 실행하면 이어서 간다.
"""

from __future__ import annotations

from tqdm.auto import tqdm

from . import config as C
from . import embedding, paths, records, vectors
from .paths import Part


def embed_part(part: Part, model, *, batch_size: int, position: int,
               total: int, limit_rows: int = 0) -> dict[str, int]:
    part.trim_partial_row()

    done = part.embedded_rows()
    total_lines = paths.count_lines(part.jsonl)
    stats = {"embedded": 0, "empty": 0}
    if done >= total_lines:
        return stats

    bar = tqdm(
        total=total_lines, initial=done,
        desc=f"[{position}/{total}] {part.label}",
        unit="row", position=1, leave=False, dynamic_ncols=True, smoothing=0.05,
    )

    buffer: list[str] = []      # 배치 내 각 줄의 텍스트 ("" 면 임베딩 안 함)
    processed = 0

    def flush(upto_line: int) -> None:
        """모아둔 줄을 임베딩해 .f16 에 append. 파일 크기가 곧 체크포인트."""
        nonlocal buffer
        if not buffer:
            return

        block = vectors.zero_block(len(buffer))
        targets = [i for i, text in enumerate(buffer) if text]
        if targets:
            encoded = embedding.encode(model, [buffer[i] for i in targets], bar.write)
            block[targets] = encoded

        vectors.append_block(part.vec, block)

        stats["embedded"] += len(targets)
        stats["empty"] += len(buffer) - len(targets)
        buffer = []
        bar.update(upto_line - bar.n)
        bar.set_postfix(empty=stats["empty"], refresh=False)

    try:
        line_no = done
        for line_no, line in paths.iter_lines_from(part.jsonl, done):
            _, text = records.parse_line(line)
            buffer.append(text)

            if len(buffer) >= batch_size:
                flush(line_no)
                processed += batch_size
                if limit_rows and processed >= limit_rows:
                    bar.write(f"[info] --limit-rows 도달, {part.jsonl.name} 중단")
                    return stats
        flush(line_no)
    finally:
        bar.close()

    return stats


def run(parts: list[Part], model, *, batch_size: int, limit_rows: int = 0) -> dict[str, int]:
    grand = {"embedded": 0, "empty": 0}
    outer = tqdm(parts, desc="parts", unit="part", position=0, dynamic_ncols=True)
    try:
        for index, part in enumerate(outer, start=1):
            outer.set_postfix_str(part.label)
            stats = embed_part(part, model, batch_size=batch_size, position=index,
                               total=len(parts), limit_rows=limit_rows)
            for key in grand:
                grand[key] += stats[key]
    finally:
        outer.close()
    return grand


def summary(parts: list[Part]) -> str:
    done = sum(p.embedded_rows() for p in parts)
    size = sum(p.vec.stat().st_size for p in parts if p.vec.exists())
    return f"{done:,} rows 임베딩 완료 ({paths.human_bytes(size)}, {C.ROW_BYTES // 1024} KiB/row)"
