#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""파트 단위 Qdrant 적재 루프.

.f16 의 N번째 row 가 .jsonl 의 N번째 줄이라는 규칙 덕분에, 벡터를 순서대로
읽으면서 같은 줄의 payload 와 짝지으면 된다. 진행 위치는 .posted 한 줄.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from tqdm.auto import tqdm

from . import config as C
from . import paths, records, vectors
from .paths import Part


def upsert(store, embeddings, ids, texts, payloads, block, retries: int, warn) -> None:
    """add_texts 의 **kwargs 는 client.upsert 로 전달되므로 wait 를 여기서 넘긴다."""
    delay = 2.0
    for attempt in range(1, retries + 1):
        embeddings.feed(block)   # 시도마다 다시 넣는다 (embed_documents 가 소비함)
        try:
            store.add_texts(texts=texts, metadatas=payloads, ids=ids,
                            batch_size=len(ids), wait=False)
            return
        except Exception as exc:
            if attempt == retries:
                raise
            warn(f"[warn] upsert 실패 {attempt}/{retries}: "
                 f"{type(exc).__name__}: {exc} -> {delay:.0f}s 후 재시도")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def upload_part(part: Part, store, embeddings, *, batch_size: int, workers: int,
                retries: int, position: int, total: int) -> dict[str, int]:
    stats = {"upserted": 0, "skipped": 0}
    if not part.vec.exists():
        return stats

    available = part.embedded_rows()   # 임베딩이 끝난 줄 수 (= .f16 크기 / row 크기)
    done = part.posted_rows()
    if done >= available:
        return stats

    bar = tqdm(
        total=available, initial=done,
        desc=f"[{position}/{total}] {part.label}",
        unit="row", position=1, leave=False, dynamic_ncols=True, smoothing=0.05,
    )
    handle = vectors.open_reader(part.vec, done)

    buffer: list[tuple[int, dict | None, str]] = []   # (줄번호, row, text)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="upsert")
    inflight: deque[tuple[Future, int, int]] = deque()   # (future, 마지막 줄, 건수)

    def drain_one() -> None:
        future, last_line, count = inflight.popleft()
        future.result()                    # 실패하면 예외 -> 체크포인트 전진 안 함
        stats["upserted"] += count
        part.set_posted_rows(last_line)    # 반영 확인 후에만 전진
        bar.update(last_line - bar.n)
        bar.set_postfix(ok=stats["upserted"], skip=stats["skipped"], refresh=False)

    def drain(force: bool) -> None:
        while inflight and (force or inflight[0][0].done()):
            drain_one()

    def flush() -> None:
        if not buffer:
            return
        block = vectors.read_rows(handle, len(buffer))   # 줄 순서 == row 순서

        ids, texts, payloads, offsets = [], [], [], []
        for offset, (_, row, text) in enumerate(buffer):
            if row is None or not text:
                continue                   # 임베딩 때 영벡터를 넣어둔 자리
            ids.append(records.point_id(row["id"]))
            texts.append(text)
            payloads.append(records.to_payload(row))
            offsets.append(offset)

        selected = block[offsets] if offsets else block[:0]
        bad = vectors.find_zero_row(selected)
        if bad is not None:
            line_no = buffer[offsets[bad]][0]
            msg = (f"{part.label} line {line_no}: abstract 가 있는데 벡터가 비어 있습니다. "
                   f"이 파트의 .f16 을 지우고 임베딩을 다시 돌리세요")
            raise RuntimeError(msg)

        stats["skipped"] += len(buffer) - len(ids)
        last_line = buffer[-1][0]
        buffer.clear()

        if not ids:                        # 전부 스킵된 배치
            drain(force=True)              # 앞선 배치보다 먼저 전진하면 안 된다
            part.set_posted_rows(last_line)
            bar.update(last_line - bar.n)
            return

        if len(inflight) >= workers:       # 큐가 차면 맨 앞 하나만 기다린다
            drain_one()
        inflight.append((
            pool.submit(upsert, store, embeddings, ids, texts, payloads,
                        selected.tolist(), retries, bar.write),
            last_line, len(ids),
        ))
        drain(force=False)

    try:
        for line_no, line in paths.iter_lines_from(part.jsonl, done):
            if line_no > available:
                break                      # 아직 임베딩 안 된 뒷부분
            buffer.append((line_no, *records.parse_line(line)))
            if len(buffer) >= batch_size:
                flush()
        flush()
    finally:
        try:
            drain(force=True)              # 남은 in-flight 까지 반영하고 확정
        except Exception as exc:
            bar.write(f"[warn] 남은 배치 반영 실패: {type(exc).__name__}: {exc}")
        pool.shutdown(wait=True)
        handle.close()
        bar.close()

    return stats


def run(parts: list[Part], store, embeddings, *, batch_size: int,
        workers: int, retries: int) -> dict[str, int]:
    grand = {"upserted": 0, "skipped": 0}
    outer = tqdm(parts, desc="parts", unit="part", position=0, dynamic_ncols=True)
    try:
        for index, part in enumerate(outer, start=1):
            outer.set_postfix_str(part.label)
            stats = upload_part(part, store, embeddings, batch_size=batch_size,
                                workers=workers, retries=retries,
                                position=index, total=len(parts))
            for key in grand:
                grand[key] += stats[key]
    finally:
        outer.close()
    return grand


def summary(parts: list[Part]) -> str:
    ready = sum(p.embedded_rows() for p in parts)
    posted = sum(p.posted_rows() for p in parts)
    return f"{ready:,} rows 임베딩 완료 / {posted:,} rows 적재 완료"
