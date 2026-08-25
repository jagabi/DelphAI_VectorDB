#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""전처리 진입점: parquet -> 연도별 jsonl -> 초록 채우기 -> 임베딩(.f16)

    python preprocess.py                    # 전체 (filter -> abstract -> embed)
    python preprocess.py --stage embed      # 임베딩만
    python preprocess.py --stage filter     # parquet 필터링만
    python preprocess.py --stage abstract   # 초록 채우기만

임베딩은 .f16 파일 크기가 곧 진행 위치라, 끊겨도 같은 명령을 다시 실행하면 이어간다.
"""

from __future__ import annotations

import argparse
import sys

from src import config as C
from src import embed_runner, embedding, paths

STAGES = ("filter", "abstract", "embed")


def run_embed(args) -> int:
    root = C.resolve_data_root(args.data_root)
    if not root.is_dir():
        print(f"[error] 데이터 루트 없음: {root}", file=sys.stderr)
        return 2

    parts = paths.discover_parts(root, args.years)
    if not parts:
        print(f"[error] {root} 아래 {args.years} 에 .jsonl 없음", file=sys.stderr)
        return 2

    source = sum(p.jsonl.stat().st_size for p in parts)
    print(f"[data]  {root}  ({len(parts)} parts, {paths.human_bytes(source)})")
    print(f"[state] {embed_runner.summary(parts)}")

    model, device = embedding.build(
        args.device, batch_size=args.embed_batch,
        max_seq_len=args.max_seq_len, fp32=args.fp32,
        show_progress=args.show_progress, tf32=not args.no_tf32,
        memory_gib=args.gpu_memory,
    )
    encode_batch = model.encode_kwargs.get("batch_size")
    print(f"[model] {C.MODEL_NAME} on {device} | max_len={args.max_seq_len} "
          f"| encode_batch={encode_batch}")
    if device == "cpu":
        print("[warn] GPU가 없습니다. bge-m3(568M)는 CPU에서 매우 느립니다.")

    try:
        stats = embed_runner.run(parts, model, batch_size=args.batch_size,
                                 limit_rows=args.limit_rows)
    except KeyboardInterrupt:
        print("\n[stop] 중단됨. 같은 명령을 다시 실행하면 이어서 진행됩니다.")
        return 130

    print(f"\n[done] 임베딩 {stats['embedded']:,} | 빈 abstract {stats['empty']:,}")
    print(f"[done] {embed_runner.summary(parts)}")
    print("[next] python post.py")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OpenAlex 전처리 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    p.add_argument("--data-root", default=None, help=f"기본: {C.DATA_ROOT}")
    p.add_argument("--years", nargs="+", default=C.YEARS)

    g = p.add_argument_group("embed")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    g.add_argument("--fp32", action="store_true", help="GPU에서도 fp16 대신 fp32")
    g.add_argument("--max-seq-len", type=int, default=C.MAX_SEQ_LEN)
    g.add_argument("--embed-batch", type=int, default=None,
                   help=f"모델 forward 배치. 기본은 장치에 따라 "
                        f"cuda={C.EMBED_BATCH_CUDA} / cpu={C.EMBED_BATCH_CPU}")
    g.add_argument("--no-tf32", action="store_true",
                   help="Ampere TF32 matmul 끄기 (기본은 켬)")
    g.add_argument("--gpu-memory", type=float, default=C.GPU_MEMORY_GIB,
                   metavar="GIB",
                   help="VRAM 상한(GiB). 0=제한 없음")
    g.add_argument("--batch-size", type=int, default=512,
                   help="파일에 한 번에 append 할 row 수 = 진행바/재개 단위")
    g.add_argument("--show-progress", action="store_true", help="모델 자체 진행바")
    g.add_argument("--limit-rows", type=int, default=0, help="파트당 최대 row (0=제한없음)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = STAGES if args.stage == "all" else (args.stage,)

    for stage in stages:
        print(f"\n{'=' * 70}\n  {stage}\n{'=' * 70}")
        if stage == "filter":
            from src.preprocess import filter as filter_stage
            filter_stage.main()
        elif stage == "abstract":
            from src.preprocess import fill_abs
            fill_abs.main()
        else:
            code = run_embed(args)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
