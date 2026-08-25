#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""적재 진입점: 컬렉션 준비 + .f16/jsonl -> Qdrant

    python post.py                      # 컬렉션 없으면 만들고 이어서 적재
    python post.py --recreate           # 컬렉션을 지우고 처음부터
    python post.py --status             # 상태만 보기
    python post.py --indexes drop --keep type,publication_year

모델을 띄우지 않는다. GPU 가 없어도 된다.

resume: 파트마다 원본 옆 part_XXXX.posted 에 '적재 완료한 줄 번호' 한 줄만 남긴다.
point id 가 openalex_id 로부터 결정되므로 겹쳐 올려도 덮어쓰기만 된다.
"""

from __future__ import annotations

import argparse
import sys

from src import collection as coll
from src import config as C
from src import paths, store, upload


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="임베딩(.f16) + jsonl -> Qdrant 적재",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default=None, help=f"기본: {C.DATA_ROOT}")
    p.add_argument("--years", nargs="+", default=C.YEARS)

    p.add_argument("--qdrant-url", default=C.QDRANT_URL)
    p.add_argument("--api-key", default=C.QDRANT_API_KEY)
    p.add_argument("--timeout", type=int, default=300, help="요청 타임아웃(초)")
    p.add_argument("--prefer-grpc", action="store_true", help="gRPC 사용")
    p.add_argument("--retries", type=int, default=6)

    p.add_argument("--batch-size", type=int, default=1024, help="upsert 1회당 row 수")
    p.add_argument("--workers", type=int, default=1, help="동시 upsert 스레드")
    p.add_argument("--restart", action="store_true", help="적재 진행 위치 초기화")

    g = p.add_argument_group("컬렉션 관리 (지정하면 적재는 하지 않음)")
    g.add_argument("--status", action="store_true", help="상태만 출력")
    g.add_argument("--recreate", action="store_true", help="삭제 후 재생성")
    g.add_argument("--yes", action="store_true", help="확인 프롬프트 건너뜀")
    g.add_argument("--segments", type=int, default=C.SEGMENT_NUMBER,
                   help="세그먼트 개수. 램 오버헤드의 곱셈 계수라 적게 잡는다")
    g.add_argument("--bulk", choices=["on", "off"],
                   help="on=적재 중 HNSW 빌드 중단 / off=재개")
    g.add_argument("--indexes", choices=["drop", "create"],
                   help="payload 인덱스를 떼거나 다시 만든다. 데이터는 그대로")
    g.add_argument("--keep", default="", help="--indexes 와 함께. 남길 필드를 콤마로")
    return p.parse_args(argv)


def manage(args, qdrant) -> int | None:
    """관리 명령이면 처리하고 종료 코드를 반환. 아니면 None."""
    exists = qdrant.collection_exists(C.COLLECTION)
    keep = {f.strip() for f in args.keep.split(",") if f.strip()}

    if args.indexes:
        if not exists:
            print(f"[error] '{C.COLLECTION}' 이 없습니다", file=sys.stderr)
            return 2
        if args.indexes == "drop":
            count = coll.drop_indexes(qdrant, keep)
            print(f"\n[ok] {count}개 제거. 데이터는 그대로입니다.")
        else:
            count = coll.create_indexes(qdrant, keep)
            print(f"\n[ok] {count}개 생성 시작 (백그라운드)")
        coll.describe(qdrant)
        return 0

    if args.bulk:
        if not exists:
            print(f"[error] '{C.COLLECTION}' 이 없습니다", file=sys.stderr)
            return 2
        coll.set_bulk_mode(qdrant, args.bulk == "on")
        if args.bulk == "on":
            print("[ok] 인덱싱 중단. 적재가 끝나면 반드시 --bulk off")
        else:
            print("[ok] 인덱싱 재개. status 가 green 이 될 때까지 기다리세요")
        coll.describe(qdrant)
        return 0

    if args.status:
        if not exists:
            print(f"[info] '{C.COLLECTION}' 이 아직 없습니다")
            return 0
        coll.describe(qdrant)
        return 0

    if exists and args.recreate:
        info = qdrant.get_collection(C.COLLECTION)
        print(f"[warn] '{C.COLLECTION}' 삭제 예정 (현재 {info.points_count:,} points)")
        if not args.yes and input("정말 삭제할까요? 'yes' 입력: ").strip() != "yes":
            print("[abort] 취소됨")
            return 1
        qdrant.delete_collection(C.COLLECTION)
        print("[ok] 삭제 완료")
        exists = False

    if not exists:
        coll.create(qdrant, segments=args.segments)
    return None


def main(argv=None) -> int:
    args = parse_args(argv)

    qdrant = coll.client(args.qdrant_url, args.api_key, args.timeout)
    early = manage(args, qdrant)
    if early is not None:
        return early

    root = C.resolve_data_root(args.data_root)
    if not root.is_dir():
        print(f"[error] 데이터 루트 없음: {root}", file=sys.stderr)
        return 2

    parts = paths.discover_parts(root, args.years)
    if not parts:
        print(f"[error] {root} 아래 {args.years} 에 .jsonl 없음", file=sys.stderr)
        return 2

    if args.restart:
        for part in parts:
            part.posted_marker.unlink(missing_ok=True)
        print("[info] --restart: 적재 진행 위치 초기화")

    missing = [p for p in parts if not p.vec.exists()]
    if missing:
        print(f"[warn] 임베딩 없는 파트 {len(missing)}개는 건너뜁니다 "
              f"(예: {missing[0].label})")

    print(f"[data]  {root}  ({len(parts)} parts)")
    print(f"[state] {upload.summary(parts)}")

    vector_store, embeddings = store.build_upload_store(
        url=args.qdrant_url, api_key=args.api_key,
        timeout=args.timeout, prefer_grpc=args.prefer_grpc,
    )

    rc = 0
    try:
        stats = upload.run(parts, vector_store, embeddings,
                           batch_size=args.batch_size, workers=args.workers,
                           retries=args.retries)
    except KeyboardInterrupt:
        print("\n[stop] 중단됨. 같은 명령을 다시 실행하면 이어서 진행됩니다.")
        return 130
    except Exception as exc:
        print(f"\n[fatal] {type(exc).__name__}: {exc}")
        print("[stop] 진행 위치는 저장되어 있습니다.")
        return 1

    print(f"\n[done] upsert {stats['upserted']:,} "
          f"| 스킵(빈 abstract) {stats['skipped']:,}")
    coll.describe(qdrant)
    print("\n[next] python api.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
