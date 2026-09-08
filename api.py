#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API 진입점. 검색 서버와 cloudflared 터널을 한 번에 띄운다.

    python api.py                     # 로컬 + 퀵 터널 (외부 주소 자동 발급)
    python api.py --no-tunnel         # 로컬만
    python api.py --tunnel-name mytun # named tunnel (주소 고정)

API 키는 .env 의 OPENALEX_VDB_API_KEY 를 쓴다. 비어 있으면 실행할 때마다 새로 만들어
콘솔에 찍는다. 고정하려면 .env 에 적어두면 된다.

두 프로세스를 따로 띄울 필요는 없다. 이 스크립트가
  1) uvicorn (FastAPI 검색 서버)
  2) cloudflared (외부 노출 터널)
를 같이 관리하고, Ctrl+C 하면 둘 다 정리한다.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import threading

from src import config as C


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OpenAlex 검색 API + 터널",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=C.API_HOST,
                   help="터널만 쓸 거면 127.0.0.1 로 두세요 (외부 직접 노출 없음)")
    p.add_argument("--port", type=int, default=C.API_PORT)
    p.add_argument("--device", default="auto",
                   help="임베딩/리랭커를 올릴 장치. auto | cuda | cuda:0 | cpu")
    p.add_argument("--reranker-device", default=None,
                   help="리랭커만 다른 장치에 올릴 때. 기본은 --device 와 같음")
    p.add_argument("--no-reranker", action="store_true",
                   help="리랭커를 올리지 않음 (VRAM 절약, 순위 품질 하락)")
    p.add_argument("--no-tf32", action="store_true",
                   help="Ampere TF32 matmul 끄기 (기본은 켬)")
    p.add_argument("--gpu-memory", type=float, default=C.GPU_MEMORY_GIB,
                   metavar="GIB",
                   help="이 프로세스가 쓸 VRAM 상한(GiB). 0=제한 없음. "
                        "같은 GPU 에 다른 모델을 올릴 때 쓴다")
    p.add_argument("--api-key", default="", help="비우면 .env 값, 그것도 없으면 임의 생성")

    p.add_argument("--no-tunnel", action="store_true", help="cloudflared 를 띄우지 않음")
    p.add_argument("--tunnel-name", default=C.TUNNEL_NAME,
                   help="named tunnel 이름. 비우면 퀵 터널(주소가 매번 바뀜)")
    p.add_argument("--cloudflared", default=C.CLOUDFLARED_BIN)
    p.add_argument("--warmup", type=int, default=C.WARMUP_QUERIES, metavar="N",
                   help="기동 직후 캐시를 데울 질의 수 (0=안 함). 백그라운드로 돈다")
    p.add_argument("--log-level", default="warning",
                   help="uvicorn 로그 레벨. info 로 하면 요청이 전부 찍힌다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import uvicorn

    from api import server
    from api.tunnel import Tunnel

    key = args.api_key or C.API_KEY or secrets.token_urlsafe(24)

    print("[api] 모델과 컬렉션을 여는 중입니다...")
    server.build(device=args.device, api_key=key,
                 with_reranker=not args.no_reranker,
                 reranker_device=args.reranker_device,
                 tf32=not args.no_tf32, memory_gib=args.gpu_memory)

    if args.warmup:
        threading.Thread(target=server.warmup, args=(args.warmup,),
                         daemon=True).start()

    tunnel = None
    if not args.no_tunnel:
        tunnel = Tunnel(args.port, name=args.tunnel_name, binary=args.cloudflared)
        tunnel.start()

    print()
    print(f"[api]  http://{args.host}:{args.port}   (문서: /docs)")
    print(f"[api]  POST /search   의미 검색  (top {C.SEARCH_CANDIDATES} -> 리랭커 -> top {C.SEARCH_LIMIT})")
    print(f"[api]  POST /keyword  제목 키워드 (top {C.KEYWORD_LIMIT})")
    print(f"[key]  X-API-Key: {key}")
    if not (args.api_key or C.API_KEY):
        print("       고정하려면 .env 에  OPENALEX_VDB_API_KEY=" + key)
    if tunnel and tunnel.name:
        print(f"[url]  named tunnel '{tunnel.name}' - 설정한 호스트명으로 접속")
    elif tunnel:
        print("[url]  퀵 터널 주소는 잠시 뒤 아래에 표시됩니다")
    print()

    try:
        uvicorn.run(server.app, host=args.host, port=args.port,
                    log_level=args.log_level)
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.stop()
        print("\n[api] 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
