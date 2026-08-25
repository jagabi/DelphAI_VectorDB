#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""외부 PC 에서 쓰는 클라이언트. 파이썬 표준 라이브러리만 쓴다.

이 파일 하나만 복사해 가면 되고, torch 도 모델도 qdrant-client 도 필요 없다.

접속 정보는 client.py 옆(또는 현재 디렉토리)의 .env 에서 읽는다:

    OPENALEX_VDB_URL=https://xxxx.trycloudflare.com
    OPENALEX_VDB_API_KEY=...

우선순위는 명령행 인자 > 환경변수 > .env > 기본값.
다른 파일을 쓰려면 --env 로 경로를 준다.

검색 방식 두 가지:

    python client.py "swarm robotics aggregation"
        의미(벡터) 검색. 기본값. 벡터로 50개 뽑아 리랭커로 10개.

    python client.py --keyword-search "swarm robotics"
        제목 키워드 검색. 기본 5개.

필터는 두 방식 모두 동일하게 쓸 수 있다:

    python client.py "graph neural network" --year-from 2024 --field "Computer Science"
    python client.py --keyword-search "diffusion model" --limit 10 --country KR
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

VECTOR_LIMIT = 10
VECTOR_CANDIDATES = 50
KEYWORD_LIMIT = 5

ENV_URL = "OPENALEX_VDB_URL"
ENV_KEY = "OPENALEX_VDB_API_KEY"


def load_env(path: str | None = None) -> Path | None:
    """.env 를 읽어 환경변수로 올린다. 이미 설정된 값은 덮어쓰지 않는다.

    의존성을 늘리지 않으려고 KEY=VALUE 만 직접 파싱한다.
    """
    if path:
        candidates = [Path(path)]
    else:
        here = Path(__file__).resolve().parent
        candidates = [here / ".env", Path.cwd() / ".env"]

    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return candidate
    return None


def _env_path_from(argv: list[str]) -> str | None:
    """argparse 기본값이 환경변수를 읽으므로 --env 는 파싱 전에 미리 본다."""
    for i, arg in enumerate(argv):
        if arg == "--env" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return None


def clean(text: str | None) -> str:
    """OpenAlex 초록에는 &#13; &gt; 같은 HTML 엔티티와 줄바꿈이 섞여 있다."""
    return " ".join(html.unescape(text or "").split())


def call(url: str, path: str, key: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:                                   # 서버가 실어준 예외를 풀어준다
            payload = json.loads(body)
            if "error" in payload:
                lines = [f"[error] HTTP {exc.code} {payload['error']}: "
                         f"{payload.get('detail', '')}"]
                if payload.get("hint"):
                    lines.append(f"        {payload['hint']}")
                raise SystemExit("\n".join(lines)) from None
            body = payload.get("detail", body)
        except ValueError:
            pass
        raise SystemExit(f"[error] HTTP {exc.code}: {str(body)[:800]}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"[error] 접속 실패: {exc.reason}") from None


def render(data: dict, args) -> None:
    head = (f"\n=== \"{data['query']}\" [{data['mode']}] "
            f"-> {data['count']}건 / {data['took_ms']} ms")
    extra = []
    if data.get("candidates"):
        extra.append(f"후보 {data['candidates']}")
    if data.get("reranked"):
        extra.append("리랭킹")
    if data.get("deduped"):
        extra.append(f"중복 -{data['deduped']}")
    print(head + (f" ({', '.join(extra)})" if extra else "") + " ===")

    for hit in data["hits"]:
        dup = f" x{hit['duplicates']}" if hit.get("duplicates", 1) > 1 else ""
        print(f"\n{hit['rank']:3d}. [{hit['score']:.4f}]{dup} "
              f"{clean(hit.get('title')) or '(제목 없음)'}")

        if args.meta:
            bits = [str(hit.get("publication_year") or "????")]
            bits += [b for b in (hit.get("source"), hit.get("topic")) if b]
            print(f"     {' | '.join(bits)}")
            if hit.get("vector_score") is not None and hit.get("rerank_score") is not None:
                print(f"     vector={hit['vector_score']:.4f}  "
                      f"rerank={hit['rerank_score']:.4f}")
            if hit.get("doi"):
                print(f"     {hit['doi']}")

        abstract = clean(hit.get("abstract"))
        if abstract and not args.no_abstract:
            if not args.full and len(abstract) > 300:
                abstract = abstract[:300] + "..."
            print(f"     {abstract}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="원격 OpenAlex 검색 클라이언트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("query", nargs="*", help="질의문 (없으면 대화형)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--vector-search", action="store_true",
                      help="의미 검색 (기본값)")
    mode.add_argument("--keyword-search", action="store_true",
                      help="제목 키워드 검색")

    p.add_argument("--limit", "-k", type=int, default=None,
                   help=f"결과 개수. 기본 vector={VECTOR_LIMIT}, keyword={KEYWORD_LIMIT}")

    v = p.add_argument_group("의미 검색")
    v.add_argument("--candidates", type=int, default=VECTOR_CANDIDATES,
                   help="리랭킹 전에 벡터로 뽑을 후보 수")
    v.add_argument("--no-rerank", action="store_true", help="리랭커 끄기 (더 빠름)")
    v.add_argument("--hnsw-ef", type=int, default=None,
                   help="HNSW 탐색 폭. 클수록 정확하고 느림 (기본은 서버 설정)")
    v.add_argument("--rescore", action="store_true",
                   help="원본 float32 로 후보 재측정. 리랭커를 끌 때만 의미 있다")

    kw = p.add_argument_group("키워드 검색")
    kw.add_argument("--phrase", action="store_true", help="구문 전체로 일치")
    kw.add_argument("--keyword-rerank", action="store_true",
                    help="찾은 것들을 질의 기준으로 재정렬")

    f = p.add_argument_group("필터 (인덱스가 걸린 필드)")
    f.add_argument("--year-from", type=int, metavar="YEAR")
    f.add_argument("--year-to", type=int, metavar="YEAR")
    f.add_argument("--field", metavar="NAME", help='예: "Computer Science"')
    f.add_argument("--subfield", metavar="NAME", help='예: "Artificial Intelligence"')
    f.add_argument("--domain", metavar="NAME", help='예: "Physical Sciences"')
    f.add_argument("--topic", metavar="NAME", help="세부 토픽 이름")
    f.add_argument("--source", metavar="NAME", help="저널/저장소 이름")
    f.add_argument("--type", metavar="TYPE", help="예: article, preprint")
    f.add_argument("--country", metavar="CC", help="예: KR, US")

    o = p.add_argument_group("출력")
    o.add_argument("--full", action="store_true", help="초록 전문 (기본 300자)")
    o.add_argument("--no-abstract", action="store_true", help="초록 생략")
    o.add_argument("--meta", action="store_true", help="연도/저널/토픽/점수/DOI")
    o.add_argument("--json", action="store_true", help="원본 JSON")
    o.add_argument("--no-dedupe", action="store_true", help="중복 제거 끄기")

    c = p.add_argument_group("접속 (.env 로 대신할 수 있음)")
    c.add_argument("--url", default=os.environ.get(ENV_URL, "http://127.0.0.1:8000"),
                   help=f".env 의 {ENV_URL}")
    c.add_argument("--key", default=os.environ.get(ENV_KEY, ""),
                   help=f".env 의 {ENV_KEY}")
    c.add_argument("--env", metavar="PATH", default=None,
                   help="쓸 .env 경로 (기본: client.py 옆 또는 현재 디렉토리)")
    c.add_argument("--timeout", type=int, default=120)
    return p.parse_args(argv)


def build_payload(query: str, args) -> tuple[str, dict]:
    filters = {
        "year_from": args.year_from, "year_to": args.year_to,
        "field": args.field, "subfield": args.subfield, "domain": args.domain,
        "topic": args.topic, "source": args.source, "type": args.type,
        "country": args.country,
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    dedupe = not args.no_dedupe

    if args.keyword_search:
        limit = args.limit if args.limit is not None else KEYWORD_LIMIT
        return "/keyword", {
            "query": query, "limit": limit, "phrase": args.phrase,
            "dedupe": dedupe, "rerank": args.keyword_rerank, **filters,
        }

    limit = args.limit if args.limit is not None else VECTOR_LIMIT
    return "/search", {
        "query": query, "limit": limit, "candidates": args.candidates,
        "rerank": not args.no_rerank, "dedupe": dedupe,
        "rescore": args.rescore,
        **({"hnsw_ef": args.hnsw_ef} if args.hnsw_ef else {}), **filters,
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env_file = load_env(_env_path_from(argv))
    args = parse_args(argv)

    if not args.key:
        target = env_file or Path(__file__).resolve().parent / ".env"
        print(f"[error] API 키가 없습니다. {target} 에 아래를 넣으세요:",
              file=sys.stderr)
        print(f"  {ENV_URL}=https://xxxx.trycloudflare.com", file=sys.stderr)
        print(f"  {ENV_KEY}=서버가_찍어준_키", file=sys.stderr)
        print("  (또는 --url / --key 로 직접 지정)", file=sys.stderr)
        return 2

    def run(query: str) -> None:
        path, payload = build_payload(query, args)
        data = call(args.url, path, args.key, payload, args.timeout)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            render(data, args)

    if args.query:
        run(" ".join(args.query))
        return 0

    mode = "keyword" if args.keyword_search else "vector"
    source = f"  <- {env_file}" if env_file else ""
    print(f"[api] {args.url}  ({mode} search){source}")
    print("질의를 입력하세요 (빈 줄이면 종료)")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        run(query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
