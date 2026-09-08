#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""전 파이프라인 공용 설정. 값은 전부 환경변수로 덮어쓸 수 있다.

.env 파일이 있으면 자동으로 읽는다 (python-dotenv 없이 직접 파싱).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """의존성 없이 KEY=VALUE 만 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 데이터 레이아웃
# ---------------------------------------------------------------------------
#
#   <DATA_ROOT>/2021/part_0000.jsonl     원본
#   <DATA_ROOT>/2021/part_0000.f16       임베딩      (preprocess.py 생성)
#   <DATA_ROOT>/2021/part_0000.posted    적재 위치   (post.py 생성)
#
# 핵심 규칙: .f16 의 N번째 row == .jsonl 의 N번째 줄.
# 빈 줄, 깨진 JSON, abstract 없는 문서도 자리를 비우지 않고 영벡터를 채운다.
# 덕분에 .f16 파일 크기가 곧 진행 위치라 별도 체크포인트가 필요 없다.

DATA_ROOT = _env("DATA_ROOT", "openalex_2021-2025_postprocessed")
YEARS = _env("YEARS", "2021,2022,2023,2024,2025").split(",")

TEXT_FIELD = "abstract"              # 임베딩 대상 = LangChain 의 page_content
KEY_MAP = {"id": "openalex_id"}      # raw jsonl key -> qdrant payload field

# ---------------------------------------------------------------------------
# 전처리 (filter.py / fill_abs.py)
# ---------------------------------------------------------------------------
#
# OpenAlex parquet 덤프의 위치. 전처리를 다시 돌릴 때만 필요하다.

PARQUET_ROOT = _env("PARQUET_ROOT", "OpenAlex_20260330_integrated_parsed_parquet")

SOURCES_SUBDIR = _env(
    "SOURCES_SUBDIR",
    "nonworks_20260330_update/openalex_sources_20260330_delta"
    "/openalex_sources_20260330_delta",
)
WORKS_SUBDIR = _env(
    "WORKS_SUBDIR",
    "works_20260330_integrated/openalex_works_20260330",
)
ABSTRACT_SUBDIR = _env(
    "ABSTRACT_SUBDIR",
    "works_20260330_integrated"
    "/openalex_works_20260330__excepted__abstract_inverted_index",
)

# 전처리 중간 산출물. 전부 DATA_ROOT 아래 (.gitignore 대상)
LOGS_SUBDIR = "logs"
CACHE_SUBDIR = "cache"
SIDECAR_SUBDIR = "_abs_sidecar"
FILTERED_ID_NAME = "filtered_id.jsonl"

ROWS_PER_PART = _env_int("ROWS_PER_PART", 1_000_000)
ATTACH_COUNTRY = _env("ATTACH_COUNTRY", "1") not in ("0", "false", "False")

# abstract 가 비었을 때 title 로 대체할지.
# preprocess 와 post 가 반드시 같은 값을 봐야 하므로 상수로 둔다.
FALLBACK_TO_TITLE = False


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

MODEL_NAME = _env("BGE_MODEL", "BAAI/bge-m3")
DIM = 1024
DTYPE = np.dtype("float16")          # row 당 2 KiB. float32 로 바꾸면 4 KiB
ROW_BYTES = DIM * DTYPE.itemsize
MAX_SEQ_LEN = _env_int("MAX_SEQ_LEN", 1024)

# forward 배치. 같은 GPU 를 다른 모델(gemma 등)과 나눠 쓸 것을 전제로 작게 잡았다.
# GPU 를 독점할 수 있으면 --embed-batch 256 처럼 직접 올리면 된다.
EMBED_BATCH_CUDA = _env_int("EMBED_BATCH_CUDA", 64)
EMBED_BATCH_CPU = _env_int("EMBED_BATCH_CPU", 16)

# Ampere 이상에서 TF32 matmul. 추론 정확도 손실은 무시할 수준이고 눈에 띄게 빠르다.
USE_TF32 = _env("USE_TF32", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# GPU 메모리 상한
# ---------------------------------------------------------------------------
#
# 같은 GPU 에 다른 모델을 같이 올릴 때, 이 프로세스가 쓸 수 있는 VRAM 을 제한한다.
# torch.cuda.set_per_process_memory_fraction 으로 캐싱 할당자의 상한을 건다.
# 넘으면 OOM 이 나지만, 임베딩/리랭커 모두 OOM 시 배치를 절반으로 줄여 재시도하므로
# 죽지 않고 상한 안에서 알아서 맞춰 돌아간다.
#
#   0  = 제한 없음 (GPU 독점)
#   12 = 12 GiB 까지만

GPU_MEMORY_GIB = float(_env("GPU_MEMORY_GIB", "0"))

# 요청 처리 후 캐싱 할당자가 붙들고 있는 블록을 드라이버에 돌려준다.
# 조금 느려지지만 옆에서 도는 모델이 그만큼 쓸 수 있게 된다.
RELEASE_CACHE = _env("RELEASE_CACHE", "1") not in ("0", "false", "False")

# 단편화를 줄여 상한 안에서 더 많이 쓸 수 있게 한다.
# torch 가 CUDA 를 초기화하기 전에 설정되어야 해서 여기서 건다.
# Windows 는 expandable_segments 를 지원하지 않아 경고만 뜨므로 건너뛴다.
if not sys.platform.startswith("win"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# 리랭커 (cross-encoder)
# ---------------------------------------------------------------------------
#
# bge-m3 의 짝. 같은 XLM-R large 계열이라 다국어 커버리지가 같다.
# 벡터 검색으로 후보를 좁힌 뒤 상위권 순서를 다시 매기는 데 쓴다.

RERANKER_MODEL = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = _env_int("RERANKER_MAX_LENGTH", 512)
RERANKER_BATCH_CUDA = _env_int("RERANKER_BATCH_CUDA", 16)
RERANKER_BATCH_CPU = _env_int("RERANKER_BATCH_CPU", 8)

# ---------------------------------------------------------------------------
# 검색 기본값
# ---------------------------------------------------------------------------

SEARCH_CANDIDATES = _env_int("SEARCH_CANDIDATES", 50)   # 벡터로 뽑는 후보 수
SEARCH_LIMIT = _env_int("SEARCH_LIMIT", 10)             # 리랭커 통과 후 최종 개수
KEYWORD_LIMIT = _env_int("KEYWORD_LIMIT", 5)            # 제목 키워드 검색 기본 개수

# HNSW 탐색 폭. 크면 정확하고 느리다.
SEARCH_HNSW_EF = _env_int("SEARCH_HNSW_EF", 64)

# rescore 는 int8 로 뽑은 후보를 원본 float32 로 다시 재는 단계다.
# 그런데 최종 순서는 어차피 리랭커(cross-encoder)가 정하므로, 후보 선별 단계의
# 양자화 오차는 리랭커가 흡수한다. 게다가 rescore 는 원본 벡터를 디스크에서
# 랜덤하게 읽어와 느린 스토리지에서는 검색 시간을 몇 배로 늘린다.
# 그래서 기본은 끄고, 리랭커를 안 쓸 때만 켜는 것을 권한다.
SEARCH_RESCORE = _env("SEARCH_RESCORE", "0") not in ("0", "false", "False")

# Qdrant 검색 요청 제한시간(초).
# 캐시가 완전히 식은 상태(오래 안 쓰다 켰을 때)에서는 배치 질의가 2분을 넘기기도 한다.
SEARCH_TIMEOUT = _env_int("SEARCH_TIMEOUT", 300)

# 기동 직후 캐시를 데울 질의 수. 0 이면 워밍업 안 함.
#
# 콜드 상태에서는 세그먼트와 벡터를 디스크에서 읽느라 첫 질의가 수십 초 걸린다.
# 미리 몇 번 훑어 두면 HNSW 그래프와 양자화 벡터가 페이지 캐시에 올라와,
# 실제 사용자의 첫 질의가 그 대가를 치르지 않는다. 백그라운드로 돈다.
WARMUP_QUERIES = _env_int("WARMUP_QUERIES", 4)


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = _env("QDRANT_COLLECTION", "openalex_2021-2025_abs")
VECTOR_NAME = ""                     # langchain-qdrant 의 기본 dense 벡터 이름

# 컬렉션 생성 파라미터
HNSW_M = _env_int("HNSW_M", 24)
HNSW_EF_CONSTRUCT = _env_int("HNSW_EF_CONSTRUCT", 256)
HNSW_PAYLOAD_M = _env_int("HNSW_PAYLOAD_M", 24)
SEGMENT_NUMBER = _env_int("SEGMENT_NUMBER", 16)
INDEXING_THRESHOLD = _env_int("INDEXING_THRESHOLD", 10000)

# payload 인덱스.
#
# 인덱스는 "필터 검색"을 위한 것이고 payload 저장과는 무관하다. 여기서 빼도
# 해당 필드는 그대로 저장되고 검색 결과에도 딸려 나온다.
#
# 인덱스 하나하나가 회수 불가능한 램을 점유하고, 그 비용이
# (인덱스 개수 x 세그먼트 개수) 로 곱해진다. payload_m 때문에 인덱싱된
# 필드마다 HNSW 에 추가 링크도 붙는다. 그래서 실제로 필터에 쓸 것만 남긴다.
#
# 제외한 것: abstract(전문 인덱스는 88M 기준 47 GiB, 의미 검색은 벡터가 담당),
#            publication_date(year 로 충분), openalex_id/doi/issn_l(전부 고유값),
#            *__id 계열(대응하는 *__display_name 과 1:1 중복), host_organization*

KEYWORD_INDEXES = [
    "type",
    "country_code",
    "primary_location__source__display_name",
    "primary_topic__display_name",
    "primary_topic__domain__display_name",
    "primary_topic__field__display_name",
    "primary_topic__subfield__display_name",
]
TEXT_INDEXES = ["title"]             # 제목 안의 단어/구문 검색용
INTEGER_INDEXES = ["publication_year"]
DATETIME_INDEXES: list[str] = []


# ---------------------------------------------------------------------------
# API / 터널
# ---------------------------------------------------------------------------

API_HOST = _env("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)

# 비워두면 실행할 때마다 임의로 생성된다. .env 에 적어두면 고정된다.
API_KEY = os.environ.get("OPENALEX_VDB_API_KEY", "")

# cloudflared named tunnel 이름. 비우면 퀵 터널(주소가 매번 바뀜).
# named tunnel 은 주소가 고정되지만 본인 소유 도메인이 Cloudflare 에 있어야 한다.
TUNNEL_NAME = os.environ.get("TUNNEL_NAME", "")
CLOUDFLARED_BIN = _env("CLOUDFLARED_BIN", "cloudflared")


def _resolve(raw: str | Path) -> Path:
    """상대경로는 (1) 현재 작업 디렉토리 (2) 레포 루트 순으로 찾는다."""
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), REPO_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def resolve_data_root(raw: str | None = None) -> Path:
    return _resolve(raw or DATA_ROOT)


def resolve_parquet_root(raw: str | None = None) -> Path:
    return _resolve(raw or PARQUET_ROOT)


class Workspace:
    """전처리 산출물 경로 모음. DATA_ROOT 하나에서 파생된다."""

    def __init__(self, data_root: Path | None = None):
        self.root = data_root or resolve_data_root()
        self.logs = self.root / LOGS_SUBDIR
        self.cache = self.root / CACHE_SUBDIR
        self.sidecar = self.root / SIDECAR_SUBDIR
        self.filtered_ids = self.root / FILTERED_ID_NAME
