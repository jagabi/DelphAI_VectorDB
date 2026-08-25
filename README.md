# OpenAlex VectorDB

OpenAlex 논문 초록 8,800만 건을 [BGE-M3](https://huggingface.co/BAAI/bge-m3)(1024차원)로
임베딩해 [Qdrant](https://qdrant.tech)에 적재하고, 의미 검색 API 로 제공한다.

```
parquet 덤프 ──filter──▶ 연도별 jsonl ──fill_abs──▶ 초록 삽입
                                                      │
                                                    embed
                                                      ▼
                                                  .f16 (2 KiB/row)
                                                      │
                                                     post
                                                      ▼
                                                   Qdrant ──▶ /search API ──▶ cloudflared
```

## 빠른 시작

```bash
pip install -r requirements.txt
cp .env.example .env      # 경로와 키를 채운다
```

```bash
python preprocess.py      # parquet -> jsonl -> 초록 -> .f16
python post.py            # 컬렉션 생성 + 적재
python api.py             # 검색 API + 외부 터널
```

각 단계는 **중단해도 같은 명령을 다시 실행하면 이어서** 진행된다.

## 진입점

| 명령 | 하는 일 | 필요 |
|---|---|---|
| `preprocess.py` | parquet 필터 → 초록 채우기 → 임베딩 | GPU (임베딩) |
| `post.py` | 컬렉션 생성/관리 + Qdrant 적재 | Qdrant |
| `api.py` | FastAPI 검색 서버 + cloudflared 터널 | Qdrant + GPU(선택) |
| `scripts/client.py` | 외부 PC 용 클라이언트 | **없음** (표준 라이브러리) |

```bash
python preprocess.py --stage embed --embed-batch 128
python post.py --status
python post.py --indexes drop --keep type,publication_year
python api.py --no-tunnel
```

## 디렉토리

```
├── preprocess.py            전처리 진입점
├── post.py                  적재 진입점
├── api.py                   API + 터널 진입점
├── src/
│   ├── config.py            모든 설정 (.env 로 덮어쓰기)
│   ├── paths.py             Part 경로, 진행 위치
│   ├── records.py           jsonl 파싱, payload 변환, point id
│   ├── vectors.py           .f16 입출력
│   ├── embedding.py         BGE-M3 로더 (CUDA OOM 자동 backoff)
│   ├── reranker.py          bge-reranker-v2-m3 cross-encoder
│   ├── dedupe.py            같은 논문의 중복 레코드 병합
│   ├── store.py             LangChain QdrantVectorStore (flat payload)
│   ├── collection.py        컬렉션 생성 / 인덱스 / 대량 적재 모드
│   ├── embed_runner.py      파트 단위 임베딩 루프
│   ├── upload.py            파트 단위 적재 루프
│   └── preprocess/
│       ├── filter.py        parquet -> 연도별 jsonl
│       └── fill_abs.py      inverted index -> 평문 초록 삽입
├── api/
│   ├── server.py            FastAPI 앱 (읽기 전용)
│   ├── models.py            요청/응답 스키마
│   └── tunnel.py            cloudflared 관리
└── scripts/
    ├── client.py            외부 클라이언트 (의존성 0)
    └── freeze.py            설치된 버전으로 requirements 생성
```

데이터(`openalex_2021-2025_postprocessed/`)와 Qdrant 스토리지(`qdrant/`)는
`.gitignore` 대상이다.

## 설계에서 중요한 것

### `.f16` 의 N번째 row == `.jsonl` 의 N번째 줄

빈 줄, 깨진 JSON, 초록 없는 문서도 자리를 비우지 않고 **영벡터**를 채운다. 덕분에

- 재개 위치 = 파일 크기 ÷ 2048 — 별도 체크포인트 파일이 없다
- point id 와 payload 는 jsonl 에서 다시 만들면 되니 id 파일도 불필요
- 중단으로 잘린 마지막 row 만 잘라내면 언제나 정합성이 맞는다

적재할 때 "초록이 있는데 벡터가 영벡터"인 줄을 만나면 정렬이 깨진 것이므로
파트명과 줄번호를 찍고 즉시 멈춘다.

### point id = `uuid5(NAMESPACE_URL, openalex_id)`

결정적이라 몇 번을 다시 올려도 같은 점을 덮어쓴다. 중복 검사가 필요 없고,
중단 지점이 조금 어긋나도 안전하다.

### payload 는 flat, 인덱스는 최소한

`QdrantVectorStore` 기본값은 payload 를 `{"page_content":…, "metadata":{…}}` 로
중첩 저장하는데, 그러면 최상위 필드에 건 인덱스가 먹지 않는다. `src/store.py` 에서
쓰기/읽기 두 메서드만 flat 으로 바꿨다.

payload 인덱스는 **필터 검색 전용**이고 저장과는 무관하다. 인덱스를 빼도 그 필드는
그대로 저장되고 결과에도 딸려 나온다. 대신 인덱스 하나하나가 회수 불가능한 램을
점유하고 그 비용이 `인덱스 개수 × 세그먼트 개수` 로 곱해지므로, 실제로 필터에 쓸
9개만 남겼다 (`src/config.py`).

### 메모리

8,800만 × 1024차원 기준:

| | 크기 | 위치 |
|---|---|---|
| int8 양자화 벡터 | ~84 GiB | 램 (`always_ram`) |
| HNSW 그래프 | ~35 GiB | 램 |
| payload 인덱스 | ~15 GiB | 램 |
| float32 원본 | ~343 GiB | 디스크 (mmap, rescoring 용) |
| payload 본문 | ~135 GiB | 디스크 (`on_disk_payload`) |

검색의 뜨거운 경로(HNSW 탐색 + 거리 계산)는 전부 램에서 끝나고, 디스크는
상위 후보 재정렬과 최종 결과 payload 조회에만 쓰인다.

### 운영 메모

- Qdrant 는 리눅스 + ext4/xfs 에서 돌리는 게 맞다. Docker Desktop for Windows 의
  바인드 마운트에서는 세그먼트 병합의 디렉토리 rename 이 거부되어 옵티마이저가
  멈춘다(`Permission denied (os error 13)`).
- 컨테이너에는 메모리 제한을 걸어라. 제한이 없으면 Qdrant 가 가용 램을 캐시로
  다 쓰다가 VM 전체의 OOM killer 를 부른다.
- `.f16` 이 이 파이프라인의 유일한 비싼 산출물이다(GPU 며칠). Qdrant 스토리지는
  거기서 언제든 다시 만들 수 있다. 백업은 `.f16` 과 `jsonl` 만 하면 된다.

## 검색

두 가지 방식이 있고, 둘 다 같은 필터를 쓴다.

### 의미(벡터) 검색 — `POST /search`

```
질의 --embed--> 벡터 top 50 --중복 제거--> 리랭커 --> top 10
```

bge-m3 는 bi-encoder 라 문서를 질의와 무관하게 벡터 하나로 압축한다. 8,800만 건을
훑으려면 그래야 하지만 상위권 순서는 거칠다. 그래서 후보를 넉넉히 뽑고
**bge-reranker-v2-m3**(cross-encoder)로 다시 정렬한다. 질의와 문서를 같이 넣고
토큰끼리 attention 을 태우므로 훨씬 정확하다.

### 제목 키워드 검색 — `POST /keyword`

`title` 에 걸린 전문 인덱스로 단어(또는 `phrase=true` 로 구문)를 포함하는 문서를
찾는다. 벡터를 쓰지 않으므로 빠르지만 의미가 아니라 표기가 기준이다. 기본 5개.

### 중복 제거

OpenAlex 에는 같은 논문이 여러 레코드로 들어있다. 출판사 원본과 기관 저장소 사본이
따로 등재되는 식이라 **DOI 도 openalex_id 도 서로 다르다**. 그래서 id 로는 못 걸러낸다.

`src/dedupe.py` 는 제목을 정규화(대소문자·구두점·공백·HTML 엔티티·유니코드 표기)한
값을 키로 쓰고, 제목이 없으면 초록 앞부분으로 대신한다. 점수가 높은 쪽을 남기고
합쳐진 개수를 `duplicates` 로 알려준다. 리랭킹 전에 수행하므로 중복에 리랭커 연산을
낭비하지 않는다.

## API

```bash
python api.py                       # 임베딩 + 리랭커를 GPU 에 올리고 터널까지
python api.py --gpu-memory 12       # VRAM 을 12 GiB 로 제한
python api.py --no-reranker         # 리랭커 없이 (VRAM 절약, 순위 품질 하락)
python api.py --device cuda:1       # 다른 GPU
python api.py --no-tunnel           # 로컬만
```

### GPU 를 다른 모델과 나눠 쓸 때

`--gpu-memory` 또는 `.env` 의 `GPU_MEMORY_GIB` 로 이 프로세스의 VRAM 상한을 건다.
`torch.cuda.set_per_process_memory_fraction` 으로 캐싱 할당자를 묶는 방식이라,
상한을 넘으면 OOM 이 나지만 **임베딩·리랭커 모두 OOM 시 배치를 절반으로 줄여
재시도**하므로 죽지 않고 상한 안에서 알아서 맞춰 돌아간다.

```bash
python api.py --gpu-memory 4    # 검색은 4 GiB, 나머지 44 GiB 는 gemma 몫
```

상한에 맞춰 배치 기본값도 자동으로 낮아진다:

| 상한 | 임베딩 배치 | 리랭커 배치 |
|---|---|---|
| 없음 / 16 GiB 이상 | 64 | 16 |
| 8~16 GiB | 32 | 8 |
| 8 GiB 미만 | 16 | 4 |

가중치는 fp16 기준 bge-m3 1.2 GiB + 리랭커 1.2 GiB = **2.4 GiB**. 4 GiB 면
activation 에 1.6 GiB 가 남는 셈이라 동작하지만 여유가 얇다. 6 GiB 면 배치는
같으면서 안정적이고, 8 GiB 부터 배치가 2배로 올라간다.

CUDA 컨텍스트(0.3~0.6 GiB)는 이 상한 바깥이라 `nvidia-smi` 에는 상한보다
그만큼 더 잡힌다.

추가로 `RELEASE_CACHE=1`(기본) 이면 요청을 마칠 때마다 `torch.cuda.empty_cache()`
로 붙들고 있던 블록을 돌려준다. torch 는 한 번 확보한 VRAM 을 재사용하려고 계속
쥐고 있어서, 옆에서 다른 모델이 돌면 그게 그대로 압박이 되기 때문이다.

배치 기본값도 GPU 공유를 전제로 작게 잡혀 있다 (임베딩 64, 리랭커 16).
GPU 를 독점할 수 있으면 `--embed-batch 256` 처럼 올리면 된다.

`X-API-Key` 가 콘솔에 찍힌다. `.env` 의 `OPENALEX_VDB_API_KEY` 에 적어두면 고정된다.

### 터널 주소

퀵 터널은 계정 없이 즉석 발급되는 임시 주소라 **실행할 때마다 바뀐다.**
Cloudflare 설계상 고정할 수 없다.

주소를 고정하려면 named tunnel 을 쓴다. 본인 소유 도메인이 Cloudflare 에
있어야 한다.

```bash
cloudflared tunnel login                                    # 한 번만
cloudflared tunnel create openalex
cloudflared tunnel route dns openalex search.example.com
python api.py --tunnel-name openalex
```

```bash
curl -X POST https://xxxx.trycloudflare.com/search   -H "Content-Type: application/json" -H "X-API-Key: $OPENALEX_VDB_API_KEY"   -d '{"query":"swarm robotics aggregation","limit":10}'
```

브라우저로 `/docs` 를 열면 바로 시험해볼 수 있다.

### 클라이언트

`scripts/client.py` 하나만 복사해 가면 된다. 표준 라이브러리만 쓴다.

```bash
python client.py "swarm robotics aggregation"       # 의미 검색, 10개
python client.py --keyword-search "swarm robotics"  # 제목 키워드, 5개
python client.py "graph neural network" --year-from 2024 --meta
python client.py "diffusion model" --field "Computer Science" --country KR --full
python client.py --json "quantum error correction" > out.json
```

필터는 `--year-from/to`, `--field`, `--subfield`, `--domain`, `--topic`,
`--source`, `--type`, `--country` 로 두 방식 모두에서 쓸 수 있다.

## requirements 갱신

설치된 버전으로 고정하려면 서버에서:

```bash
python scripts/freeze.py > requirements.txt
```

