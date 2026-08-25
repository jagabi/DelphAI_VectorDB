"""
abstract_inverted_index -> 평문 abstract 로 변환 후
filter.py 가 만든 연도별 part JSONL 에 "abstract" 키를 삽입한다.

3단계
  Phase 0  filtered_id.jsonl -> (id int64 정렬배열, part 코드 배열) 인덱스 구축
           * dict 대신 numpy 배열. 2천만 건 기준 ~300MB
  Phase 1  ABS_DIR 스트리밍 -> 매칭되는 id만 평문화 -> part별 사이드카로 분배
           * abstract 전체를 메모리에 모으지 않는다 (모으면 수십 GB)
  Phase 2  part 파일 하나씩 사이드카와 조인해서 "abstract" 삽입 후 원자적 교체

각 단계 재시작 가능. 경로는 src/config.py 에서 온다. 실행:
    python preprocess.py --stage abstract
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from .. import config as C

# ---------------------------------------------------------------- 경로 설정
ABS_DIR = C.resolve_parquet_root() / C.ABSTRACT_SUBDIR

_WS = C.Workspace()
OUT_ROOT = _WS.root
LOG_DIR = _WS.logs
CACHE_DIR = _WS.cache
SIDE_DIR = _WS.sidecar
FILTERED_ID_PATH = _WS.filtered_ids
CKPT_PATH = LOG_DIR / "fill_abs_checkpoint.json"
IDX_CACHE = CACHE_DIR / "part_index.npz"

ABS_ID_COL = "id"
ABS_VALUE_COL = "value"
ABS_READ_BATCH = 50_000

# Phase 2에서 사이드카를 메모리에 올릴 때의 예산(바이트).
# 사이드카가 이보다 크면 자동으로 여러 패스로 나눠 처리한다.
MEM_BUDGET = 1_500_000_000  # 1.5GB

# abstract가 지나치게 긴 경우 자를 길이. None이면 자르지 않음.
MAX_ABSTRACT_CHARS = None


# ------------------------------------------------- Windows 안전 파일 연산
def _retry_io(fn, retries: int = 6, delay: float = 0.3):
    last = None
    for i in range(retries):
        try:
            return fn()
        except PermissionError as e:
            last = e
            time.sleep(delay * (2**i))
    raise last


def safe_unlink(p: Path) -> None:
    _retry_io(lambda: p.unlink(missing_ok=True))


def safe_replace(src: Path, dst: Path) -> None:
    _retry_io(lambda: src.replace(dst))


# ------------------------------------------------------------ id <-> int64
def _as_string_array(arr) -> pa.Array:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_dictionary(arr.type):
        arr = arr.cast(pa.string())
    if pa.types.is_large_string(arr.type):
        arr = arr.cast(pa.string())
    return arr


def wid_to_int(s: str) -> int:
    """'https://openalex.org/W2741809807' 또는 'W2741809807' -> 2741809807"""
    i = s.rfind("/")
    try:
        return int(s[i + 2:])
    except (TypeError, ValueError):
        return -1


def ids_to_int64(arr) -> np.ndarray:
    arr = _as_string_array(arr)
    if len(arr) == 0:
        return np.empty(0, dtype=np.int64)
    if arr.null_count == len(arr):
        return np.full(len(arr), -1, dtype=np.int64)
    sample = arr.drop_null()[0].as_py()
    off = sample.rfind("/") + 2
    sliced = pc.utf8_slice_codeunits(arr, off)
    try:
        ints = pc.cast(sliced, pa.int64(), safe=False)
    except pa.ArrowInvalid:
        return np.asarray(
            [wid_to_int(s) if s else -1 for s in sliced.to_pylist()], dtype=np.int64
        )
    return pc.fill_null(ints, -1).to_numpy(zero_copy_only=False).astype(np.int64)


# ----------------------------------------------------------- inverted index
def invert_abstract(inv) -> str | None:
    """inverted index -> 평문 abstract

    parquet의 value 컬럼이 아래 중 무엇이든 처리한다.
      - JSON 문자열
      - dict
      - map/struct 변환형 [(key, positions), ...]
    """
    if inv is None:
        return None
    if isinstance(inv, str):
        s = inv.strip()
        if not s:
            return None
        try:
            inv = json.loads(s)
        except (ValueError, TypeError):
            return None
    if isinstance(inv, dict):
        items = inv.items()
    elif isinstance(inv, (list, tuple)):
        # pyarrow MapArray.to_pylist() -> [(key, value), ...]
        try:
            items = [(k, v) for k, v in inv]
        except (TypeError, ValueError):
            return None
    else:
        return None

    pairs: list[tuple[int, str]] = []
    for tok, positions in items:
        if positions is None:
            continue
        for pos in positions:
            if pos is None:
                continue
            pairs.append((pos, tok))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    text = " ".join(tok for _, tok in pairs)
    if MAX_ABSTRACT_CHARS and len(text) > MAX_ABSTRACT_CHARS:
        text = text[:MAX_ABSTRACT_CHARS]
    return text or None


# -------------------------------------------------------------- checkpoint
def load_ckpt() -> dict:
    if not CKPT_PATH.exists():
        return {"scan": {"status": "pending", "last_file": -1}, "apply": {"done": []}}
    with CKPT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_ckpt(ckpt: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    safe_replace(tmp, CKPT_PATH)


# ------------------------------------------------- Phase 0: part 인덱스 구축
def _encode(year: int, part_idx: int) -> int:
    return year * 10_000 + part_idx


def _decode(code: int) -> tuple[int, int]:
    return int(code) // 10_000, int(code) % 10_000


def build_part_index() -> tuple[np.ndarray, np.ndarray]:
    """filtered_id.jsonl -> (정렬된 id int64 배열, 같은 순서의 part 코드 배열)"""
    if IDX_CACHE.exists():
        z = np.load(IDX_CACHE)
        ids, codes = z["ids"], z["codes"]
        print(f"[cache] part_index.npz 재사용 ({ids.size:,})")
        return ids, codes

    if not FILTERED_ID_PATH.exists():
        raise SystemExit(f"filtered_id.jsonl 없음: {FILTERED_ID_PATH.resolve()}")

    raw_i = CACHE_DIR / "_pi_ids.raw"
    raw_c = CACHE_DIR / "_pi_codes.raw"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_unlink(raw_i)
    safe_unlink(raw_c)

    buf_i: list[int] = []
    buf_c: list[int] = []
    n_bad = 0
    with FILTERED_ID_PATH.open("r", encoding="utf-8") as f, \
            raw_i.open("wb") as fi, raw_c.open("wb") as fc:
        for line in tqdm(f, desc="index filtered_id", unit="line"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            path = o.get("path")
            if not path:
                n_bad += 1
                continue
            # "2021/part_0003.jsonl"
            year_s, fname = path.split("/", 1)
            part_idx = int(Path(fname).stem.split("_")[1])
            buf_i.append(wid_to_int(o["id"]))
            buf_c.append(_encode(int(year_s), part_idx))
            if len(buf_i) >= 1_000_000:
                fi.write(np.asarray(buf_i, dtype=np.int64).tobytes())
                fc.write(np.asarray(buf_c, dtype=np.int32).tobytes())
                buf_i.clear()
                buf_c.clear()
        if buf_i:
            fi.write(np.asarray(buf_i, dtype=np.int64).tobytes())
            fc.write(np.asarray(buf_c, dtype=np.int32).tobytes())
    del buf_i, buf_c

    ids = np.fromfile(raw_i, dtype=np.int64)
    codes = np.fromfile(raw_c, dtype=np.int32)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    codes = codes[order]
    del order
    np.savez(IDX_CACHE, ids=ids, codes=codes)
    safe_unlink(raw_i)
    safe_unlink(raw_c)
    if n_bad:
        print(f"  path 없는 항목 {n_bad:,}건 건너뜀")
    return ids, codes


# --------------------------------------------- Phase 1: abs 스캔 -> 사이드카
class SidecarWriters:
    """part 코드별 append 파일 핸들 관리. part 수는 보통 수십 개 수준."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._fps: dict[int, object] = {}

    def path(self, code: int) -> Path:
        year, part = _decode(code)
        return self.root / f"{year}_{part:04d}.jsonl"

    def write(self, code: int, wid: int, abstract: str) -> None:
        fp = self._fps.get(code)
        if fp is None:
            fp = self.path(code).open("a", encoding="utf-8")
            self._fps[code] = fp
        fp.write(json.dumps({"i": wid, "a": abstract}, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        for fp in self._fps.values():
            fp.flush()

    def close(self) -> None:
        for fp in self._fps.values():
            fp.close()
        self._fps.clear()


def _rowgroups_to_read(pf: pq.ParquetFile, ids: np.ndarray) -> list[int]:
    """id 통계로 우리 필터셋과 겹치지 않는 row group을 건너뛴다.

    abs 파일이 id 순으로 정렬돼 있으면 큰 폭으로 시간이 준다.
    통계가 없으면 전부 읽는다.
    """
    md = pf.metadata
    if md is None:
        return list(range(pf.num_row_groups))
    try:
        j = list(md.schema.names).index(ABS_ID_COL)
    except ValueError:
        return list(range(pf.num_row_groups))

    keep = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(j).statistics
        if st is None or not st.has_min_max:
            keep.append(i)
            continue
        try:
            lo = wid_to_int(str(st.min))
            hi = wid_to_int(str(st.max))
        except Exception:
            keep.append(i)
            continue
        if lo < 0 or hi < 0 or lo > hi:
            keep.append(i)
            continue
        if np.searchsorted(ids, lo, "left") < np.searchsorted(ids, hi, "right"):
            keep.append(i)
    return keep


def scan_abstracts(ids: np.ndarray, codes: np.ndarray, ckpt: dict) -> None:
    st = ckpt.setdefault("scan", {"status": "pending", "last_file": -1})
    if st.get("status") == "done":
        print("[skip] Phase 1 이미 완료")
        return

    files = sorted(ABS_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"ABS_DIR에 parquet 없음: {ABS_DIR.resolve()}")

    last = int(st.get("last_file", -1))
    if last < 0:
        # 처음부터 -> 사이드카 초기화
        if SIDE_DIR.exists():
            for p in SIDE_DIR.glob("*.jsonl"):
                safe_unlink(p)
    SIDE_DIR.mkdir(parents=True, exist_ok=True)

    writers = SidecarWriters(SIDE_DIR)
    n_hit = int(st.get("n_hit", 0))
    n_skip_rg = int(st.get("n_skip_rg", 0))
    n_total_rg = int(st.get("n_total_rg", 0))

    try:
        bar = tqdm(range(last + 1, len(files)), desc="abs scan", unit="file")
        for fi in bar:
            f = files[fi]
            pf = pq.ParquetFile(f)
            rgs = _rowgroups_to_read(pf, ids)
            n_total_rg += pf.num_row_groups
            n_skip_rg += pf.num_row_groups - len(rgs)

            if rgs:
                for rb in pf.iter_batches(
                    batch_size=ABS_READ_BATCH,
                    columns=[ABS_ID_COL, ABS_VALUE_COL],
                    row_groups=rgs,
                ):
                    arr = ids_to_int64(rb.column(ABS_ID_COL))
                    pos = np.searchsorted(ids, arr)
                    np.clip(pos, 0, ids.size - 1, out=pos)
                    hit = ids[pos] == arr
                    sel = np.flatnonzero(hit)
                    if sel.size == 0:
                        del rb, arr, pos, hit, sel
                        continue

                    sel_codes = codes[pos[sel]]
                    sel_ids = arr[sel]
                    vals = rb.column(ABS_VALUE_COL).take(pa.array(sel)).to_pylist()
                    for k, v in enumerate(vals):
                        text = invert_abstract(v)
                        if text is None:
                            continue
                        writers.write(int(sel_codes[k]), int(sel_ids[k]), text)
                        n_hit += 1
                    del rb, arr, pos, hit, sel, sel_codes, sel_ids, vals
            del pf

            writers.flush()
            st["status"] = "in_progress"
            st["last_file"] = fi
            st["last_file_name"] = f.name
            st["n_hit"] = n_hit
            st["n_skip_rg"] = n_skip_rg
            st["n_total_rg"] = n_total_rg
            ckpt["scan"] = st
            save_ckpt(ckpt)
            bar.set_postfix(hit=f"{n_hit:,}")
    finally:
        writers.close()

    st["status"] = "done"
    ckpt["scan"] = st
    save_ckpt(ckpt)
    print(f"Phase 1 완료: abstract {n_hit:,}건 확보")
    if n_total_rg:
        print(f"  row group 스킵: {n_skip_rg:,}/{n_total_rg:,}")


# ----------------------------------------- Phase 2: 사이드카를 part에 삽입
def apply_one_part(side: Path) -> tuple[int, int]:
    """사이드카 하나를 대응하는 part JSONL에 병합. (전체행, 채운행) 반환"""
    year_s, part_s = side.stem.split("_")
    part = OUT_ROOT / year_s / f"part_{int(part_s):04d}.jsonl"
    if not part.exists():
        print(f"  ! part 없음, 건너뜀: {part}")
        return 0, 0

    size = side.stat().st_size
    n_passes = max(1, math.ceil(size / MEM_BUDGET))

    total = filled = 0
    for k in range(n_passes):
        mapping: dict[int, str] = {}
        with side.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if n_passes == 1 or (o["i"] % n_passes) == k:
                    mapping[o["i"]] = o["a"]

        tmp = part.with_suffix(".jsonl.tmp")
        total = filled = 0
        with part.open("r", encoding="utf-8") as src, \
                tmp.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                o = json.loads(line)
                total += 1
                a = mapping.pop(wid_to_int(o["id"]), None)
                if a is not None:
                    o["abstract"] = a
                elif "abstract" not in o:
                    o["abstract"] = None
                if o.get("abstract") is not None:
                    filled += 1
                dst.write(json.dumps(o, ensure_ascii=False) + "\n")
        safe_replace(tmp, part)
        del mapping
    return total, filled


def apply_sidecars(ckpt: dict) -> None:
    ap = ckpt.setdefault("apply", {"done": []})
    done = set(ap.get("done", []))

    sides = sorted(SIDE_DIR.glob("*.jsonl")) if SIDE_DIR.exists() else []
    if not sides:
        raise SystemExit("사이드카가 없습니다. Phase 1이 끝났는지 확인하세요.")

    grand_total = grand_filled = 0
    for side in tqdm(sides, desc="apply", unit="part"):
        if side.name in done:
            continue
        rows, filled = apply_one_part(side)
        grand_total += rows
        grand_filled += filled
        done.add(side.name)
        ap["done"] = sorted(done)
        ckpt["apply"] = ap
        save_ckpt(ckpt)

    print(f"Phase 2 완료: {grand_filled:,} / {grand_total:,} 행에 abstract 삽입")
    if grand_total and grand_filled < grand_total:
        print(f"  비어 있는 행 {grand_total - grand_filled:,}건 (abstract=null)")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not ABS_DIR.exists():
        raise SystemExit(f"ABS_DIR 경로 없음: {ABS_DIR.resolve()}")

    ckpt = load_ckpt()
    ids, codes = build_part_index()
    if ids.size == 0:
        raise SystemExit("filtered_id 인덱스가 비어 있습니다.")
    print(f"대상 id {ids.size:,}건")

    scan_abstracts(ids, codes, ckpt)
    del ids, codes

    apply_sidecars(ckpt)
    print(f"\n검증이 끝나면 사이드카를 지워도 됩니다: {SIDE_DIR}")


if __name__ == "__main__":
    main()
