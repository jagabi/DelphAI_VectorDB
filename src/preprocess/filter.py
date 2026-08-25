"""
OpenAlex works 필터 -> 연도별 JSONL 저장 (메모리 상수, 재시작 가능)

필터 조건 = publication_year in YEARS  AND  abstract 존재
country_code 는 필터가 아니라 "조회"로만 쓴다.
  ATTACH_COUNTRY=True  -> sources 에서 source_id -> country_code lookup 을 만들어
                          각 행에 실제 국가 코드를 채운다 (미상은 null)
  ATTACH_COUNTRY=False -> sources 를 아예 읽지 않고 country_code 필드를 뺀다

경로는 전부 src/config.py 에서 온다. 실행은 레포 루트에서:
    python preprocess.py --stage filter
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from .. import config as C

# ---------------------------------------------------------------- 경로 설정
_PARQUET = C.resolve_parquet_root()
SRC_DIR = _PARQUET / C.SOURCES_SUBDIR
WORKS_DIR = _PARQUET / C.WORKS_SUBDIR
ABS_DIR = _PARQUET / C.ABSTRACT_SUBDIR

_WS = C.Workspace()
OUT_ROOT = _WS.root
LOG_DIR = _WS.logs
CACHE_DIR = _WS.cache
FILTERED_ID_PATH = _WS.filtered_ids
CKPT_PATH = LOG_DIR / "checkpoint.json"

YEARS = [int(y) for y in C.YEARS]
ATTACH_COUNTRY = C.ATTACH_COUNTRY
ROWS_PER_FILE = C.ROWS_PER_PART

READ_BATCH = 100_000
ABS_READ_BATCH = 200_000

ABS_ID_COL = "id"
ABS_VALUE_COL = "value"

WORK_COLS = [
    "id",
    "doi",
    "title",
    "publication_date",
    "publication_year",
    "type",
    "primary_location__source__id",
    "primary_location__source__display_name",
    "primary_location__source__issn_l",
    "primary_location__source__host_organization",
    "primary_location__source__host_organization_name",
    "primary_topic__id",
    "primary_topic__display_name",
    "primary_topic__domain__id",
    "primary_topic__domain__display_name",
    "primary_topic__field__id",
    "primary_topic__field__display_name",
    "primary_topic__subfield__id",
    "primary_topic__subfield__display_name",
]

YEAR_SET = pa.array(YEARS, pa.int64())


# ------------------------------------------------- Windows 안전 파일 연산
def _retry_io(fn, retries: int = 6, delay: float = 0.3):
    """PermissionError(파일 사용 중)에 대해 지수적으로 재시도."""
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


def ids_to_int64(arr) -> np.ndarray:
    """'https://openalex.org/W2741809807' 또는 'W2741809807' -> 2741809807

    null / 파싱 실패는 -1. 실제 id는 양수라 -1은 절대 매칭되지 않는다.
    """
    arr = _as_string_array(arr)
    if len(arr) == 0:
        return np.empty(0, dtype=np.int64)
    if arr.null_count == len(arr):
        return np.full(len(arr), -1, dtype=np.int64)

    sample = arr.drop_null()[0].as_py()
    off = sample.rfind("/") + 2  # '/' 다음 접두 문자(W, S ...) 한 글자 skip
    sliced = pc.utf8_slice_codeunits(arr, off)
    try:
        ints = pc.cast(sliced, pa.int64(), safe=False)
    except pa.ArrowInvalid:
        vals = []
        for s in sliced.to_pylist():
            try:
                vals.append(int(s))
            except (TypeError, ValueError):
                vals.append(-1)
        return np.asarray(vals, dtype=np.int64)
    return pc.fill_null(ints, -1).to_numpy(zero_copy_only=False).astype(np.int64)


def isin_sorted(vals: np.ndarray, sorted_arr: np.ndarray) -> np.ndarray:
    """정렬된 배열에 대한 멤버십 검사. set 없이 O(log N)."""
    if sorted_arr.size == 0 or vals.size == 0:
        return np.zeros(vals.size, dtype=bool)
    idx = np.searchsorted(sorted_arr, vals)
    np.clip(idx, 0, sorted_arr.size - 1, out=idx)
    return sorted_arr[idx] == vals


def to_year(col) -> pa.Array:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_dictionary(col.type):
        col = col.cast(pa.string())
    if pa.types.is_integer(col.type):
        return pc.cast(col, pa.int64())
    if pa.types.is_floating(col.type):
        return pc.cast(pc.fill_null(col, 0.0), pa.int64(), safe=False)
    try:
        return pc.cast(col, pa.int64(), safe=False)
    except pa.ArrowInvalid:
        vals = []
        for s in col.to_pylist():
            try:
                vals.append(int(s))
            except (TypeError, ValueError):
                vals.append(0)
        return pa.array(vals, pa.int64())


# ------------------------------------------------------------- 디스크 정렬
def _flush_sorted_npy(raw_path: Path, out_path: Path) -> np.ndarray:
    arr = np.fromfile(raw_path, dtype=np.int64)
    arr.sort()  # in-place introsort
    np.save(out_path, arr)
    safe_unlink(raw_path)
    return arr


# ------------------------------------------- source_id -> country (조회 전용)
def load_source_country() -> tuple[np.ndarray, np.ndarray]:
    """(정렬된 source id int64, 같은 순서의 country_code) 반환.

    sources는 작아서 통째로 올려도 수십 MB 수준이다.
    필터가 아니라 값 채우기 용도.
    """
    cache = CACHE_DIR / "source_country.npz"
    if cache.exists():
        z = np.load(cache)
        ids, cc = z["ids"], z["cc"]
        print(f"[cache] source_country.npz 재사용 ({ids.size:,})")
        return ids, cc

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC_DIR.glob("*.parquet"))
    id_chunks: list[np.ndarray] = []
    cc_chunks: list[np.ndarray] = []
    for f in tqdm(files, desc="sources (country)", unit="file"):
        t = pq.read_table(f, columns=["id", "country_code"])
        id_chunks.append(ids_to_int64(t["id"]))
        raw = _as_string_array(t["country_code"]).to_pylist()
        cc_chunks.append(np.asarray([c or "" for c in raw], dtype="<U8"))
        del t, raw

    ids = np.concatenate(id_chunks) if id_chunks else np.empty(0, np.int64)
    cc = np.concatenate(cc_chunks) if cc_chunks else np.empty(0, "<U8")
    del id_chunks, cc_chunks

    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    cc = cc[order]
    del order
    np.savez(cache, ids=ids, cc=cc)
    return ids, cc


def lookup_country(src_int: np.ndarray, s_ids: np.ndarray, s_cc: np.ndarray):
    if s_ids.size == 0 or src_int.size == 0:
        return [None] * int(src_int.size)
    pos = np.searchsorted(s_ids, src_int)
    np.clip(pos, 0, s_ids.size - 1, out=pos)
    hit = s_ids[pos] == src_int
    out = np.where(hit, s_cc[pos], "")
    return [c if c else None for c in out.tolist()]


# ------------------------------------------------- abstract 보유 id (필터용)
def _abs_value_all_nonnull(pf: pq.ParquetFile) -> bool:
    """parquet 통계만 보고 value 컬럼에 null이 없는지 확인.

    True면 무거운 value 컬럼을 디코딩하지 않고 id만 읽는다.
    """
    md = pf.metadata
    if md is None:
        return False
    try:
        j = list(md.schema.names).index(ABS_VALUE_COL)
    except ValueError:
        return False
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(j).statistics
        if st is None or not st.has_null_count or st.null_count > 0:
            return False
    return True


def load_ids_with_abstract() -> np.ndarray:
    cache = CACHE_DIR / "abstract_ids.npy"
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        print(f"[cache] abstract_ids.npy 재사용 ({arr.size:,}) - abs 재스캔 안 함")
        return arr

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = CACHE_DIR / "abstract_ids.raw"
    safe_unlink(raw)

    files = sorted(ABS_DIR.glob("*.parquet"))
    n_skip = 0
    with raw.open("wb") as fp:
        for f in tqdm(files, desc="abstract ids", unit="file"):
            pf = pq.ParquetFile(f)
            skip_value = _abs_value_all_nonnull(pf)
            n_skip += int(skip_value)
            cols = [ABS_ID_COL] if skip_value else [ABS_ID_COL, ABS_VALUE_COL]
            for rb in pf.iter_batches(batch_size=ABS_READ_BATCH, columns=cols):
                ids = rb.column(ABS_ID_COL)
                if not skip_value:
                    keep = pc.invert(pc.is_null(rb.column(ABS_VALUE_COL)))
                    ids = ids.filter(keep)
                    del keep
                if len(ids):
                    fp.write(ids_to_int64(ids).tobytes())
                del rb, ids
            del pf
    print(f"  value 컬럼 디코딩 생략: {n_skip}/{len(files)} 파일")
    return _flush_sorted_npy(raw, cache)


# ------------------------------------------------------------------ writer
def _clean_line(s: str) -> str:
    """LINE/PARAGRAPH SEPARATOR 제거. jsonl 은 한 줄이 한 레코드라 깨지면 안 된다."""
    return s.replace("\u2028", " ").replace("\u2029", " ")


def truncate_file(path: Path, n_lines: int) -> None:
    """앞 n_lines 줄만 남기고 자른다. 호출 시 path가 열려 있으면 안 된다."""
    if not path.exists():
        if n_lines > 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return
    if n_lines <= 0:
        path.write_text("", encoding="utf-8")
        return
    tmp = path.with_suffix(path.suffix + ".trunc")
    kept = 0
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            kept += 1
            if kept >= n_lines:
                break
    safe_replace(tmp, path)


class JsonlPartWriter:
    """연도별 part 파일 writer.

    __init__은 파일을 열지 않는다. start_fresh() 또는 restore() 후에 write할 것.
    (열어둔 채로 unlink/replace하면 Windows에서 PermissionError)
    """

    def __init__(self, year: int, year_dir: Path, rows_per_file: int = ROWS_PER_FILE):
        self.year = year
        self.year_dir = year_dir
        self.rows_per_file = rows_per_file
        self.part_idx = 0
        self.in_part = 0
        self.total = 0
        self._fp = None

    def _path(self, part_idx: int) -> Path:
        return self.year_dir / f"part_{part_idx:04d}.jsonl"

    def relative_path(self) -> str:
        return f"{self.year}/part_{self.part_idx:04d}.jsonl"

    def start_fresh(self) -> None:
        self.close()
        self.year_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.year_dir.glob("part_*.jsonl")):
            safe_unlink(p)
        self.part_idx = 0
        self.in_part = 0
        self.total = 0
        self._fp = self._path(0).open("w", encoding="utf-8")

    def restore(self, state: dict | None) -> None:
        self.close()  # 어떤 정리 작업보다 먼저 핸들을 놓는다
        state = state or {}
        part_idx = int(state.get("part_idx", 0))
        in_part = int(state.get("in_part", 0))
        self.total = int(state.get("total", 0))

        self.year_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.year_dir.glob("part_*.jsonl")):
            try:
                idx = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if idx > part_idx:
                safe_unlink(p)

        path = self._path(part_idx)
        truncate_file(path, in_part)
        self.part_idx = part_idx
        self.in_part = in_part
        self._fp = path.open("a", encoding="utf-8")

    def state(self) -> dict:
        return {"part_idx": self.part_idx, "in_part": self.in_part, "total": self.total}

    def write(self, obj: dict) -> str:
        if self._fp is None:
            raise RuntimeError(f"{self.year}: start_fresh()/restore()를 먼저 호출할 것")
        if self.in_part >= self.rows_per_file:
            self._fp.close()
            self.part_idx += 1
            self._fp = self._path(self.part_idx).open("w", encoding="utf-8")
            self.in_part = 0
        self._fp.write(_clean_line(json.dumps(obj, ensure_ascii=False)) + "\n")
        self.in_part += 1
        self.total += 1
        return self.relative_path()

    def flush(self) -> None:
        if self._fp:
            self._fp.flush()

    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None


# -------------------------------------------------------------- checkpoint
def load_checkpoint() -> dict:
    if not CKPT_PATH.exists():
        return {
            "status": "pending",
            "last_completed_file": -1,
            "id_lines": 0,
            "year_state": {},
        }
    with CKPT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(ckpt: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    safe_replace(tmp, CKPT_PATH)


# ------------------------------------------------------------------ export
def row_to_obj(r: dict, country: str | None) -> dict:
    obj = {
        "id": r["id"],
        "doi": r["doi"],
        "title": r["title"],
        "publication_date": r["publication_date"],
        "publication_year": r["_year"],
        "type": r["type"],
    }
    if ATTACH_COUNTRY:
        obj["country_code"] = country
    obj.update(
        {
            "primary_location__source__host_organization": r[
                "primary_location__source__host_organization"
            ],
            "primary_location__source__host_organization_name": r[
                "primary_location__source__host_organization_name"
            ],
            "primary_location_source_id": r["primary_location__source__id"],
            "primary_location__source__display_name": r[
                "primary_location__source__display_name"
            ],
            "issn_l": r["primary_location__source__issn_l"],
            "primary_topic__domain__id": r["primary_topic__domain__id"],
            "primary_topic__domain__display_name": r[
                "primary_topic__domain__display_name"
            ],
            "primary_topic__field__id": r["primary_topic__field__id"],
            "primary_topic__field__display_name": r[
                "primary_topic__field__display_name"
            ],
            "primary_topic__subfield__id": r["primary_topic__subfield__id"],
            "primary_topic__subfield__display_name": r[
                "primary_topic__subfield__display_name"
            ],
            "primary_topic__id": r["primary_topic__id"],
            "primary_topic__display_name": r["primary_topic__display_name"],
        }
    )
    return obj


def process_batch(rb, abs_ids, s_ids, s_cc, writers, id_fp) -> int:
    t = pa.Table.from_batches([rb])
    year = to_year(t["publication_year"])

    # 필터 1: 연도
    mask = pc.fill_null(pc.is_in(year, value_set=YEAR_SET), False)
    if not pc.any(mask).as_py():
        return 0
    t = t.append_column("_year", year).filter(mask)
    del mask, year
    if t.num_rows == 0:
        return 0

    # 필터 2: abstract 존재
    wid_int = ids_to_int64(t["id"])
    keep = isin_sorted(wid_int, abs_ids)
    del wid_int
    if not keep.any():
        return 0
    t = t.filter(pa.array(keep))
    del keep
    if t.num_rows == 0:
        return 0

    # 값 채우기(필터 아님): country_code
    if ATTACH_COUNTRY:
        src_int = ids_to_int64(t["primary_location__source__id"])
        countries = lookup_country(src_int, s_ids, s_cc)
        del src_int
    else:
        countries = [None] * t.num_rows

    written = 0
    for i, row in enumerate(t.to_pylist()):
        w = writers.get(row["_year"])
        if w is None:
            continue
        rel = w.write(row_to_obj(row, countries[i]))
        id_fp.write(
            json.dumps({"id": row["id"], "path": rel}, ensure_ascii=False) + "\n"
        )
        written += 1
    del t, countries
    return written


def check_dirs() -> None:
    targets = [("WORKS_DIR", WORKS_DIR), ("ABS_DIR", ABS_DIR)]
    if ATTACH_COUNTRY:
        targets.insert(0, ("SRC_DIR", SRC_DIR))
    for name, d in targets:
        if not d.exists():
            raise SystemExit(f"{name} 경로 없음: {d.resolve()}")
        n = len(list(d.glob("*.parquet")))
        if n == 0:
            sub = len(list(d.rglob("*.parquet")))
            raise SystemExit(
                f"{name}에 parquet 없음: {d.resolve()}\n"
                f"  하위까지 포함하면 {sub}개 -> 경로를 한 단계 조정하세요."
            )
        print(f"{name}: {n} files")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    check_dirs()
    print(f"필터: year in {YEARS} AND abstract 존재  (국가 필터 없음)")

    ckpt = load_checkpoint()
    if ckpt.get("status") == "done":
        print("이미 완료됨. 다시 돌리려면 logs/checkpoint.json 삭제.")
        return

    abs_ids = load_ids_with_abstract()
    if abs_ids.size == 0:
        raise SystemExit("abstract id 없음")

    if ATTACH_COUNTRY:
        s_ids, s_cc = load_source_country()
    else:
        s_ids = np.empty(0, np.int64)
        s_cc = np.empty(0, "<U8")
    print(f"abs_ids={abs_ids.size:,}  sources={s_ids.size:,}")

    files = sorted(WORKS_DIR.glob("*.parquet"))
    last_file = int(ckpt.get("last_completed_file", -1))
    id_lines = int(ckpt.get("id_lines", 0))
    year_state = ckpt.get("year_state") or {}
    resuming = last_file >= 0

    # 정리를 먼저 하고, 그 다음에 파일을 연다.
    writers: dict[int, JsonlPartWriter] = {}
    for y in YEARS:
        w = JsonlPartWriter(y, OUT_ROOT / str(y))
        if resuming:
            w.restore(year_state.get(str(y)))
        else:
            w.start_fresh()
        writers[y] = w

    if resuming:
        truncate_file(FILTERED_ID_PATH, id_lines)
        print(f"재시작: works[{last_file + 1}:] / 기존 {id_lines:,}행 유지")
    else:
        safe_unlink(FILTERED_ID_PATH)
        id_lines = 0

    id_fp = FILTERED_ID_PATH.open("a", encoding="utf-8")
    try:
        bar = tqdm(range(last_file + 1, len(files)), desc="works", unit="file")
        for fi in bar:
            f = files[fi]
            pf = pq.ParquetFile(f)
            for rb in pf.iter_batches(batch_size=READ_BATCH, columns=WORK_COLS):
                id_lines += process_batch(rb, abs_ids, s_ids, s_cc, writers, id_fp)
                del rb
            del pf

            for w in writers.values():
                w.flush()
            id_fp.flush()

            ckpt["status"] = "in_progress"
            ckpt["last_completed_file"] = fi
            ckpt["last_file_name"] = f.name
            ckpt["id_lines"] = id_lines
            ckpt["year_state"] = {str(y): writers[y].state() for y in YEARS}
            save_checkpoint(ckpt)
            bar.set_postfix(rows=f"{id_lines:,}")
    finally:
        id_fp.close()
        for w in writers.values():
            w.close()

    ckpt["status"] = "done"
    ckpt["id_lines"] = id_lines
    ckpt["year_state"] = {str(y): writers[y].state() for y in YEARS}
    save_checkpoint(ckpt)
    print(f"완료. filtered rows = {id_lines:,}")
    for y in YEARS:
        print(f"  {y}: {writers[y].total:,}")


if __name__ == "__main__":
    main()
