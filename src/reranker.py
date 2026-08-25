#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BGE 리랭커 (cross-encoder).

bge-m3 는 bi-encoder 라 문서를 질의와 무관하게 벡터 하나로 압축해둔다. 8,800만 건을
훑으려면 그 방식이어야 하지만, 상위권 순서는 거칠다.

리랭커는 cross-encoder 라 질의와 문서를 같이 넣고 토큰끼리 attention 을 태운다.
전수로는 못 쓰지만 후보 수십 개를 다시 정렬하는 데는 훨씬 정확하다.

    벡터 검색 top 50  ->  리랭커  ->  top 10
"""

from __future__ import annotations

import inspect

import torch

from . import config as C
from . import gpu
from .embedding import is_oom, resolve_device


def _build_cross_encoder(model_name: str, device: str, max_length: int, fp16: bool):
    """sentence-transformers 버전에 따라 모델 인자 이름이 다르다.

    5.x 는 model_kwargs, 그 이전은 automodel_args 를 쓴다. 시그니처를 보고 고른다.
    """
    from sentence_transformers import CrossEncoder

    params = inspect.signature(CrossEncoder.__init__).parameters
    kwargs = {"device": device, "max_length": max_length}

    inner = {}
    if fp16 and device.startswith("cuda"):
        inner["torch_dtype"] = torch.float16
    inner.update(gpu.attention_kwargs())

    slot = "model_kwargs" if "model_kwargs" in params else (
        "automodel_args" if "automodel_args" in params else None)

    if slot is None:
        return CrossEncoder(model_name, **kwargs)

    try:
        return CrossEncoder(model_name, **kwargs, **{slot: inner})
    except (TypeError, ValueError) as exc:
        # attn_implementation 을 모르는 조합이면 그것만 빼고 재시도
        if "attn_implementation" not in str(exc):
            raise
        inner.pop("attn_implementation", None)
        return CrossEncoder(model_name, **kwargs, **{slot: inner})


class Reranker:
    """(질의, 문서) 쌍의 관련도를 매긴다. 점수가 높을수록 관련도가 높다."""

    def __init__(self, model_name: str | None = None, device: str = "auto", *,
                 max_length: int | None = None, batch_size: int | None = None,
                 fp32: bool = False, tf32: bool = True,
                 memory_gib: float | None = None):
        self.model_name = model_name or C.RERANKER_MODEL
        self.device = resolve_device(device)
        gpu.tune(self.device, tf32=tf32, memory_gib=memory_gib)

        self.max_length = max_length or C.RERANKER_MAX_LENGTH
        self.batch_size = batch_size or gpu.default_batch(
            self.device, cuda=C.RERANKER_BATCH_CUDA, cpu=C.RERANKER_BATCH_CPU)
        self.model = _build_cross_encoder(
            self.model_name, self.device, self.max_length, not fp32)
        gpu.report_memory(self.device, "reranker")

    def score(self, query: str, documents: list[str], warn=print) -> list[float]:
        """단일 질의에 대한 점수."""
        return self._predict([(query, doc) for doc in documents], warn)

    def _predict(self, pairs: list[tuple[str, str]], warn=print) -> list[float]:
        """OOM 이 나면 배치를 절반으로 줄여 재시도한다."""
        if not pairs:
            return []
        batch = self.batch_size
        while True:
            try:
                with torch.inference_mode():
                    scores = self.model.predict(pairs, batch_size=batch,
                                                show_progress_bar=False)
                return [float(s) for s in scores]
            except Exception as exc:
                if not is_oom(exc) or batch <= 1:
                    raise
                reduced = max(1, batch // 2)
                warn(f"[warn] reranker OOM -> batch {batch} -> {reduced}")
                batch = reduced
                self.batch_size = reduced
                torch.cuda.empty_cache()

    def rerank(self, query: str, items: list[dict], *, text_of, top_k: int,
               warn=print) -> list[dict]:
        """items 를 재정렬해 상위 top_k 를 돌려준다. 각 item 에 rerank_score 를 붙인다."""
        return self.rerank_many([query], [items], text_of=text_of,
                                top_k=top_k, warn=warn)[0]

    def rerank_many(self, queries: list[str], item_lists: list[list[dict]], *,
                    text_of, top_k: int, warn=print) -> list[list[dict]]:
        """질의 여러 개를 한 번의 forward 로 처리한다.

        (질의, 문서) 쌍을 전부 펼쳐서 한 배치로 넘긴다. GPU 를 한 번만 왕복하므로
        질의마다 따로 부르는 것보다 훨씬 빠르다.
        """
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []          # 질의별 (시작, 끝) 구간
        for query, items in zip(queries, item_lists):
            start = len(pairs)
            pairs.extend((query, text_of(item)) for item in items)
            spans.append((start, len(pairs)))

        scores = self._predict(pairs, warn) if pairs else []

        out: list[list[dict]] = []
        for items, (start, end) in zip(item_lists, spans):
            for item, score in zip(items, scores[start:end]):
                item["rerank_score"] = score
            items.sort(key=lambda item: item["rerank_score"], reverse=True)
            out.append(items[:top_k])
        return out
