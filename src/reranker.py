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
from .embedding import is_oom, resolve_device


def _build_cross_encoder(model_name: str, device: str, max_length: int, fp16: bool):
    """sentence-transformers 버전에 따라 fp16 전달 인자 이름이 다르다."""
    from sentence_transformers import CrossEncoder

    kwargs = {"device": device, "max_length": max_length}
    if fp16 and device.startswith("cuda"):
        params = inspect.signature(CrossEncoder.__init__).parameters
        dtype = {"torch_dtype": torch.float16}
        if "model_kwargs" in params:
            kwargs["model_kwargs"] = dtype
        elif "automodel_args" in params:
            kwargs["automodel_args"] = dtype
    return CrossEncoder(model_name, **kwargs)


class Reranker:
    """(질의, 문서) 쌍의 관련도를 매긴다. 점수가 높을수록 관련도가 높다."""

    def __init__(self, model_name: str | None = None, device: str = "auto", *,
                 max_length: int | None = None, batch_size: int | None = None,
                 fp32: bool = False):
        self.model_name = model_name or C.RERANKER_MODEL
        self.device = resolve_device(device)
        self.max_length = max_length or C.RERANKER_MAX_LENGTH
        self.batch_size = batch_size or C.RERANKER_BATCH
        self.model = _build_cross_encoder(
            self.model_name, self.device, self.max_length, not fp32)

    def score(self, query: str, documents: list[str], warn=print) -> list[float]:
        """OOM 이 나면 배치를 절반으로 줄여 재시도한다."""
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        batch = self.batch_size
        while True:
            try:
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
        if not items:
            return []
        scores = self.score(query, [text_of(item) for item in items], warn)
        for item, score in zip(items, scores):
            item["rerank_score"] = score
        items.sort(key=lambda item: item["rerank_score"], reverse=True)
        return items[:top_k]
