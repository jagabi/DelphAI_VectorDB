#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BGE-M3 임베딩. GPU 있으면 GPU(fp16), 없으면 CPU."""

from __future__ import annotations

from typing import Any

import torch

from . import config as C

# torch 1.13+ 는 전용 예외가 있지만, 없으면 메시지로 판별한다
_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def is_oom(exc: Exception) -> bool:
    return isinstance(exc, _OOM) and "out of memory" in str(exc).lower()


def resolve_device(requested: str = "auto") -> str:
    if requested in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build(
    device: str = "auto",
    *,
    batch_size: int = 64,
    max_seq_len: int | None = None,
    fp32: bool = False,
    show_progress: bool = False,
    model_name: str | None = None,
):
    """LangChain HuggingFaceEmbeddings 로 bge-m3 를 올린다."""
    from langchain_huggingface import HuggingFaceEmbeddings

    device = resolve_device(device)
    model_kwargs: dict[str, Any] = {"device": device}
    if device.startswith("cuda") and not fp32:
        # SentenceTransformer(..., model_kwargs={"torch_dtype": ...}) 로 전달된다
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name or C.MODEL_NAME,
        model_kwargs=model_kwargs,
        # bge-m3 는 query instruction 없이 학습된 모델이라 별도 프롬프트가 필요 없다.
        # 컬렉션이 cosine 이므로 정규화해서 저장한다.
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        show_progress=show_progress,
    )

    # 초록은 길어야 수백 토큰. 8192 컨텍스트를 다 열면 느려지기만 한다.
    st_model = getattr(embeddings, "_client", None) or getattr(embeddings, "client", None)
    if st_model is not None:
        st_model.max_seq_length = max_seq_len or C.MAX_SEQ_LEN

    return embeddings, device


def encode(embeddings, texts: list[str], warn=print) -> list[list[float]]:
    """OOM 이 나면 forward 배치를 절반으로 줄여 재시도한다.

    peak 메모리는 배치 안에서 가장 긴 초록이 좌우해서, 며칠 돌다가 어느 배치에서
    갑자기 튈 수 있다. 그때 죽지 말고 알아서 줄이고 계속 가라는 뜻.
    """
    while True:
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            if not is_oom(exc):
                raise
            current = embeddings.encode_kwargs.get("batch_size", 32)
            if current <= 1:
                raise
            reduced = max(1, current // 2)
            embeddings.encode_kwargs["batch_size"] = reduced
            torch.cuda.empty_cache()
            warn(f"[warn] CUDA OOM -> encode batch {current} -> {reduced} 로 낮춰 재시도")
