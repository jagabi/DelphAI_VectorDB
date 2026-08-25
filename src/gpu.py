#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU 설정. 임베딩과 리랭커가 공통으로 쓴다.

두 가지를 한다.

1) 성능
   A6000 은 Ampere(sm_86) 라 TF32 를 지원한다. fp32 로 남는 연산을 TF32 로 돌리면
   추론 정확도 손실이 무시할 수준이면서 눈에 띄게 빨라진다. attention 은 SDPA 커널.

2) 메모리 상한
   같은 GPU 에 다른 모델(gemma 등)을 같이 올릴 때 이 프로세스의 VRAM 사용을 묶는다.
   set_per_process_memory_fraction 은 캐싱 할당자의 상한이라, 넘으면 OOM 이 난다.
   임베딩/리랭커 모두 OOM 시 배치를 절반으로 줄여 재시도하므로 죽지 않고
   상한 안에서 알아서 맞춰 돌아간다.
"""

from __future__ import annotations

import torch

from . import config as C

_TUNED: set[str] = set()
_CAPPED: dict[str, float] = {}


def _index(device: str) -> int:
    return int(device.split(":")[1]) if ":" in device else 0


def diagnose_cpu_fallback() -> None:
    """GPU 가 있어야 하는데 CPU 로 떨어졌을 때 왜 그런지 알려준다.

    가장 흔한 원인은 CPU 전용 torch 휠이 설치된 경우다. PyPI 기본 휠이
    그럴 수 있어서, cu124 인덱스에서 다시 받아야 한다.
    """
    import os

    build = torch.__version__
    cuda_build = torch.version.cuda
    print(f"[warn] CUDA 를 못 씁니다. CPU 로 돕니다 (bge-m3 는 CPU 에서 매우 느립니다)")
    print(f"       torch={build}  torch.version.cuda={cuda_build}")

    if cuda_build is None:
        print("       -> CPU 전용 빌드입니다. CUDA 빌드로 다시 설치하세요:")
        print("          pip uninstall -y torch")
        print("          pip install torch --index-url "
              "https://download.pytorch.org/whl/cu124")
        return

    hidden = os.environ.get("CUDA_VISIBLE_DEVICES")
    if hidden is not None and hidden.strip() in ("", "-1"):
        print(f"       -> CUDA_VISIBLE_DEVICES={hidden!r} 때문에 GPU 가 가려져 있습니다")
        return

    print("       -> CUDA 빌드는 맞는데 드라이버가 안 잡힙니다. nvidia-smi 를 확인하세요")


def total_gib(device: str) -> float:
    return torch.cuda.get_device_properties(_index(device)).total_memory / (1 << 30)


def device_info(device: str) -> str:
    if not device.startswith("cuda"):
        return "CPU"
    props = torch.cuda.get_device_properties(_index(device))
    return (f"{props.name} | sm_{props.major}{props.minor} | "
            f"{props.total_memory / (1 << 30):.0f} GiB | "
            f"torch {torch.__version__} | cuda {torch.version.cuda}")


def cap_memory(device: str, gib: float | None = None) -> float | None:
    """이 프로세스가 쓸 수 있는 VRAM 을 GiB 단위로 제한한다. 실제 상한을 반환."""
    if not device.startswith("cuda"):
        return None
    gib = C.GPU_MEMORY_GIB if gib is None else gib
    if not gib or gib <= 0:
        return None

    total = total_gib(device)
    gib = min(gib, total)
    fraction = max(0.01, min(1.0, gib / total))
    torch.cuda.set_per_process_memory_fraction(fraction, _index(device))
    _CAPPED[device] = gib

    if gib < 2 * WEIGHTS_GIB + 0.5:
        print(f"[warn] 상한 {gib:.1f} GiB 는 임베딩+리랭커 가중치({2 * WEIGHTS_GIB:.1f} GiB)에"
              f" 여유가 거의 없습니다.")
        print("       --no-reranker 로 하나만 올리거나 상한을 3.5 GiB 이상으로 잡으세요.")
    return gib


def tune(device: str, *, tf32: bool = True, benchmark: bool = True,
         memory_gib: float | None = None) -> None:
    """장치별로 한 번만 적용한다."""
    if not device.startswith("cuda") or device in _TUNED:
        return
    _TUNED.add(device)

    # Ampere 이상에서 matmul 을 TF32 로. 트랜스포머 추론에서 손실이 무시할 수준이다.
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    # 입력 길이가 배치마다 달라지긴 하지만, 자주 나오는 형태는 캐시된다.
    torch.backends.cudnn.benchmark = benchmark

    print(f"[gpu] {device_info(device)}")
    print(f"[gpu] tf32={tf32} cudnn.benchmark={benchmark}")

    capped = cap_memory(device, memory_gib)
    if capped:
        total = total_gib(device)
        print(f"[gpu] 메모리 상한 {capped:.1f} / {total:.0f} GiB "
              f"(나머지 {total - capped:.1f} GiB 는 다른 프로세스 몫)")
    else:
        print("[gpu] 메모리 상한 없음 (GPU_MEMORY_GIB 로 제한 가능)")


def release(device: str | None = None) -> None:
    """캐싱 할당자가 붙들고 있는 블록을 드라이버에 돌려준다.

    torch 는 한 번 확보한 VRAM 을 재사용하려고 계속 들고 있는다. 옆에서 다른
    모델이 돌면 그게 그대로 압박이 되므로, 요청 처리 후 놓아준다.
    """
    if device is not None and not device.startswith("cuda"):
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# bge-m3 와 bge-reranker-v2-m3 는 둘 다 XLM-R large(약 568M). fp16 이면 각 1.2 GB.
# 둘을 같이 올리면 가중치만 2.4 GB 이므로 상한이 이보다 작으면 아예 못 뜬다.
WEIGHTS_GIB = 1.2


def default_batch(device: str, *, cuda: int, cpu: int,
                  cap_gib: float | None = None) -> int:
    """장치와 메모리 상한에 맞는 기본 배치.

    상한이 좁으면 큰 배치로 시작해봐야 OOM -> 절반 -> OOM 을 반복할 뿐이라,
    처음부터 상한에 맞는 값에서 출발한다.
    """
    if not device.startswith("cuda"):
        return cpu
    cap = cap_gib if cap_gib is not None else _CAPPED.get(device)
    if not cap or cap >= 16:
        return cuda
    if cap >= 8:
        return max(4, cuda // 2)
    return max(2, cuda // 4)


def attention_kwargs() -> dict:
    """SDPA 커널을 쓰도록 요청. 지원하지 않는 버전이면 호출부에서 무시된다."""
    return {"attn_implementation": "sdpa"}


def free_gib(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    free, _total = torch.cuda.mem_get_info(_index(device))
    return free / (1 << 30)


def report_memory(device: str, label: str = "") -> None:
    if not device.startswith("cuda"):
        return
    index = _index(device)
    allocated = torch.cuda.memory_allocated(index) / (1 << 30)
    reserved = torch.cuda.memory_reserved(index) / (1 << 30)
    free = free_gib(device) or 0.0
    cap = _CAPPED.get(device)
    limit = f" / 상한 {cap:.1f}" if cap else ""
    tag = f" {label}" if label else ""
    print(f"[gpu]{tag} allocated {allocated:.1f} / reserved {reserved:.1f}{limit} "
          f"| GPU 여유 {free:.1f} GiB")
