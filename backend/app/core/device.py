"""Device resolution: DEVICE=auto -> CUDA if usable, else CPU. CPU is a first-class target."""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str  # "cpu" | "cuda"
    reason: str
    torch_available: bool
    torch_version: str | None
    cuda_device_name: str | None
    cpu: str
    threads: int | None


def _torch():
    try:
        import torch  # noqa: WPS433

        return torch
    except Exception:  # pragma: no cover - torch missing is a supported state
        return None


@lru_cache
def resolve_device(requested: str = "auto", torch_threads: int = 0) -> DeviceInfo:
    torch = _torch()
    cpu_name = platform.processor() or platform.machine()
    if torch is None:
        return DeviceInfo(requested, "cpu", "PyTorch not installed", False, None, None, cpu_name, None)

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    threads = torch.get_num_threads()

    cuda_ok = bool(torch.cuda.is_available())
    cuda_name = torch.cuda.get_device_name(0) if cuda_ok else None

    if requested == "cpu":
        resolved, reason = "cpu", "DEVICE=cpu requested"
    elif requested == "cuda":
        if cuda_ok:
            resolved, reason = "cuda", "DEVICE=cuda requested and available"
        else:
            resolved, reason = "cpu", "DEVICE=cuda requested but CUDA unavailable; falling back to CPU"
            log.warning(reason)
    else:
        resolved = "cuda" if cuda_ok else "cpu"
        reason = "auto: CUDA available" if cuda_ok else "auto: no usable CUDA device"

    return DeviceInfo(
        requested=requested,
        resolved=resolved,
        reason=reason,
        torch_available=True,
        torch_version=torch.__version__,
        cuda_device_name=cuda_name,
        cpu=cpu_name or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        threads=threads,
    )
