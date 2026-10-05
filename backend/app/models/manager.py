"""Lazy, LRU-bounded model loading. Models are loaded on first use and evicted when more than
MAX_LOADED_MODELS are held, keeping CPU RAM usage predictable."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.core.config import Settings, get_settings
from app.core.device import resolve_device
from app.core.errors import ModelUnavailable, OutOfMemory
from app.models.registry import ModelRegistry, ModelSpec, get_registry

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    spec: ModelSpec
    device: str
    model: Any
    processor: Any = None
    extras: dict[str, Any] = field(default_factory=dict)
    load_time_s: float = 0.0


class ModelManager:
    def __init__(self, registry: ModelRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()
        self._failures: dict[str, str] = {}

    # ------------------------------------------------------------------ status
    def device_for(self, spec: ModelSpec) -> str:
        requested = spec.device if spec.device != "auto" else self.settings.device
        return resolve_device(requested, self.settings.torch_threads).resolved

    def is_available(self, spec: ModelSpec) -> bool:
        return spec.enabled and spec.weights_present(self.settings.checkpoint_dir)

    def loaded_ids(self) -> list[str]:
        with self._lock:
            return list(self._cache)

    def last_failure(self, model_id: str) -> str | None:
        return self._failures.get(model_id)

    # ------------------------------------------------------------------ loading
    def get(self, model_id: str) -> LoadedModel:
        spec = self.registry.get(model_id)
        if not spec.enabled:
            raise ModelUnavailable(f"Model '{model_id}' is disabled in the registry.")
        if spec.kind == "algorithm":
            raise ModelUnavailable(f"'{model_id}' is an algorithm, not a loadable model.")

        with self._lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]

            path = spec.path(self.settings.checkpoint_dir)
            if not spec.weights_present(self.settings.checkpoint_dir):
                if not self.settings.allow_remote_model_download or not spec.repo_id:
                    msg = (
                        f"Weights for '{spec.name}' not found at {path}. "
                        f"Run: python scripts/download_models.py --only {model_id}"
                    )
                    self._failures[model_id] = msg
                    raise ModelUnavailable(msg, details={"model": model_id})
                source = spec.repo_id
            else:
                source = str(path)

            while len(self._cache) >= self.settings.max_loaded_models:
                evicted_id, _ = self._cache.popitem(last=False)
                log.info("model_evicted", extra={"model": evicted_id})
                gc.collect()

            from app.models.loaders import LOADERS  # local import keeps torch optional at import time

            loader = LOADERS.get(spec.backend)
            if loader is None:
                raise ModelUnavailable(f"No loader registered for backend '{spec.backend}'.")

            device = self.device_for(spec)
            t0 = time.perf_counter()
            try:
                loaded = loader(spec, source, device)
            except MemoryError as exc:
                gc.collect()
                self._failures[model_id] = "out of memory while loading"
                raise OutOfMemory(f"Not enough RAM to load '{spec.name}'.") from exc
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed, user-safe error
                if "out of memory" in str(exc).lower():
                    gc.collect()
                    raise OutOfMemory(f"Not enough memory to load '{spec.name}'.") from exc
                self._failures[model_id] = f"{type(exc).__name__}: {exc}"
                log.exception("model_load_failed", extra={"model": model_id})
                raise ModelUnavailable(f"Failed to load '{spec.name}': {type(exc).__name__}") from exc
            loaded.load_time_s = time.perf_counter() - t0
            self._cache[model_id] = loaded
            self._failures.pop(model_id, None)
            log.info("model_loaded", extra={"model": model_id, "device": device, "load_s": round(loaded.load_time_s, 2)})
            return loaded

    def unload_all(self) -> None:
        with self._lock:
            self._cache.clear()
        gc.collect()


@lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager(get_registry(), get_settings())
