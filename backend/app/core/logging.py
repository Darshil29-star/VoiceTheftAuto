"""Structured logging. Logs request metadata only — never image content."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extra = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if extra:
            base += " " + json.dumps(extra, default=str, separators=(",", ":"))
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "urllib3", "PIL", "rasterio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_request(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra=fields)
