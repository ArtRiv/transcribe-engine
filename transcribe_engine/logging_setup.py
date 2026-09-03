"""Plain-text rotating logs (D-21 — non-developer can scan).

Tray menu "View logs" opens this file via the OS default text viewer.
NOT JSON — D-21 reasoning is "non-developer can scan."
"""

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .platform.paths import log_dir

# V8 Data Protection — redact HF tokens from log output (Phase 7 plan 07-05 may collect one).
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


class _RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _HF_TOKEN_PATTERN.sub("hf_<REDACTED>", record.msg)
        if record.args:
            # Only attempt redaction on string args; leave others (numbers, paths) untouched.
            record.args = tuple(
                _HF_TOKEN_PATTERN.sub("hf_<REDACTED>", a) if isinstance(a, str) else a
                for a in record.args
            )
        return True


def setup_logging() -> Path:
    """Wire up RotatingFileHandler. Returns the log file path so __main__
    can pass it to the tray menu (D-09 'View logs')."""
    log_path = log_dir() / "transcribe-engine.log"
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_RedactSecretsFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    return log_path
