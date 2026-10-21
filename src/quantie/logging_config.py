"""
Logging setup:
  stdout    → WARNING+ only, plain human-readable (no JSON)
  app.log   → INFO+, full JSON for analysis/debugging
  audit.log → audit events only, JSON
"""
import datetime
import logging
import logging.handlers
import sys
from pathlib import Path

import structlog


LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
APP_LOG = LOGS_DIR / "app.log"
AUDIT_LOG = LOGS_DIR / "audit.log"

_AUDIT_LOGGER_NAME = "quantie.audit"


class _PlainConsoleFormatter(logging.Formatter):
    """Single-line human-readable formatter for WARNING+ console output."""

    _PREFIX = {
        "WARNING":  "⚠  WARN",
        "ERROR":    "✖ ERROR",
        "CRITICAL": "✖  CRIT",
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).strftime("%H:%M:%S")
        prefix = self._PREFIX.get(record.levelname, record.levelname)
        # structlog stores the event in record.msg when using ProcessorFormatter;
        # fall back to standard getMessage() for non-structlog records.
        msg = record.getMessage()
        return f"[{ts}] {prefix}  {record.name}: {msg}"


def setup_logging(log_level: str = "INFO") -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    # Console: WARNING+ only, plain text — no JSON noise
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.WARNING)
    stdout_handler.setFormatter(_PlainConsoleFormatter())

    # File: INFO+, full JSON
    file_handler = logging.handlers.RotatingFileHandler(
        APP_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(json_formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    # Audit logger: separate file only, never propagates to root
    audit_handler = logging.handlers.RotatingFileHandler(
        AUDIT_LOG, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    audit_handler.setFormatter(json_formatter)
    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    audit_logger.setLevel(logging.DEBUG)
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def get_audit_logger() -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(_AUDIT_LOGGER_NAME)
