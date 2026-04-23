"""
Centralized debug logger for the biometric auth PoC.

Writes human-readable logs to `logs/debug.log` (txt) at the project root.
Use `get_logger(__name__)` in any module to grab a logger, or call the
helpers below from routers to record auth-stage decisions.

Log line format:
  2026-04-18 14:03:11.482 | INFO  | auth.keystroke | user=a@b.com | msg...

The file rotates at ~5 MB (keeps last 5 files) so it never eats the disk.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "debug.log")
AUTH_LOG_FILE = os.path.join(LOG_DIR, "auth_events.log")

os.makedirs(LOG_DIR, exist_ok=True)

# ── Formatter ─────────────────────────────────────────────────────────────────
class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # millisecond-precision timestamp
        ct = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        ms = int(record.msecs)
        ts = f"{ct}.{ms:03d}"

        user = getattr(record, "user", "-")
        stage = getattr(record, "stage", record.name)
        base = f"{ts} | {record.levelname:<5} | {stage} | user={user} | {record.getMessage()}"

        extra = getattr(record, "extra_data", None)
        if extra:
            try:
                base += " | data=" + json.dumps(extra, default=str, ensure_ascii=False)
            except Exception:
                base += f" | data={extra!r}"

        if record.exc_info:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return base


# ── Setup ─────────────────────────────────────────────────────────────────────
_INITIALIZED = False

def _build_handler(path: str) -> RotatingFileHandler:
    h = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    h.setFormatter(_Formatter())
    return h


def init_logging(level: int = logging.DEBUG) -> None:
    """Idempotent — safe to call multiple times."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger("biometric")
    root.setLevel(level)
    root.propagate = False

    # general debug file
    root.addHandler(_build_handler(LOG_FILE))

    # dedicated auth-events file (only auth.* loggers)
    auth_handler = _build_handler(AUTH_LOG_FILE)
    auth_handler.addFilter(lambda r: r.name.startswith("biometric.auth"))
    root.addHandler(auth_handler)

    # also echo to console so you can tail it live
    console = logging.StreamHandler()
    console.setFormatter(_Formatter())
    root.addHandler(console)

    _INITIALIZED = True
    root.info(
        "logger initialized",
        extra={"stage": "logger", "user": "-", "extra_data": {"log_file": LOG_FILE}},
    )


def get_logger(name: str) -> logging.Logger:
    init_logging()
    # namespace everything under "biometric" so one root controls all output
    return logging.getLogger(f"biometric.{name}")


# ── Convenience helpers for auth flow ─────────────────────────────────────────
def log_auth_stage(
    stage: str,
    username: str,
    result: str,  # "granted" | "denied" | "uncertain" | "error"
    *,
    score: float | None = None,
    threshold: float | None = None,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Record a single auth-stage decision. Example:
      log_auth_stage("keystroke", "a@b.com", "granted",
                     score=0.82, threshold=0.55, model="rf",
                     extra={"samples": 12, "maturity": 4})
    """
    logger = get_logger(f"auth.{stage}")
    data: dict[str, Any] = {"result": result}
    if score is not None:     data["score"] = round(float(score), 4)
    if threshold is not None: data["threshold"] = round(float(threshold), 4)
    if model is not None:     data["model"] = model
    if extra:                 data.update(extra)

    msg = f"{stage} -> {result}"
    if score is not None and threshold is not None:
        msg += f" (score={data['score']} vs thr={data['threshold']})"

    logger.log(level, msg, extra={"stage": f"auth.{stage}", "user": username, "extra_data": data})

    # Update the per-user login-attempts summary file.
    # Rebuild is cheap (current auth_events.log is small, rotates at 5MB) and
    # keeps the summary accurate after each stage event.
    try:
        from backend.utils.login_attempt_log import rebuild_safe
        rebuild_safe()
    except Exception:
        try:
            from utils.login_attempt_log import rebuild_safe
            rebuild_safe()
        except Exception:
            pass


def log_error(stage: str, username: str, err: BaseException, *, extra: dict[str, Any] | None = None) -> None:
    logger = get_logger(f"auth.{stage}")
    logger.error(
        f"{type(err).__name__}: {err}",
        exc_info=(type(err), err, err.__traceback__),
        extra={"stage": f"auth.{stage}", "user": username, "extra_data": extra or {}},
    )


# ── FastAPI middleware: one line per request ──────────────────────────────────
async def request_logging_middleware(request, call_next):
    """
    Install via:  app.middleware("http")(request_logging_middleware)
    Logs: method, path, status, duration_ms, client IP, and username if the
    request body carries one (best-effort, JSON only).
    """
    logger = get_logger("http")
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    username = "-"

    # Best-effort username sniff (only for JSON POSTs, body is re-buffered)
    body_bytes = b""
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            if body_bytes and b"username" in body_bytes[:4096]:
                payload = json.loads(body_bytes.decode("utf-8", errors="ignore"))
                if isinstance(payload, dict):
                    username = str(payload.get("username", "-"))
        except Exception:
            pass

        # Re-inject body so downstream handlers can still read it
        async def _receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        request._receive = _receive  # type: ignore[attr-defined]

    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as e:
        log_error("http", username, e, extra={"path": request.url.path, "method": request.method})
        raise
    finally:
        dur_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            f"{request.method} {request.url.path} -> {status}",
            extra={
                "stage": "http",
                "user": username,
                "extra_data": {
                    "status": status,
                    "duration_ms": round(dur_ms, 1),
                    "client": client,
                    "query": str(request.url.query) or None,
                },
            },
        )
