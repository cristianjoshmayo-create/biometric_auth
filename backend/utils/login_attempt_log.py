"""
Login-attempt aggregator.

Parses `logs/auth_events.log` and writes a compact per-attempt summary to
`logs/login_attempts.log`, one line per login attempt per user, numbered
per user.

Line format:
  2026-04-19 14:04:53.115 | user=nayvejessica@gmail.com | attempt 1 : keystroke: 85.3% - accepted
  2026-04-19 14:08:27.626 | user=nayvejessica@gmail.com | attempt 3 : keystroke: 73.9%, voice: 87.2%, fusion: 81.2% - accepted
  2026-04-19 14:15:40.732 | user=nayvejessica@gmail.com | attempt 9 : keystroke: 47.7% - denied

Attempt boundary rules (when scanning events for a single user):
  - A new attempt starts on every `auth.password` event.
  - An attempt closes at the next `auth.fusion` event, OR when the next
    `auth.password` event for the same user arrives (whichever first).
  - If only a keystroke event followed the password (no voice/fusion),
    the keystroke result determines accepted/denied.
  - If only a password event fired (no keystroke), password result decides.

Rebuilt from scratch each call — auth_events.log is the source of truth.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
AUTH_LOG_FILE = os.path.join(LOG_DIR, "auth_events.log")
ATTEMPTS_LOG_FILE = os.path.join(LOG_DIR, "login_attempts.log")

_LINE_RE = re.compile(
    r"^(?P<ts>\S+ \S+)\s+\|\s+\S+\s+\|\s+auth\.(?P<stage>\w+)\s+\|\s+"
    r"user=(?P<user>[^\s|]+)\s+\|\s+.*?(?:\|\s+data=(?P<data>\{.*\}))?\s*$"
)

_lock = threading.Lock()


def _parse_event(line: str) -> dict | None:
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    stage = m.group("stage")
    if stage not in ("password", "keystroke", "voice", "fusion"):
        return None
    user = m.group("user")
    if user in ("-", ""):
        return None
    data_raw = m.group("data")
    result = None
    score = None
    if data_raw:
        try:
            d = json.loads(data_raw)
            result = d.get("result")
            score = d.get("score")
            # voice uses `ecapa_similarity` as its primary score; fall back to it
            if stage == "voice" and (score is None or score == 0) and "ecapa_similarity" in d:
                score = d.get("ecapa_similarity")
        except Exception:
            pass
    return {
        "ts":     m.group("ts"),
        "stage":  stage,
        "user":   user,
        "result": result,
        "score":  score,
    }


def _fmt_pct(v: Any) -> str | None:
    try:
        if v is None:
            return None
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return None


def _emit_line(
    ts: str, user: str, attempt_no: int,
    pw_ok: bool | None,
    ks: float | None, voice: float | None, fusion: float | None,
    final: str,
) -> str:
    parts: list[str] = []
    if pw_ok is False:
        parts.append("password: failed")
    elif pw_ok is True and ks is None and voice is None and fusion is None:
        parts.append("password only (no biometric step logged)")
    if ks is not None:
        parts.append(f"keystroke: {_fmt_pct(ks)}")
    if voice is not None:
        parts.append(f"voice: {_fmt_pct(voice)}")
    if fusion is not None:
        parts.append(f"fusion: {_fmt_pct(fusion)}")
    body = ", ".join(parts) if parts else "-"
    return f"{ts} | user={user} | attempt {attempt_no} : {body} - {final}"


def rebuild() -> int:
    """
    Rebuild login_attempts.log from auth_events.log. Returns lines written.
    Safe to call repeatedly; writes the file atomically.
    """
    if not os.path.exists(AUTH_LOG_FILE):
        return 0

    with _lock:
        events: list[dict] = []
        # auth_events.log may be rotated; we only read the current file.
        with open(AUTH_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                ev = _parse_event(line)
                if ev:
                    events.append(ev)

        # Per-user sliding window: build attempts by grouping events between
        # successive password events for the same user.
        per_user_open: dict[str, dict] = {}
        per_user_count: dict[str, int] = {}
        out_lines: list[str] = []

        def _close(user: str) -> None:
            st = per_user_open.pop(user, None)
            if not st:
                return
            per_user_count[user] = per_user_count.get(user, 0) + 1
            n = per_user_count[user]

            pw_ok   = st.get("pw_ok")
            ks      = st.get("ks_score")
            ks_res  = st.get("ks_result")
            voice   = st.get("voice_score")
            fusion  = st.get("fusion_score")
            fus_res = st.get("fusion_result")

            # Decide the final verdict
            if pw_ok is False:
                final = "denied"
            elif fus_res is not None:
                final = "accepted" if fus_res == "granted" else "denied"
            elif ks_res is not None:
                final = "accepted" if ks_res == "granted" else "denied"
            else:
                final = "accepted"  # password-only success, no further events logged

            out_lines.append(_emit_line(
                st["ts"], user, n, pw_ok, ks, voice, fusion, final,
            ))

        for ev in events:
            user  = ev["user"]
            stage = ev["stage"]
            res   = ev["result"]
            sc    = ev["score"]

            if stage == "password":
                # Close any prior open attempt for this user first
                _close(user)
                per_user_open[user] = {
                    "ts":    ev["ts"],
                    "pw_ok": (res == "granted"),
                }
                if res != "granted":
                    # failed-password attempt closes immediately
                    _close(user)
            else:
                st = per_user_open.get(user)
                if st is None:
                    # stage event with no preceding password — synthesize a bare attempt
                    st = {"ts": ev["ts"], "pw_ok": None}
                    per_user_open[user] = st
                if stage == "keystroke":
                    st["ks_score"]  = sc
                    st["ks_result"] = res
                elif stage == "voice":
                    st["voice_score"]  = sc
                    st["voice_result"] = res
                elif stage == "fusion":
                    st["fusion_score"]  = sc
                    st["fusion_result"] = res
                    _close(user)  # fusion always terminates the attempt

        # Flush any still-open attempts (e.g. keystroke-only granted with no fusion follow-up)
        for user in list(per_user_open.keys()):
            _close(user)

        tmp_path = ATTEMPTS_LOG_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in out_lines:
                f.write(line + "\n")
        os.replace(tmp_path, ATTEMPTS_LOG_FILE)

        return len(out_lines)


def rebuild_safe() -> None:
    """Fire-and-forget wrapper; swallows errors so logging never breaks auth."""
    try:
        rebuild()
    except Exception:
        pass


if __name__ == "__main__":
    n = rebuild()
    print(f"Wrote {n} attempt lines to {ATTEMPTS_LOG_FILE}")
