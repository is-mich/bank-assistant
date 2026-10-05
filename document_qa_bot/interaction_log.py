"""
interaction_log.py
------------------
Append-only JSON-lines log of user-app interactions, with obvious personal data masked.

Masking is a safety net, not a guarantee: the app also tells users not to enter personal data.
"""

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

REDACTIONS = [
    (re.compile(r"\b[A-Za-z]{2}\d{2}(?:\s?[A-Za-z0-9]{4}){3,7}(?:\s?[A-Za-z0-9]{1,3})?\b"), "[IBAN]"),
    (re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), "[ΑΡΙΘΜΟΣ]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
]


def redact(text: str) -> str:
    """Mask IBANs, long digit sequences (card numbers) and e-mail addresses."""
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class InteractionLogger:
    """Thread-safe JSONL writer; `static` fields (e.g. the experiment) go into every record."""

    def __init__(self, path: str | Path, **static: Any):
        self.path = Path(path)
        self.static = static
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            **self.static,
            **_clean(fields),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
