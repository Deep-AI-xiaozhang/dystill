"""执行台账（沿用 EXECUTION_LOG 台账纪律，改按身份键记录）。

作用：优化/排查"哪里慢、哪步失败"以台账为准，不翻会话记录。
JSONL 按行 append，与产物目录（身份键命名）可对账。
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def ledger_path(app) -> Path:
    return Path(app.cache_dir) / "ledger.jsonl"


def append_event(app, event: dict) -> None:
    p = ledger_path(app)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_events(app, key: str | None = None) -> list[dict]:
    p = ledger_path(app)
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if key is None or r.get("key") == key:
            out.append(r)
    return out
