#!/usr/bin/env python
"""第一步：导出待标注的声明清单（不需要 API key）。

用法：
    python scripts/make_bench.py <记录目录|笔记文件> [...] -o bench/claims.csv

只跑 L1（规则拆分）+ L2（证据回查），把每条声明连同**原视频证据与时间戳**导成 CSV，
`human_verdict` / `human_note` 两列留空。

为什么要人工标注：M1 验收是"50 条基准准确率 ≥90%"，而基准必须人来定
（拿引擎自己的判定当标准答案 = 自己给自己判分）。本脚本把人工成本从
"翻视频找声明"降到"对着证据改一个字母"——填 ok / mismatch / doubt 即可。
填完跑 scripts/bench_check.py 算准确率（那一步才会调模型）。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_workshop.config import load_config
from distill_workshop.core.artifacts import read_if_valid
from distill_workshop.core.check import resolve_record_dir
from distill_workshop.core.check.claim_extractor import extract_claims
from distill_workshop.core.check.engine import find_note
from distill_workshop.core.check.evidence_retriever import SourceIndex, retrieve_all
from distill_workshop.core.check.modes import get_mode

FIELDS = ["id", "key", "section", "kind", "claim", "evidence_ts", "evidence",
          "human_verdict", "human_note"]


def collect(target: str, app, mode_name: str, offset: int) -> list[dict]:
    rd = resolve_record_dir(target, app)
    note_file = find_note(rd)
    if not note_file:
        print(f"[!] 跳过（目录里没笔记）：{rd}")
        return []
    note = read_if_valid(note_file, "note") or ""
    transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
    vlm = read_if_valid(rd / "vlm.txt", "vlm") or ""

    mode = get_mode(mode_name)
    claims, _ = extract_claims(note, mode, chat_fn=None)      # 只做规则拆分，不花 token
    retrieve_all(claims, SourceIndex.build(transcript, vlm, rd / "frames"), on_step=None)

    rows = [{
        "id": f"B{offset + i:03d}",
        "key": rd.name,
        "section": c.section,
        "kind": c.kind,
        "claim": c.text,
        "evidence_ts": c.ts,
        "evidence": (c.evidence[0].text[:160] if c.evidence else ""),
        "human_verdict": "",           # ← 你要填的列：ok / mismatch / doubt
        "human_note": "",
    } for i, c in enumerate(claims, 1)]
    print(f"[+] {rd.name}: 导出 {len(rows)} 条待标注声明")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="生成核对基准集草稿 CSV（不调模型）")
    ap.add_argument("targets", nargs="+", help="记录目录 / 笔记文件 / 抖音链接")
    ap.add_argument("-o", "--out", default="bench/claims.csv")
    ap.add_argument("--mode", default="standard")
    args = ap.parse_args()

    app = load_config()
    rows: list[dict] = []
    for t in args.targets:
        try:
            rows += collect(t, app, args.mode, len(rows))
        except Exception as e:  # noqa: BLE001 —— 单个目标失败不打断整批
            print(f"[!] {t}: {type(e).__name__}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[✓] 已写 {out}（{len(rows)} 条）。填好 human_verdict 后运行：\n"
          f"    python scripts/bench_check.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
