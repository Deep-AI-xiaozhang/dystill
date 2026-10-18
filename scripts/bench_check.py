#!/usr/bin/env python
"""第二步：跑引擎、对人工标注、算准确率（M1 验收）。

用法：
    python scripts/bench_check.py bench/claims.csv [--mode standard] [--out bench/result.md]
    python scripts/bench_check.py bench/claims_*.csv          # 多视频合并验收（口径 v1 要求 ≥3 题材）
    python scripts/bench_check.py bench/claims.csv --from-json <某次 check.json>

它做的事：读你标好的 CSV（可多份/可通配符，Windows 终端不展开通配符所以这里自己展开）
→ 拿到每条声明的**引擎判定**（默认现跑引擎；给 --from-json 就复用某次
已跑完的报告，省 token）→ 按声明正文配对 → 输出准确率、混淆矩阵、假阳性清单。

关键指标是**假阳性率**（人工判非 ok、引擎判 ok）：核对工具的致命失败模式，
不是"漏判存疑"，而是"把编造当成一致"——用户看到 ✅ 就不再回看原片，错误就此固化。
单轮判定看四项（**口径 v1.1**，DECISIONS D23）：**假阳性率 ≤1%**（硬线）、
**每条 ✅ 都带可回查证据**（硬线，✅ 没有时间戳＝无法回查，承诺就靠这句兑现）、
声明条数 ≥50、视频来源 ≥3 个。一致率**只作参考项**，不参与通过判定——
D22 实测漂移带宽 5.1pp，小于噪声量级的改进确认成本极高，「取最低值」仍是发布时的报告口径。
注意本脚本只出**单轮**数字：口径要求独立跑 3 轮、以**最差一轮**作验收结论（取最差而非平均：
FP 实测是稳定复现的系统性错误，多跑轮次不会摊薄），
聚合用 `--json-out` 存每轮摘要，再 `python scripts/bench_summary.py bench/run*.json`。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_workshop.config import load_config
from distill_workshop.core.artifacts import read_if_valid
from distill_workshop.core.check import analyze
from distill_workshop.core.check.engine import find_note
from distill_workshop.core.check.models import VERDICT_LABEL

VALID = {"ok", "mismatch", "doubt", "一致", "不一致", "存疑"}

# 口径 v1.1（D23）：FP 率与「✅ 带可回查证据」是硬线，一致率降为参考项。
# 这里与 bench_summary.py 的常量必须同步改——口径升级时最容易漏的就是散落的判定常量。
MAX_FP_RATE = 0.01
MIN_TOTAL = 50
MIN_KEYS = 3


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    return {"一致": "ok", "不一致": "mismatch", "存疑": "doubt"}.get(v, v)


def key_to_dir(key: str, data_dir: Path) -> Path:
    return Path(data_dir) / (key if key.startswith(("aweme_", "fp_")) else key.replace(":", "_"))


def engine_from_json(path: Path) -> dict[str, str]:
    """复用某次已跑完的报告：按判定句与原句双键索引（与 engine.load_previous 同一套键）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for c in data.get("claims") or []:
        if not c.get("verdict"):
            continue
        for k in (str(c.get("text") or "").strip(), str(c.get("original") or "").strip()):
            if k:
                out.setdefault(k, c["verdict"])
    return out


def evidence_gaps_from_json(path: Path) -> set[str]:
    """复用报告里「判 ok 但没有任何带时间戳证据」的声明（口径 v1.1 硬线二：✅ 必须可回查）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: set[str] = set()
    for c in data.get("claims") or []:
        if c.get("verdict") != "ok":
            continue
        if any(str(e.get("ts") or "").strip() for e in (c.get("evidence") or [])):
            continue
        for k in (str(c.get("text") or "").strip(), str(c.get("original") or "").strip()):
            if k:
                out.add(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="核对引擎基准评测")
    ap.add_argument("csv_paths", nargs="+",
                    help="已标注 CSV，可多份/可用通配符（如 bench/claims_*.csv，脚本自己展开）")
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--from-json", dest="from_json", default=None,
                    help="复用某次 check.json 的判定，不重新调用模型（省 token）")
    ap.add_argument("--out", default=None, help="报告输出路径（默认第一份 CSV 同目录 result.md）")
    ap.add_argument("--json-out", dest="json_out", default=None,
                    help="额外写机器可读摘要（口径 v1 的多轮聚合输入，配 bench_summary.py 用）")
    args = ap.parse_args()

    # Windows 的 cmd/PowerShell 不展开通配符，必须脚本自己 glob；给普通路径则原样收下
    paths: list[Path] = []
    for p in args.csv_paths:
        hits = sorted(glob.glob(p))
        paths.extend(Path(h) for h in hits) if hits else paths.append(Path(p))
    src = paths[0]
    rows: list[dict] = []
    for f in paths:
        with f.open(encoding="utf-8-sig", newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if norm(r.get("human_verdict")) in VALID]
    if not rows:
        print("[!] CSV 里没有已标注的行（human_verdict 需填 ok/mismatch/doubt，口径见 docs/LABELING.md）")
        return 2

    app = load_config()
    engine: dict[str, str] = {}
    providers: set[str] = set()   # 实际使用的判定通道（跨记录聚合；D19：聚合时要核对通道）
    ok_no_ts: set[str] = set()    # 判 ok 却拿不出带时间戳证据的声明（v1.1 硬线二的违约集合）
    if args.from_json:
        engine = engine_from_json(Path(args.from_json))
        ok_no_ts |= evidence_gaps_from_json(Path(args.from_json))
        print(f"[=] 复用报告：{args.from_json}（{len(engine)} 个键）")
    else:
        for key in sorted({r["key"] for r in rows}):
            rd = key_to_dir(key, Path(app.data_dir))
            note_file = find_note(rd)
            if not note_file:
                print(f"[!] 找不到记录，跳过该组：{rd}")
                continue
            note = read_if_valid(note_file, "note") or ""
            transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
            vlm = read_if_valid(rd / "vlm.txt", "vlm") or ""
            res = analyze(note, transcript, vlm, mode=args.mode, app=app,
                          frames_dir=rd / "frames", key=key, on_step=None)
            for c in res.claims:
                engine[c.text.strip()] = c.verdict
                if c.verdict == "ok" and not any(str(e.ts or "").strip() for e in c.evidence):
                    ok_no_ts.add(c.text.strip())
            if res.provider:
                providers.add(res.provider)
            print(f"[+] {key}: 引擎判定 {res.total} 条（{args.mode}）")

    pairs, unmatched, fp = [], [], []
    for r in rows:
        claim = (r["claim"] or "").strip()
        e, h = engine.get(claim), norm(r["human_verdict"])
        if e is None:
            unmatched.append(r)
            continue
        pairs.append((r, e, h))
        if e == "ok" and h != "ok":
            fp.append((r, e, h))

    total = len(pairs)
    hit = sum(1 for _, e, h in pairs if e == h)
    acc = hit / total if total else 0.0
    fp_rate = len(fp) / total if total else 0.0
    no_ev = [r for r, e, _ in pairs if e == "ok" and (r["claim"] or "").strip() in ok_no_ts]
    matrix = Counter((h, e) for _, e, h in pairs)
    # 口径 v1（D15）：样本构成也是验收对象——全部来自 1 个视频的基准测的是"背题"不是泛化
    n_keys = len({r.get("key", "") for r, _, _ in pairs})

    lines = [f"# 核对基准评测 · 模式 {args.mode}", "",
             f"- 已标注并配对：**{total}** 条（CSV 已标注 {len(rows)} 条，未匹配 {len(unmatched)} 条）",
             f"- 视频来源数：**{n_keys}** 个（口径 v1 要求 ≥3，见 D15）",
             f"- 总体一致率（参考项）：**{acc:.1%}**（{hit}/{total}）",
             (f"- 假阳性（人工非 ok 而引擎判 ok）：**{len(fp)}** 条 = {fp_rate:.2%}"
             f" {'⚠️ 需修' if fp_rate > MAX_FP_RATE else '✅'}"),
             f"- ✅ 缺可回查证据：**{len(no_ev)}** 条 {'⚠️ 需修' if no_ev else '✅'}",
             "", "## 混淆矩阵（行=人工，列=引擎）", "",
             "| 人工 \\ 引擎 | ✅ ok | ❌ mismatch | ⚠️ doubt |", "|---|---|---|---|"]
    for h in ("ok", "mismatch", "doubt"):
        lines.append(f"| {VERDICT_LABEL[h]} | " +
                     " | ".join(str(matrix.get((h, e), 0)) for e in ("ok", "mismatch", "doubt")) + " |")
    if fp:
        lines += ["", "## 假阳性清单（最高优先级：假 ✅ 会让用户不再回看原片）", ""]
        lines += [f"- [{r['id']}] {r['claim']}" for r, _, _ in fp]
    if no_ev:
        lines += ["", "## ✅ 缺可回查证据清单（v1.1 硬线二违约：✅ 拿不出时间戳＝用户回查不到）", ""]
        lines += [f"- [{r['id']}] {r['claim']}" for r in no_ev]
    if unmatched:
        lines += ["", "## 未匹配（笔记文本与本次判定集不一致，可能是笔记已改 / 引擎代码已变）", ""]
        lines += [f"- [{r['id']}] {r['claim']}" for r in unmatched[:20]]
    lines += ["", "## 本轮判定（单轮 · 口径 v1.1，D23）", "",
              (f"- 假阳性率 ≤{MAX_FP_RATE:.0%}（硬线）：{'✅ 达标' if fp_rate <= MAX_FP_RATE else '❌ 未达标'}"
              f"（{len(fp)}/{total} = {fp_rate:.2%}）"),
              f"- 每条 ✅ 带可回查证据（硬线）：{'✅ 达标' if not no_ev else f'❌ 未达标（{len(no_ev)} 条无时间戳）'}",
              f"- 声明条数 ≥{MIN_TOTAL}：{'✅' if total >= MIN_TOTAL else f'❌（当前 {total}）'}",
              f"- 视频来源数 ≥{MIN_KEYS}：{'✅' if n_keys >= MIN_KEYS else f'❌（当前 {n_keys}）'}",
              f"- —— 一致率（参考项，不参与通过判定）：{acc:.1%}",
              "",
              "> 注意：这是**单轮**数字。D8/D10 已证明条目级不可复现，最终验收必须独立跑 3 轮",
              "> 并用 `python scripts/bench_summary.py bench/run*.json` 聚合（口径 v1.1 取**最差一轮**）。"]

    report = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else src.parent / "result.md"
    out.write_text(report, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "mode": args.mode, "csv": str(src), "total": total, "hit": hit,
            "acc": round(acc, 4), "false_positives": len(fp), "fp_rate": round(fp_rate, 4),
            "ok_no_evidence": len(no_ev), "keys": n_keys,
            "unmatched": len(unmatched), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "providers": sorted(providers),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"[✓] 报告已写 {out}")
    # 口径 v1.1：硬线是 FP 率与「✅ 带证据」；一致率不参与（参考项）
    return 0 if (fp_rate <= MAX_FP_RATE and not no_ev
                 and total >= MIN_TOTAL and n_keys >= MIN_KEYS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
