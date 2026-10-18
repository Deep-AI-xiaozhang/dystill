#!/usr/bin/env python
"""选模型：用**本项目自己的基准集**实测候选模型，而不是看排行榜。

用法：
    MODELSCOPE_API_KEY=xxx python scripts/model_pick.py --csv bench/claims.csv \
        --check <check.json> --models "Qwen/Qwen3.8-27B" "deepseek-ai/DeepSeek-V4.1-Flash" ...
    # 换别家（任何 OpenAI 兼容端点）：--base + --key-env/--key-file
    python scripts/model_pick.py --csv bench/claims.csv --check <check.json> \
        --models agnes-3.0-flash --base https://api.agnes-ai.cn/v1/chat/completions \
        --key-file /path/to/your_key.txt

为什么这么做：换模型要看的是"在这个任务上判得准不准 + 多快"，通用榜单答不了。
本脚本把基准集里已人工标注的声明连同 L2 回查到的证据，按真实的 L3 提示词发给候选模型，
再和人工标注对一致率与假阳性——跟正式跑核对用的是同一套提示词与解析器，结果可直接比较。

只在本地跑，从环境变量或 key 文件取 key（不写进任何文件）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace

from distill_workshop.core.check.models import Claim, Evidence
from distill_workshop.core.check.modes import get_mode
from distill_workshop.core.check.verifier import (
    _PROMPT,
    apply_gates,
    build_items,
    parse_verdicts,
)

BASE = "https://api-inference.modelscope.cn/v1/chat/completions"


def load_cases(csv_path: Path, check_json: Path, batch: int) -> list[tuple[list[Claim], dict]]:
    """从基准 CSV 与某次 check.json 组装 (一批声明, {声明正文: 人工标注})。"""
    labels = {r["claim"].strip(): r["human_verdict"] for r
              in csv.DictReader(csv_path.open(encoding="utf-8-sig", newline=""))
              if (r.get("human_verdict") or "").strip()}
    data = json.loads(check_json.read_text(encoding="utf-8"))
    claims = []
    for c in data["claims"]:
        if c["text"].strip() not in labels:
            continue
        claims.append(Claim(
            cid=c.get("cid", ""), text=c["text"], section=c.get("section", ""),
            original=c.get("original", ""), miss_numbers=c.get("miss_numbers") or [],
            miss_names=c.get("miss_names") or [], miss_biz=c.get("miss_biz") or [],
            score=c.get("score", 0.0),
            evidence=[Evidence(source=e.get("source", ""), text=e.get("text", ""),
                               ts=e.get("ts", ""), score=e.get("score") or 0.0,
                               fact_hit=e.get("fact_hit") or 0.0)
                      for e in (c.get("evidence") or [])]))
    return [(claims[i:i + batch], labels) for i in range(0, len(claims), batch)]


def ask(model: str, prompt: str, key: str, base: str = BASE, timeout: int = 180,
        max_tokens: int = 4000) -> tuple[str, float, str]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(base, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        msg = ((d.get("choices") or [{}])[0].get("message") or {})
        return (msg.get("content") or "").strip(), time.time() - t, ""
    except urllib.error.HTTPError as e:
        return "", time.time() - t, f"HTTP {e.code}: {e.read().decode()[:90]}"
    except Exception as e:  # noqa: BLE001
        return "", time.time() - t, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="用本项目基准集实测候选模型")
    ap.add_argument("--csv", default="bench/claims.csv")
    ap.add_argument("--check", required=True, help="提供证据的 check.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="只跑前 N 批（0=全部）。思考型模型很慢时用它先拿个初步信号")
    ap.add_argument("--base", default=BASE, help="OpenAI 兼容的 chat/completions 端点")
    ap.add_argument("--key-env", default="MODELSCOPE_API_KEY")
    ap.add_argument("--key-file", default="", help="也可直接给 key 文件路径（各家 *.txt）")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="有些模型对 max_tokens 有硬上限（如 glm-4v-flash 只能 1024）")
    ap.add_argument("--dump", default="", help="把逐条明细写成 CSV（定位假阳性必用）")
    ap.add_argument("--gates", action="store_true",
                    help="额外过一遍本项目的四道确定性闸门（模拟产品实际结论，而非裸模型判断）")
    ap.add_argument("--mode", default="fast", help="配合 --gates 用哪种核对模式")
    args = ap.parse_args()

    if args.key_file:
        key = Path(args.key_file).read_text(encoding="utf-8").strip()
    else:
        key = os.environ.get(args.key_env, "").strip()
    if not key:
        print(f"[x] 没有 key：设 {args.key_env} 环境变量，或用 --key-file（不要写进文件）")
        return 2
    cases = load_cases(Path(args.csv), Path(args.check), args.batch)
    if args.max_batches > 0:
        cases = cases[:args.max_batches]
    total = sum(len(c) for c, _ in cases)
    print(f"基准：{total} 条声明 / {len(cases)} 批\n")

    rows = []
    details: list[dict] = []
    for model in args.models:
        agree = fp = judged = 0
        secs, errs = [], []
        for batch, labels in cases:
            raw, sec, err = ask(model, _PROMPT.format(items=build_items(batch)), key,
                                base=args.base, max_tokens=args.max_tokens)
            secs.append(sec)
            if err:
                errs.append(err)
                continue
            got = parse_verdicts(raw, len(batch))
            for i, c in enumerate(batch, 1):
                hv = labels.get(c.text.strip())
                if not hv or i not in got:
                    continue
                judged += 1
                verdict = got[i]["verdict"]
                if args.gates:      # 加上本项目的四道确定性闸门，看产品实际会给什么结论
                    probe = replace(c, verdict=verdict, reason=got[i].get("reason", ""))
                    apply_gates(probe, get_mode(args.mode))
                    verdict = probe.verdict
                same = verdict == hv
                agree += same
                if verdict == "ok" and hv != "ok":
                    fp += 1
                details.append({"model": model, "cid": c.cid, "human": hv,
                                "model_verdict": verdict, "same": int(same),
                                "claim": c.text.strip(), "reason": got[i]["reason"]})
        rows.append((model, judged, agree / judged if judged else 0.0, fp,
                     sum(secs) / len(secs) if secs else 0.0, errs[:1]))

    if args.dump:
        out = Path(args.dump)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "cid", "human", "model_verdict",
                                              "same", "claim", "reason"])
            w.writeheader()
            w.writerows(details)
        print(f"[✓] 逐条明细已写 {out}"
              f"（找假阳性：human 非 ok 且 model_verdict=ok）\n")

    print(f"{'模型':40} {'判定条数':>7} {'一致率':>7} {'假阳性':>6} {'平均耗时':>8}")
    print("-" * 78)
    for m, n, acc, fp, sec, err in sorted(rows, key=lambda r: (-r[2], r[4])):
        flag = "" if not err else f"  ← {err[0][:34]}"
        print(f"{m:40} {n:>7} {acc:>6.1%} {fp:>6} {sec:>7.1f}s{flag}")
    print("\n口径：一致率 = 与人工标注同判；假阳性 = 人工判非 ok 而模型判 ok（最要命，优先看它）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
