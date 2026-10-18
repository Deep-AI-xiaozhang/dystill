#!/usr/bin/env python
"""L3 定点回放：用**存档证据**重判一遍，隔离「判定提示词/L3 逻辑」这一个变量。

为什么要这个脚本（而不是直接跑 bench_check）：
- `bench_check.py` 跑的是完整链路，L1 消歧、L2 检索都会重新调模型；改一句提示词，
  测出来的差异里混着 L1/L2 的漂移，**说不清是哪一层的功劳**（D8/D10 已证明条目级不可复现）。
- 存档 check.json 里每条声明都带着当时的证据、覆盖度、缺失硬事实，**只换了提示词**，
  重判结果与旧判定的差异就是提示词的效果，可归因、且便宜（不动转写/切帧/检索）。
- 它也是改提示词的正确姿势：先定点看方向对不对，方向对了再跑全场 3 轮。

用法：
    python scripts/replay_verdicts.py bench/claims_kun_master.csv --mode standard \
        --json-out bench/replay_prompt3.json [--limit 0] [--dry-run]

`--dry-run` 只拼 prompt 不调模型（看条目数、批量数、prompt 长度，零成本）。

`--re-retrieve`：证据不搬存档，改用**现版 L2 算法**从转写/画面重建（其余不变）——
这就是 L2 的定点回放：存档基线 = 旧选段 + 同一提示词，新跑 = 新选段 + 同一提示词，
差异只能来自选段算法。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_workshop.config import load_config
from distill_workshop.core.artifacts import read_if_valid
from distill_workshop.core.check import verifier as _verifier_mod
from distill_workshop.core.check.engine import _default_chat
from distill_workshop.core.check.evidence_retriever import (
    SourceIndex,
    business_leap,
    hard_miss,
)
from distill_workshop.core.check.models import Claim, Evidence
from distill_workshop.core.check.modes import get_mode
from distill_workshop.core.check.verifier import build_items, verify_claims

_ALIAS = {"一致": "ok", "不一致": "mismatch", "存疑": "doubt"}


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    return _ALIAS.get(v, v)


def load_old(key: str, data_dir: Path) -> list[dict]:
    p = Path(data_dir) / key.replace(":", "_") / "check.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("claims") or []


def rebuild(rows: list[dict], key: str) -> list[Claim]:
    """把存档声明还原成 Claim（带证据/覆盖度/缺失硬事实），判定清空后交给现提示词重判。"""
    out: list[Claim] = []
    fields = set(Evidence.__dataclass_fields__)
    for c in rows:
        ev = [Evidence(**{k: v for k, v in e.items() if k in fields})
              for e in (c.get("evidence") or []) if isinstance(e, dict)]
        out.append(Claim(
            cid=str(c.get("cid") or ""), text=str(c.get("text") or ""),
            section=str(c.get("section") or ""), kind=str(c.get("kind") or "other"),
            original=str(c.get("original") or ""), evidence=ev,
            score=float(c.get("score") or 0.0),
            miss_numbers=list(c.get("miss_numbers") or []),
            miss_names=list(c.get("miss_names") or []),
            miss_biz=list(c.get("miss_biz") or []),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 定点回放（用存档证据重判）")
    ap.add_argument("csv_paths", nargs="+",
                    help="已标注 CSV（可多份，多视频合并回放；声明正文 → 人工判定 的对照）")
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--json-out", dest="json_out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部），用来先小样探路")
    ap.add_argument("--dry-run", action="store_true", help="只拼 prompt、不调模型")
    ap.add_argument("--prompt-file", dest="prompt_file", default=None,
                    help="用文件里的提示词替换 verifier._PROMPT 再判（提示词 A/B 的 treatment 臂；"
                         "不加 = 当前源码提示词 = 对照臂）。同一天、同证据、只换这一个变量（D16）")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="判定批次之间的间隔秒数（限流严的通道调大，如 3）")
    ap.add_argument("--provider", default=None,
                    help="只用 config 里指定 name 的 provider 做判定（判定模型上限测试用；"
                         "缺省 = 按配置顺序的完整降级链）")
    ap.add_argument("--re-retrieve", action="store_true",
                    help="证据用现版 L2 算法重建（L2 定点回放），不用存档证据")
    args = ap.parse_args()

    app = load_config()
    data_dir = Path(app.data_dir)
    # 按 CSV **行**保留（而不是按唯一正文）：master CSV 里有 7 行正文重复，
    # 按正文去重会把 346 压成 339，与 bench_check / 口径 v1 的分母对不上。
    rows_in: list[tuple[str, str]] = []
    keys: dict[str, str] = {}
    csv_files: list[Path] = []
    for pth in args.csv_paths:
        hits = sorted(Path().glob(pth)) if any(c in pth for c in "*?[") else [Path(pth)]
        csv_files.extend(hits)
    for cf in csv_files:
        with cf.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                text = (r.get("claim") or "").strip()
                h = norm(r.get("human_verdict"))
                if text and h:
                    rows_in.append((text, h))
                    keys[text] = (r.get("key") or "").strip()
    wanted = {t for t, _ in rows_in}

    claims: list[Claim] = []
    old: dict[str, str] = {}
    seen: set[str] = set()
    idx_cache: dict[str, SourceIndex] = {}
    for key in sorted(set(keys.values())):
        if not key:
            continue
        stored = load_old(key, data_dir)
        if not stored:
            print(f"[!] 缺存档 check.json：{key}（先跑一次 dystill check）")
            continue
        if args.re_retrieve and key not in idx_cache:
            rd = Path(data_dir) / key.replace(":", "_")
            idx_cache[key] = SourceIndex.build(
                read_if_valid(rd / "transcript.txt", "transcript") or "",
                read_if_valid(rd / "vlm.txt", "vlm") or "", rd / "frames")
        for c in rebuild(stored, key):
            t = c.text.strip()
            if t not in wanted or t in seen:
                continue          # 正文相同的条目只调一次模型，统计时仍按行数还原
            seen.add(t)
            if args.re_retrieve:
                # 只换 L2 这一个变量：判定与硬事实清单都按现算法重算
                idx = idx_cache[key]
                score, ev = idx.retrieve(c.text, context=c.context)
                c.score = round(score, 4)
                c.evidence = ev
                src = idx.full_text()
                c.miss_numbers, c.miss_names = hard_miss(c.text, src)
                c.miss_biz = business_leap(c.text, src)
            old[t] = norm(next(x["verdict"] for x in stored
                               if str(x.get("text") or "").strip() == t))
            claims.append(c)
    if args.limit:
        claims = claims[:args.limit]
    # 限样时必须把**旧判定与统计分母一起收窄**，否则会出现「旧=全量、新=20 条」这种
    # 分母不一致的对比（自己先踩过一次：20 条那次旧列印的是 76.3%，其实是全量 346 条的数字）。
    keep = {c.text.strip() for c in claims}
    old = {t: v for t, v in old.items() if t in keep}
    print(f"[=] 存档可比对声明 {len(claims)} 条（CSV 已标注 {len(rows_in)} 行 → 唯一正文 {len(wanted)} 条）")

    m = get_mode(args.mode)
    batches = (len(claims) + m.batch_size - 1) // m.batch_size
    print(f"[=] 模式 {args.mode}，每批 {m.batch_size} 条 → {batches} 次调用；"
          f"首条 prompt {len(build_items(claims[:m.batch_size]))} 字")
    if args.dry_run:
        print("[i] --dry-run：不调模型")
        return 0

    if args.provider:
        picked = [x for x in app.ready_providers if x.name == args.provider]
        if not picked:
            print(f"[!] config 里没有名为 {args.provider} 的就绪 provider（检查 name 拼写/key 文件）")
            return 2
        from distill_workshop.providers.llm import chat
        to = int(app.pipeline.get("judge_timeout") or 300)
        chat_fn = (lambda messages: chat(messages, picked, max_tokens=4000, temperature=0.0, timeout=to))
        hint = f"{picked[0].name}/{picked[0].model}（单通道，不降级）"
    else:
        chat_fn, hint = _default_chat(app)
    if chat_fn is None:
        print("[!] 没有可用判定通道（providers 为空），无法回放")
        return 2
    print(f"[=] 判定通道：{hint}")
    t0 = time.time()
    if args.prompt_file:
        _verifier_mod._PROMPT = Path(args.prompt_file).read_text(encoding="utf-8")
        print(f"[*] 提示词已替换：{args.prompt_file}（{len(_verifier_mod._PROMPT)} 字）")
    verify_claims(claims, m, chat_fn, on_step=lambda s: print(s, flush=True),
                  batch_delay=args.delay)
    print(f"[=] 用时 {time.time() - t0:.0f}s")

    new = {c.text.strip(): c.verdict for c in claims}
    live = [(t, h) for t, h in rows_in if t in keep]
    cm_old = Counter((h, old[t]) for t, h in live if t in old)
    cm_new = Counter((h, new[t]) for t, h in live if t in new)

    def fp(cm):
        return sum(v for (h, e), v in cm.items() if e == "ok" and h != "ok")

    def acc(cm):
        tot = sum(cm.values())
        return (sum(v for (h, e), v in cm.items() if h == e) / tot) if tot else 0.0

    flips = Counter()
    for t in old:
        if new.get(t, old[t]) != old[t]:
            flips[(old[t], new[t])] += 1
    over_old = [(t, h) for t, h in live if h == "ok" and old.get(t) == "doubt"]
    over_new = [(t, h) for t, h in live if h == "ok" and new.get(t) == "doubt"]
    print("\n=== 结果（同一批条目、同一批证据，只换了 L3 提示词）===")
    print(f"{'':14s}{'一致率':>9s}{'假阳性':>8s}{'过度存疑':>10s}")
    print(f"{'旧(存档)':14s}{acc(cm_old):>8.1%}{fp(cm_old):>8d}{len(over_old):>10d}")
    print(f"{'新(现代码)':14s}{acc(cm_new):>8.1%}{fp(cm_new):>8d}{len(over_new):>10d}")
    print("\n--- 判定变化 ---")
    for (a, b), n in flips.most_common():
        print(f"  {a:>8s} → {b:<8s} {n:3d} 条")
    if not flips:
        print("  （无变化）")
    print("\n--- 新提示词下仍存的假阳性（人工非 ok、引擎 ok）---")
    for t, h in live:
        if new.get(t) == "ok" and h != "ok":
            print(f"  [人工 {h}] {t[:78]}")
    old_set = {t for t, _ in over_old}
    print("\n--- 新提示词下的过度存疑（人工 ok、引擎 doubt）；`+` = 新增 ---")
    for t, _ in over_new:
        print(f"  {'+' if t not in old_set else ' '} {t[:78]}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "mode": args.mode, "items": len(live),
            "acc_old": round(acc(cm_old), 4), "acc_new": round(acc(cm_new), 4),
            "fp_old": fp(cm_old), "fp_new": fp(cm_new),
            "over_doubt_old": len(over_old), "over_doubt_new": len(over_new),
            "flips": {f"{a}->{b}": n for (a, b), n in flips.items()},
            # 逐条落盘：FP/过度存疑清单只在日志里、acc 拆不开，归因必须靠这张表
            "per_claim": [{"text": t, "human": h, "old": old.get(t, ""),
                           "new": new.get(t, "")} for t, h in live],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[✓] 摘要已写 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())