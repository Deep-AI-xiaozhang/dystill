#!/usr/bin/env python
"""L2 召回体检（**零成本**）：判定错之前，先量"证据到底有没有送到模型面前"。

用法：
    python scripts/recall_audit.py <记录目录> [--csv bench/claims.csv]

为什么先做这个：首轮 M1 基准一致率 42.5%，偏差全在「引擎比人严」。
责任可能在两头——① L2 没把出处送上去；② L3 判定太抠字眼。
前者跟模型无关，纯本地就能量出来，所以先量它，别猜。

量法：对每条声明，把 L2 实际交付的证据拼起来，算它对声明的**字面覆盖度**，
再按人工标注分组（我判 ok 的一组、我判非 ok 的一组）。
- 若 ok 组覆盖度明显低 → 是 L2 的锅（证据没送到）；
- 若 ok 组覆盖度本来就高、却仍被判存疑 → 是 L3 的锅（送到了也不认），得改提示词/判定线。
顺带用覆盖度做一个**离线代理判定**（覆盖率 ≥ 阈值 → ok，且带缺失数字一律不 ok），
与人工标注对一下，就能免费估出"把证据补齐后一致率能到哪"。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_workshop.core.artifacts import read_if_valid
from distill_workshop.core.check.evidence_retriever import (
    SourceIndex,
    coverage,
    tokenize,
)

# (名称, top_k, pin, window)  —— A 近似首轮基准时的行为，B 是当前行为
CONFIGS = (("A 旧口径(top2/无邻域)", 2, False, 0),
           ("B 现口径(top4/邻域2/钉事实)", 4, True, 2))


def delivered_cov(idx: SourceIndex, claim: str, top_k: int, pin: bool, window: int) -> float:
    _, ev = idx.retrieve(claim, top_k=top_k, pin=pin, window=window)
    got = set()
    for e in ev:
        got |= tokenize(e.text)
    return coverage(tokenize(claim), got)


def best_threshold(rows: list[tuple[float, str]]) -> tuple[float, float, int]:
    """在覆盖度上扫一遍阈值，取「与人工标注一致率最高」的那个。返回 (阈值, 一致率, 假阳性数)。"""
    best = (0.0, -1.0, 0)
    for t in [i / 100 for i in range(5, 96)]:
        ok = 0
        fp = 0
        for cov, hv in rows:
            pred = "ok" if cov >= t else "doubt"
            if pred == hv:
                ok += 1
            if pred == "ok" and hv != "ok":
                fp += 1
        acc = ok / len(rows) if rows else 0.0
        if (acc, -fp) > (best[1], -best[2]):
            best = (t, acc, fp)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="L2 召回体检（零成本）")
    ap.add_argument("record_dir")
    ap.add_argument("--csv", default="bench/claims.csv")
    args = ap.parse_args()

    rd = Path(args.record_dir)
    idx = SourceIndex.build(read_if_valid(rd / "transcript.txt", "transcript") or "",
                            read_if_valid(rd / "vlm.txt", "vlm") or "", rd / "frames")
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)
            if (r.get("human_verdict") or "").strip()]
    if not rows:
        print("[!] CSV 里没有已标注的行")
        return 2
    print(f"记录：{rd.name}｜已标注 {len(rows)} 条\n")

    for name, top_k, pin, window in CONFIGS:
        vals = []
        for r in rows:
            hv = "ok" if r["human_verdict"] == "ok" else "no"
            vals.append((delivered_cov(idx, r["claim"], top_k, pin, window), hv))
        oks = [c for c, hv in vals if hv == "ok"]
        nos = [c for c, hv in vals if hv == "no"]
        t, acc, fp = best_threshold(vals)
        print(f"■ {name}")
        print(f"   我判 ok 的 {len(oks):2} 条：交付证据覆盖度 均值 {sum(oks)/len(oks):.3f}　"
              f"最低 {min(oks):.3f}")
        print(f"   我判非ok 的 {len(nos):2} 条：覆盖度 均值 {sum(nos)/len(nos):.3f}　"
              f"最高 {max(nos):.3f}")
        print(f"   离线代理（覆盖度≥{t:.2f} 判 ok）：与人工标注一致 {acc:.1%}，假阳性 {fp} 条")
        # 最需要关注的：我判 ok 但交付覆盖度极低的（证据确实没送到）
        bad = [(c, r["id"]) for (c, hv), r in zip(vals, rows) if hv == "ok" and c < 0.35]
        print(f"   我判 ok 但交付覆盖度 <0.35 的：{len(bad)} 条 → "
              f"{'证据确实没送到，属 L2 责任' if bad else '证据送到了'}")
        for c, i in bad[:6]:
            print(f"      {i} 覆盖度 {c:.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
