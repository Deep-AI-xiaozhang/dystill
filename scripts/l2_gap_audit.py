#!/usr/bin/env python
"""L2 交付缺口批量审计（零成本、纯离线）。

对全部有真值标注的旧视频声明，逐条量两个数：
- **交付覆盖度**：L2 当前算法实际送达的证据对声明的字面覆盖（config B：top4/邻域2/钉事实）；
- **上界覆盖度**：整条原素材（转写+画面全文）对声明的字面覆盖——检索再完美也只能到这。

两者之差 = **检索损失**（证据在素材里但没送到）；上界本身低 = **词汇墙**
（同义改写/ASR 噪声导致字面上素材里就没有，检索无解，属于 L2 的天花板）。

口径 D26：只用已烧掉的旧视频做诊断，第二批留出集不碰。
"""
from __future__ import annotations

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

VIDS = {
    "aweme_7686005693291171112": 67,
    "aweme_7686442635317234953": 64,
    "aweme_7687942146737802981": 120,
    "aweme_7689851246873759030": 56,
    "aweme_7690808441719311625": 28,
}


def delivered(idx: SourceIndex, claim: str) -> float:
    _, ev = idx.retrieve(claim, top_k=4, pin=True, window=2)
    got = set()
    for e in ev:
        got |= tokenize(e.text)
    return coverage(tokenize(claim), got)


def main() -> int:
    rows = []
    for vid in VIDS:
        csv_path = Path(f"bench/claims_{vid}.csv")
        rd = Path(f"data/{vid}")
        if not rd.is_dir():
            print(f"[!] 缺记录目录，跳过 {vid}")
            continue
        idx = SourceIndex.build(read_if_valid(rd / "transcript.txt", "transcript") or "",
                                read_if_valid(rd / "vlm.txt", "vlm") or "",
                                rd / "frames")
        full = set(tokenize(idx.full_text()))
        with open(csv_path, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                hv = (r.get("human_verdict") or "").strip()
                claim = (r.get("claim") or "").strip()
                if not hv or not claim:
                    continue
                q = tokenize(claim)
                if not q:
                    continue
                d = delivered(idx, claim)
                u = coverage(q, full)
                rows.append({"id": r["id"], "key": r["key"], "claim": claim[:50],
                             "human": hv, "delivered": round(d, 3), "upper": round(u, 3),
                             "loss": round(u - d, 3)})

    n = len(rows)
    md = sum(r["delivered"] for r in rows) / n
    mu = sum(r["upper"] for r in rows) / n
    ml = sum(r["loss"] for r in rows) / n
    print(f"共 {n} 条已标声明（当前 L2 参数）")
    print(f"平均交付覆盖度 {md:.3f} | 平均上界 {mu:.3f} | 平均检索损失 {ml:.3f}")
    print()

    # 检索损失分布：上界高但交付低 = 检索本可救
    bucket = {"both_high": 0, "retrievable": 0, "wall": 0, "both_low": 0}
    for r in rows:
        hi, lo = 0.35, 0.15
        if r["upper"] >= hi:
            bucket["retrievable" if r["delivered"] < lo else "both_high"] += 1
        elif r["upper"] < lo:
            bucket["both_low"] += 1
        else:
            bucket["retrievable" if r["delivered"] < r["upper"] - 0.15 else "wall"] += 1
    print(f"四象限：交付上界双高 {bucket['both_high']} | 检索可救(上界高交付低) {bucket['retrievable']}"
          f" | 词汇墙(上界就低) {bucket['wall']} | 双低 {bucket['both_low']}")
    print()

    # 按「检索可救」分组看最差案例
    worst = sorted((r for r in rows if r["loss"] >= 0.25), key=lambda x: -x["loss"])[:12]
    print("检索损失最大的 12 条（证据在素材里但没送到）：")
    for r in worst:
        print(f"  [{r['id']} {r['human']}] 交付 {r['delivered']:.2f} / 上界 {r['upper']:.2f} | {r['claim']}")
    print()

    # 词汇墙样本
    wall = sorted((r for r in rows if r["upper"] < 0.15), key=lambda x: -x["upper"])[:8]
    print("词汇墙样本（整条素材字面上就没有，检索无解）：")
    for r in wall:
        print(f"  [{r['id']} {r['human']}] 上界 {r['upper']:.2f} | {r['claim']}")

    import json
    with open('bench/l2_gap_audit.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n明细已写 bench/l2_gap_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
