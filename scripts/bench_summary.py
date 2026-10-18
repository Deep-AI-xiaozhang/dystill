#!/usr/bin/env python
"""第三步：把 bench_check 的多轮摘要聚合成验收结论（口径 v1.1，DECISIONS D23）。

用法：
    python scripts/bench_check.py bench/claims.csv --json-out bench/run1.json
    # ……独立再跑两轮（不要 --from-json 复用判定，那不是"重复测量"，只是重算同一批数）……
    python scripts/bench_summary.py bench/run1.json bench/run2.json bench/run3.json

为什么必须有这一步：D8 实测同一基准两次跑分差 7.5pp、D10 实测条目级不可复现
（聚合 65.0% 相同但 ❌ 条数 0 vs 5）。单次数字当结论 = 掷骰子。

判定结构（v1.1，D23）：**假阳性率 ≤1%**（硬线）+ **每条 ✅ 带可回查证据**（硬线）
+ 轮数 / 条数 / 视频数门槛。一致率**只报数字、不参与通过判定**——D22 实测漂移带宽
5.1pp，小于噪声量级的改进确认成本已反转，继续为它刷分不划算。

聚合仍刻意保守：**硬线取最差一轮**（不取平均）。D22 实测 FP 是稳定复现的系统性错误
——三轮 8 条去重只剩 4 条、其中 2 条三轮全中，多跑轮次不会把它摊薄，取平均会放过它。
轮数 <3 只报区间、不下"通过"结论。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

MAX_FP_RATE = 0.01
MIN_TOTAL = 50
MIN_KEYS = 3
MIN_RUNS = 3


def load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in ("acc", "total", "keys", "false_positives", "mode"):
        if k not in d:
            raise ValueError(f"{path} 缺少字段 {k}（请确认是 bench_check.py --json-out 的产物）")
    return d


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    runs = [load(Path(p)) for p in argv]
    accs = [r["acc"] for r in runs]
    lo, hi = min(accs), max(accs)
    med = statistics.median(accs)
    # 旧产物没有 fp_rate 字段：不留死角，直接按 FP 条数 / 配对条数现算
    fp_rates = [r.get("fp_rate", (r["false_positives"] / r["total"]) if r["total"] else 1.0)
                for r in runs]
    worst_fp = max(fp_rates)
    fp_total = sum(r["false_positives"] for r in runs)
    total_all = sum(r["total"] for r in runs)
    fp_rate_all = fp_total / total_all if total_all else 1.0
    # v1.1 硬线二：旧产物没采集该字段 ⇒ 记「未采集」，不能当通过（没验证过的不许算过）
    ev_uncollected = [i for i, r in enumerate(runs, 1) if r.get("ok_no_evidence") is None]
    ev_missing = [i for i, r in enumerate(runs, 1) if r.get("ok_no_evidence")]

    lines = ["# 核对验收聚合 · 口径 v1.1（D23）", "",
             "| 轮次 | 模式 | 一致率 | 配对条数 | 视频数 | 假阳性 | 假阳性率 | ✅ 缺证据 |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs, 1):
        nev = r.get("ok_no_evidence")
        lines.append(f"| {i} | {r['mode']} | {r['acc']:.1%} | {r['total']} | {r['keys']} | "
                     f"{r['false_positives']} | {fp_rates[i - 1]:.2%} | "
                     f"{'未采集' if nev is None else nev} |")
    lines += ["",
              (f"- 一致率（**参考项**，不参与通过判定）：区间 **{lo:.1%} ~ {hi:.1%}**"
              f"（中位 {med:.1%}），最低 {lo:.1%}"),
              f"- 假阳性率：最差一轮 **{worst_fp:.2%}**"
              + (f"，合计 {fp_total}/{total_all} = **{fp_rate_all:.2%}**" if len(runs) > 1 else ""),
              f"- 一致率区间宽度 {hi - lo:.1%}"
              + ("（>10pp：先查判定为什么漂，别急着下结论）" if hi - lo > 0.10 else ""),
              "", "## 验收判定（全部满足才通过）", ""]
    ok_runs = len(runs) >= MIN_RUNS
    if ev_uncollected:
        ev_ok_all = False
        ev_now = f"轮 {ev_uncollected} 未采集（旧产物；重跑 bench_check --json-out 即补上）"
    elif ev_missing:
        ev_ok_all, ev_now = False, f"轮 {ev_missing} 存在 ✅ 缺时间戳"
    else:
        ev_ok_all, ev_now = True, "全部 ✅ 都带时间戳"
    checks = [
        (f"独立轮数 ≥{MIN_RUNS}", ok_runs, f"当前 {len(runs)}"),
        (f"每轮假阳性率 ≤{MAX_FP_RATE:.0%}（硬线）", worst_fp <= MAX_FP_RATE,
         f"最差一轮 {worst_fp:.2%}"
         + (f"，合计 {fp_rate_all:.2%}" if len(runs) > 1 else "")),
        ("每条 ✅ 带可回查证据（硬线）", ev_ok_all, ev_now),
        (f"声明条数 ≥{MIN_TOTAL}", all(r["total"] >= MIN_TOTAL for r in runs),
         f"各轮 {[r['total'] for r in runs]}"),
        (f"视频来源 ≥{MIN_KEYS}", all(r["keys"] >= MIN_KEYS for r in runs),
         f"各轮 {[r['keys'] for r in runs]}"),
    ]
    passed = True
    for name, ok, now in checks:
        passed = passed and ok
        lines.append(f"- {'✅' if ok else '❌'} {name}：{now}")
    lines += ["", "> 一致率不参与上面的通过判定：D23 起它是参考项（徽章保持「已抽核」）。",
              "> 判定阈值散落在本文件与 bench_check.py，口径升级时两处必须同步改。"]
    if not ok_runs:
        lines += ["", f"> 轮数不足 {MIN_RUNS}：本轮只报区间，**不构成通过/不通过的结论**。"]
        passed = False
    report = "\n".join(lines) + "\n"
    print(report)
    out = Path(argv[0]).parent / "summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"[✓] 聚合报告已写 {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))