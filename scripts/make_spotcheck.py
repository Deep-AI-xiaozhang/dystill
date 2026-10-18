#!/usr/bin/env python
"""生成「人工抽检表」（单文件 HTML，本地看，不入库）。

用法：
    python scripts/make_spotcheck.py <记录目录> [--csv bench/claims.csv] [--out bench/抽检表.html]

为什么要有这一步：bench/claims.csv 里 evidence 字段带换行，用表格软件打开会很乱；
而验收要求「人工抽检」——得让人在一屏里同时看到：笔记原句、我的标注与依据、引擎判定与理由、
以及**原素材原文（带时间戳）**。抽检的价值就在于能当场去回看原片核对，所以素材原文必须同屏。

它不调用模型、不花 token，纯读 CSV + check.json + 素材，随时可重生成。
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_workshop.core.artifacts import read_if_valid
from distill_workshop.core.check.evidence_retriever import SourceIndex

V_MARK = {"ok": "✅ 一致", "mismatch": "❌ 不一致", "doubt": "⚠️ 存疑"}
V_CLASS = {"ok": "ok", "mismatch": "bad", "doubt": "warn"}


def load_engine(rd: Path) -> dict[str, dict]:
    p = rd / "check.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for c in data.get("claims") or []:
        for k in (str(c.get("text") or "").strip(), str(c.get("original") or "").strip()):
            if k:
                out.setdefault(k, c)
    return out


def pick_sample(rows: list[dict]) -> list[str]:
    """有目的地挑 10 条：按"最可能出错的地方"排优先级，不随机。

    顺序即优先级：
    ① 潜在假阳性（我判非 ok、引擎判 ok）——产品命门，出现即 P0
    ② 潜在假 ❌（我判 ok、引擎判 mismatch）——**这一类的错可能在人工标注那边**，
       复核它比复核"引擎漏判"更有价值：假 ❌ 的伤害是让人不再相信 ❌ 标记
    ③ 我判 ❌ 的（双向都能验）
    ④ 补一条双方一致的 ok 当基线
    """
    fp = [r for r in rows if r["human_verdict"] != "ok" and r["engine_verdict"] == "ok"]
    fake_bad = [r for r in rows if r["human_verdict"] == "ok" and r["engine_verdict"] == "mismatch"]
    mine_bad = [r for r in rows if r["human_verdict"] == "mismatch"]
    eng_bad = [r for r in rows if r["engine_verdict"] == "mismatch"]
    base = [r for r in rows if r["human_verdict"] == "ok" and r["engine_verdict"] == "ok"]
    picked, seen = [], set()
    for group in (fp, fake_bad, mine_bad, eng_bad, base[:1]):
        for r in group:
            if r["id"] not in seen and len(picked) < 10:
                picked.append(r["id"])
                seen.add(r["id"])
    return picked


def render(rows: list[dict], sample: list[str], key: str, mode: str) -> str:
    todo = [r for r in rows if r["id"] in sample]
    rest = [r for r in rows if r["id"] not in sample]
    n_real = sum(1 for r in rows if r["human_verdict"] == "ok")
    agree = sum(1 for r in rows if r["human_verdict"] == r["engine_verdict"])
    fp = sum(1 for r in rows if r["human_verdict"] != "ok" and r["engine_verdict"] == "ok")

    def card(r: dict, highlight: bool) -> str:
        h, e = r["human_verdict"], r["engine_verdict"]
        cls = "card hl" if highlight else "card"
        return f"""
<section class="{cls}">
  <div class="head">
    <span class="cid">{r['id']}</span>
    <span class="sec">{html.escape(r['section'])}</span>
    <span class="pill {V_CLASS.get(h, '')}">我的标注：{V_MARK.get(h, h)}</span>
    <span class="pill {V_CLASS.get(e, '')}">引擎：{V_MARK.get(e, e)}</span>
  </div>
  <p class="claim">{html.escape(r['claim'])}</p>
  <dl>
    <dt>我的依据</dt><dd>{html.escape(r['human_note'] or '—')}</dd>
    <dt>引擎理由</dt><dd>{html.escape(r['engine_reason'] or '—')}</dd>
    <dt>原素材原文</dt><dd class="src">{r['source_html']}</dd>
  </dl>
  <label class="chk"><input type="checkbox"> 我核对过这条</label>
</section>"""

    def table(rs: list[dict]) -> str:
        body = "\n".join(
            f"<tr><td>{r['id']}</td><td>{html.escape(r['claim'][:90])}</td>"
            f"<td class='{V_CLASS.get(r['human_verdict'], '')}'>{V_MARK.get(r['human_verdict'], r['human_verdict'])}</td>"
            f"<td class='{V_CLASS.get(r['engine_verdict'], '')}'>{V_MARK.get(r['engine_verdict'], r['engine_verdict'])}</td></tr>"
            for r in rs)
        return ("<table><thead><tr><th>编号</th><th>声明</th><th>我的标注</th><th>引擎</th></tr></thead>"
                f"<tbody>{body}</tbody></table>")

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>核对基准 · 人工抽检表</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#F7F6F2;color:#2C2C2A;
      margin:0;padding:28px 22px 60px;line-height:1.65}}
 .wrap{{max-width:920px;margin:0 auto}}
 h1{{font-size:20px;font-weight:600;margin:0 0 6px}}
 h2{{font-size:15px;font-weight:600;margin:30px 0 10px;padding-bottom:6px;border-bottom:1px solid #E2DFD6}}
 .meta{{font-size:13px;color:#5F5E5A;margin:0 0 18px}}
 .kpi{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 4px}}
 .kpi div{{background:#fff;border:1px solid #E2DFD6;border-radius:10px;padding:10px 14px;font-size:13px}}
 .kpi b{{display:block;font-size:19px;font-weight:600;margin-top:2px}}
 .card{{background:#fff;border:1px solid #E2DFD6;border-radius:12px;padding:14px 16px;margin:12px 0}}
 .card.hl{{border-color:#BA7517;box-shadow:0 0 0 3px #FAEEDA inset}}
 .head{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px;margin-bottom:8px}}
 .cid{{font-weight:600;font-size:13px}}
 .sec{{color:#888780}}
 .pill{{border-radius:99px;padding:2px 9px;font-size:12px;border:1px solid #D3D1C7;background:#F1EFE8}}
 .pill.ok{{background:#E1F5EE;border-color:#5DCAA5;color:#085041}}
 .pill.bad{{background:#FCEBEB;border-color:#F09595;color:#791F1F}}
 .pill.warn{{background:#FAEEDA;border-color:#EF9F27;color:#633806}}
 .claim{{font-size:14px;margin:6px 0 10px;padding-left:10px;border-left:3px solid #D3D1C7}}
 dl{{margin:0;font-size:13px}}
 dt{{font-weight:600;color:#5F5E5A;margin-top:8px}}
 dd{{margin:2px 0 0}}
 dd.src{{font-size:12.5px;color:#444441;background:#FBFAF6;border:1px solid #EFEDE6;border-radius:8px;
        padding:8px 10px;white-space:pre-wrap}}
 .chk{{display:inline-block;margin-top:12px;font-size:12.5px;color:#5F5E5A}}
 table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;
        border:1px solid #E2DFD6;border-radius:10px;overflow:hidden}}
 th,td{{padding:7px 9px;border-bottom:1px solid #EFEDE6;text-align:left;vertical-align:top}}
 th{{background:#F1EFE8;font-weight:600;font-size:12.5px}}
 td.ok{{color:#085041}} td.bad{{color:#791F1F}} td.warn{{color:#633806}}
 .note{{background:#E6F1FB;border:1px solid #85B7EF;border-radius:10px;padding:12px 14px;font-size:13px;margin:16px 0}}
 code{{background:#F1EFE8;border-radius:4px;padding:1px 5px;font-size:12.5px}}
</style></head><body><div class="wrap">
<h1>核对基准 · 人工抽检表</h1>
<p class="meta">记录 {html.escape(key)} ｜ 模式 {html.escape(mode)} ｜ 共 {len(rows)} 条声明
｜ 标注者：AI（<b>需要你抽检才算数</b>）</p>

<div class="kpi">
  <div>已标注并配对<b>{len(rows)} 条</b></div>
  <div>人工口径一致率<b>{agree / len(rows):.1%}</b></div>
  <div>假阳性（我判非 ok / 引擎判 ok）<b>{fp} 条</b></div>
  <div>我的 ok 条数<b>{n_real}</b></div>
</div>

<div class="note">
  <b>怎么抽检（约 10 分钟）</b>：下面 10 条是按「最容易出错的地方」挑出来的——先看我和引擎判定不一致的，
  再看我判 ❌ 的。每条都贴了<b>原素材原文（带时间戳）</b>，你可以直接回原片核对。
  若有不同意见：把 <code>bench/claims.csv</code> 里那行的 <code>human_verdict</code>
  改成 <code>ok / mismatch / doubt</code>，再跑一次
  <code>python scripts/bench_check.py bench/claims.csv --from-json &lt;check.json&gt;</code> 就会重算准确率。
  （10 条里如有 ≥1 条我标错，一致率与假阳性都要以你改后的结果为准。）
</div>

<h2>建议抽检的 10 条</h2>
{''.join(card(r, True) for r in todo)}

<h2>其余 {len(rest)} 条（可选看）</h2>
{table(rest)}
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成人工抽检表（单文件 HTML）")
    ap.add_argument("record_dir")
    ap.add_argument("--csv", default="bench/claims.csv")
    ap.add_argument("--out", default="bench/抽检表.html")
    args = ap.parse_args()

    rd = Path(args.record_dir)
    engine = load_engine(rd)
    transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
    vlm = read_if_valid(rd / "vlm.txt", "vlm") or ""
    idx = SourceIndex.build(transcript, vlm, rd / "frames")
    try:
        key = json.loads((rd / "meta.json").read_text(encoding="utf-8")).get("key") or rd.name
    except Exception:  # noqa: BLE001
        mode = ""
    try:
        mode = json.loads((rd / "check.json").read_text(encoding="utf-8")).get("mode", "")
    except Exception:  # noqa: BLE001
        mode = ""

    rows = []
    with Path(args.csv).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            e = engine.get(r["claim"].strip()) or {}
            _, ev = idx.retrieve(r["claim"], top_k=2)
            src = "\n".join(f"[{e2.ts or '画面'}] {e2.text[:300]}" for e2 in ev) or "（未回查到原文片段）"
            rows.append({**r, "engine_verdict": e.get("verdict", ""),
                         "engine_reason": e.get("reason", ""), "source_html": html.escape(src)})

    sample = pick_sample(rows)
    out = Path(args.out)
    out.write_text(render(rows, sample, key, mode), encoding="utf-8")
    print(f"[OK] 已生成 {out}")
    print("建议抽检：", "、".join(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
