#!/usr/bin/env python
"""生成「浏览器标注工作表」：一份自包含 HTML，对着原片时间戳逐条判 ok/mismatch/doubt，
点右上角导出就下载填好的 CSV，直接喂给 bench_check.py。

为什么不用 Excel 填 CSV：evidence 字段带换行，Excel 里一坨很难读；而 LABELING.md 要求
"回看原片 + 盲标（先不看引擎结果）"。本表只给【声明 + 栏目 + 时间戳 + 参考证据】，
**绝不显示引擎判定**（避免抄作业），证据明确标注"只作参考，真值以原片为准"。

300 条量级的人工会死在两个地方，所以本表内建三件工具（2026-10-04 留出集实测）：
① **点时间戳直接跳原片**——标注是「回片」动作，掐表找位置每条要拖几十秒进度条，
   跳转后每条降到「点一下 + 看几秒」；② **localStorage 自动草稿**——中途刷新/误关不丢进度，
   且不写回 CSV（真值仍以人导出的文件为准，避免半成品被当结果）；③ **j/k 或 ↑/↓ 跳条**
   + `1/2/3` 判 ok/mismatch/doubt。

用法：
    python scripts/make_label_sheet.py            # 为所有 bench/claims_aweme_*.csv 各生成一份 .html
    python scripts/make_label_sheet.py <csv> ...  # 指定
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path

# 工具住在 scripts/，标注数据住在 bench/（后者 gitignore：含笔记原文=个人内容）
BENCH = Path(__file__).resolve().parents[1] / "bench"
REPO = Path(__file__).resolve().parents[1]

RUBRIC = [
    ("ok 一致", "原视频里找得到出处：话说过、字亮过、数字对得上——哪怕换了说法。"),
    ("mismatch 不一致", "与原片【正面矛盾】：数字不符 / 说法相反 / 评级结论对调 / 没做成做成。"),
    ("doubt 存疑", "素材不足以确认，或『片段里没找到』（≠原片没有），或证据讲的是另一款工具，或作者推论被升格成『对外提供/变现』。"),
]
ASR_RULES = [
    "同音/近音按音近：千万=通义千问、质普=ZCode、codes=Codex、occa=Orca。",
    "数字先归一再比：三点八↔3.8、两千↔2000；只有数值本身不符才 mismatch。",
    "专名以画面为准：转写没听清的，看抽帧里以文字出现过的。",
    "栏目名是上下文：『测试任务：…』说的是要做的事，不是已发生的事。",
    "拿不准 → 标 doubt 并在理由写『待复核』，绝不硬猜。",
]


def esc(s: str) -> str:
    return html.escape(s or "")


def ts_seconds(ts: str) -> float | None:
    """「01:12」/「1:02:03」→ 秒。解析不了就返回 None（表里降级成纯文本，不给假跳转）。"""
    parts = (ts or "").strip().split(":")
    if not parts or not all(p.isdigit() for p in parts) or len(parts) > 3:
        return None
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, s = nums
    return h * 3600 + m * 60 + s


def video_map() -> dict[str, str]:
    """aweme key → 本地 mp4 的 file:// URL。

    标注要「回看原片」，而原片就在 cache/dl/ 下（跑过 distill 就一定在）。
    只给 file:// 链接、不内嵌视频：内嵌会把 4 个视频的体积塞进 HTML，浏览器直接卡死。
    """
    dl = REPO / "cache" / "dl"
    if not dl.is_dir():
        return {}
    out: dict[str, str] = {}
    for mp4 in dl.glob("*.mp4"):
        # 缓存文件名结尾是 aweme id（如 ..._7692458902478245129.mp4）
        m = re.search(r"(\d{15,19})$", mp4.stem)
        if m:
            out["aweme_" + m.group(1)] = mp4.resolve().as_uri()
    return out


def render_rows(rows: list[dict], vids: dict[str, str]) -> str:
    blocks = []
    for r in rows:
        # 行唯一键必须带视频：id 在不同视频间撞号（各视频独立从 B001 编起），
        # 单用 id 会让 radio 组/草稿/导出串行（2026-10-05 用户实测踩到）
        rid = esc(f'{r["id"]}@{r["key"].replace("aweme_", "")}')
        claim = esc(r["claim"])
        section = esc(r["section"])
        kind = esc(r.get("kind", ""))
        ts_raw = (r.get("evidence_ts") or "").strip()
        ts = esc(ts_raw) or "—"
        ev_raw = (r.get("evidence") or "").strip()
        ev = esc(ev_raw) if ev_raw else "（无字面证据）"
        # 时间戳可解析 + 本地有原片 ⇒ 给一个能直接跳过去的按钮；否则降级成纯文本，
        # 绝不给一个点了没反应的假链接（标注时才发现更浪费时间）。
        secs = ts_seconds(ts_raw)
        url = vids.get(r.get("key", ""))
        if secs is not None and url:
            ts_html = (f'<a class="jump" href="{esc(url)}#t={secs:g}" target="_blank" '
                       f'title="在新标签页打开原片并跳到 {esc(ts_raw)}">▶ 原片 {ts}</a>')
        else:
            ts_html = f'<span class="ts">原片 <b>{ts}</b> 起</span>'
        # 预载：CSV 里已有 human_verdict/human_note 的（如 AI 草稿），渲染成已选中状态，
        # 复核者逐条确认或改判即可——LABELING 铁律一：草稿≠真值，过目才算。
        cur = (r.get("human_verdict") or "").strip().lower()
        note = esc((r.get("human_note") or "").strip())
        radios = "".join(
            f'<label><input type="radio" name="v-{rid}" value="{v}"'
            f'{" checked" if v == cur else ""}><span>{lbl}</span></label>'
            for v, lbl in (("ok", "✅ 一致"), ("mismatch", "❌ 不一致"), ("doubt", "⚠️ 存疑")))
        blocks.append(f"""
<div class="row" data-id="{rid}">
  <div class="rh">
    <span class="cid">{rid}</span>
    <span class="sec">{section}</span>
    <span class="kind">{kind}</span>
    {ts_html}
    {"<span class='kind draft'>有草稿</span>" if cur else ""}
  </div>
  <p class="claim">{claim}</p>
  <details class="ev"><summary>参考证据（只作参考，真值以原片为准）</summary><pre>{ev}</pre></details>
  <div class="verdict" role="group">
    {radios}
    <input class="note" type="text" value="{note}" placeholder="理由（非 ok 必填，回片看到什么）">
  </div>
</div>""")
    return "\n".join(blocks)


def build_html(csv_path: Path) -> str:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames or []
        rows = list(rdr)
    key = rows[0]["key"] if rows else csv_path.stem
    vids = video_map()
    data = json.dumps({"fields": fields, "rows": rows}, ensure_ascii=False)

    rub = "".join(f"<li><b>{esc(a)}</b>：{esc(b)}</li>" for a, b in RUBRIC)
    asr = "".join(f"<li>{esc(x)}</li>" for x in ASR_RULES)
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>标注工作表 · {esc(key)}</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#F7F6F2;color:#2C2C2A;margin:0;line-height:1.6}}
 header{{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #E2DFD6;padding:12px 22px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
 header h1{{font-size:16px;margin:0;font-weight:600}}
 .prog{{font-size:13px;color:#5F5E5A}} .prog b{{color:#085041;font-size:15px}}
 button{{background:#1D6F42;color:#fff;border:0;border-radius:8px;padding:8px 16px;font-size:14px;cursor:pointer}}
 button.sec{{background:#F1EFE8;color:#2C2C2A;border:1px solid #D3D1C7}}
 .wrap{{max-width:900px;margin:0 auto;padding:20px 22px 80px}}
 .rule{{background:#fff;border:1px solid #E2DFD6;border-radius:10px;padding:12px 16px;font-size:13px;margin-bottom:16px}}
 .rule h2{{font-size:13px;margin:0 0 6px;color:#888780;font-weight:600}}
 .rule ul{{margin:0 0 10px;padding-left:20px}} .rule li{{margin:2px 0}}
 .row{{background:#fff;border:1px solid #E2DFD6;border-radius:12px;padding:14px 16px;margin:12px 0}}
 .row.done{{border-color:#5DCAA5;box-shadow:0 0 0 2px #E1F5EE inset}}
 .rh{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px;color:#888780;margin-bottom:6px}}
 .cid{{font-weight:700;color:#2C2C2A;font-size:13px}} .kind{{background:#F1EFE8;border-radius:6px;padding:1px 7px}}
 .kind.draft{{background:#FAEEDA;border-color:#EF9F27;color:#633806}}
 .ts b{{color:#1D6F42;font-family:monospace}}
 .jump{{color:#1D6F42;font-weight:600;text-decoration:none;border:1px solid #B7E0C9;
        border-radius:8px;padding:2px 9px;font-size:12px;background:#EAF7F0}}
 .jump:hover{{background:#D6EFE2}}
 .row.cur{{box-shadow:0 0 0 3px #9AD3B4}}
 .claim{{font-size:15px;margin:4px 0 10px;padding-left:10px;border-left:3px solid #D3D1C7}}
 details.ev summary{{cursor:pointer;font-size:12px;color:#888780}}
 details.ev pre{{white-space:pre-wrap;font-size:12px;color:#444441;background:#FBFAF6;border:1px solid #EFEDE6;border-radius:8px;padding:8px 10px;margin:6px 0 0;max-height:280px;overflow:auto}}
 .verdict{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}}
 .verdict label input{{display:none}}
 .verdict label span{{border:1px solid #D3D1C7;border-radius:99px;padding:4px 12px;font-size:13px;cursor:pointer;background:#F1EFE8}}
 label:has(input[value=ok]:checked) span{{background:#E1F5EE;border-color:#5DCAA5;color:#085041;font-weight:600}}
 label:has(input[value=mismatch]:checked) span{{background:#FCEBEB;border-color:#F09595;color:#791F1F;font-weight:600}}
 label:has(input[value=doubt]:checked) span{{background:#FAEEDA;border-color:#EF9F27;color:#633806;font-weight:600}}
 .note{{flex:1;min-width:200px;border:1px solid #D3D1C7;border-radius:8px;padding:6px 10px;font-size:13px}}
 .hint{{font-size:12px;color:#888780;margin:2px 22px}}
</style></head><body>
<header>
  <h1>标注工作表 · {esc(key)}</h1>
  <span class="prog">已标 <b id="done">0</b> / {len(rows)}（验收需全库累计 ≥50）</span>
  <span style="flex:1"></span>
  <button id="exp">导出填好的 CSV</button>
</header>
<div class="hint">快捷键：<b>j / ↓</b> 下一条 · <b>k / ↑</b> 上一条 · <b>1</b> 一致 · <b>2</b> 不一致 · <b>3</b> 存疑 · <b>Shift+J</b> 打开当前条原片。进度自动存草稿（只存浏览器，不改 CSV）。</div>
<div class="wrap">
  <div class="rule">
    <h2>三值判据（拿不准一律 doubt，别硬猜）</h2><ul>{rub}</ul>
    <h2>中文口播规则</h2><ul>{asr}</ul>
  </div>
  {render_rows(rows, vids)}
</div>
<script id="data" type="application/json">{data}</script>
<script>
const {{fields, rows}} = JSON.parse(document.getElementById('data').textContent);
const doneEl = document.getElementById('done');
const SKEY='dystill_label_draft_' + {json.dumps(csv_path.name)};
// 草稿只在浏览器 localStorage，不回写 CSV：真值仍以人点「导出」的文件为准，
// 免得标到一半的半成品被当成结果误用。
function rowOf(id){{ return document.querySelector('.row[data-id="'+id+'"]'); }}
function save(){{
  const d={{}};
  rows.forEach(r=>{{
    const el=rowOf(uk(r)); if(!el) return;
    const v=el.querySelector('input[type=radio]:checked');
    const n=el.querySelector('.note').value.trim();
    if(v || n) d[uk(r)]=[v?v.value:'', n];
  }});
  try{{ localStorage.setItem(SKEY, JSON.stringify(d)); }}catch(e){{}}
}}
function load(){{
  let d={{}}; try{{ d=JSON.parse(localStorage.getItem(SKEY)||'{{}}'); }}catch(e){{ return; }}
  Object.entries(d).forEach(([id,[v,n]])=>{{
    const el=rowOf(id); if(!el) return;
    if(v){{ const r=el.querySelector('input[value="'+v+'"]'); if(r) r.checked=true; }}
    if(n) el.querySelector('.note').value=n;
  }});
}}
function refresh(){{
  let n=0;
  document.querySelectorAll('.row').forEach(r=>{{
    const v=r.querySelector('input[type=radio]:checked');
    r.classList.toggle('done', !!v); if(v) n++;
  }});
  doneEl.textContent=n;
}}
document.addEventListener('change', e=>{{ refresh(); save(); }});
document.addEventListener('input', e=>{{ if(e.target.classList.contains('note')) save(); }});
// 键盘：j/k 或 ↓/↑ 跳条，1/2/3 判 ok/mismatch/doubt。300 条逐条点鼠标太慢。
let cur=0;
const rowEls=()=>Array.from(document.querySelectorAll('.row'));
function goto(i){{
  const els=rowEls(); if(!els.length) return;
  cur=Math.max(0,Math.min(els.length-1,i));
  els.forEach((e,j)=>e.classList.toggle('cur', j===cur));
  els[cur].scrollIntoView({{block:'center'}});
}}
document.addEventListener('keydown', e=>{{
  if(e.target.tagName==='TEXTAREA') return;
  const k=e.key;
  if(k==='j'||k==='ArrowDown'){{ goto(cur+1); e.preventDefault(); }}
  else if(k==='k'||k==='ArrowUp'){{ goto(cur-1); e.preventDefault(); }}
  else if(k==='1'||k==='2'||k==='3'){{
    const el=rowEls()[cur]; if(!el) return;
    const r=el.querySelector('input[value="'+['ok','mismatch','doubt'][+k-1]+'"]');
    if(r){{ r.checked=true; refresh(); save(); }}
    e.preventDefault();
  }} else if(k==='J'||(e.shiftKey && (k==='j'||k==='J'))){{
    const el=rowEls()[cur]; if(el){{ const a=el.querySelector('.jump'); if(a) a.click(); }}
  }}
}});
function q(s){{ return /[",\\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; }}
document.getElementById('exp').onclick = ()=>{{
  const out=[]; rows.forEach(r=>{{
    const row=rowOf(uk(r));
    const v=row.querySelector('input[type=radio]:checked');
    r.human_verdict = v? v.value : '';
    r.human_note = row.querySelector('.note').value.trim();
    out.push(fields.map(f=>q(r[f] ?? '')).join(','));
  }});
  const csv='\\ufeff'+fields.join(',')+'\\n'+out.join('\\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));
  a.download='{esc(csv_path.name)}'; a.click();
}};
load(); refresh(); goto(0);
</script></body></html>"""


def main(argv: list[str]) -> int:
    targets = [Path(p) for p in argv] if argv else sorted(BENCH.glob("claims_aweme_*.csv"))
    if not targets:
        print("[!] 没找到 bench/claims_aweme_*.csv")
        return 2
    for csv_path in targets:
        html_out = csv_path.with_suffix(".标注表.html")
        html_out.write_text(build_html(csv_path), encoding="utf-8")
        print(f"[✓] {html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
