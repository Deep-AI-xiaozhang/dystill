"""L4 核对报告生成（核对四层第四层 + 用户感知设计）。

三种交付形态，各自服务一个场景：
1. `check.json` —— note_check 记录，给下游/后续版本用（唯一机器真相源）
2. `check.md`   —— 可折叠逐条报告，方便搜索、复制和进 Git
3. `check.html` —— 零依赖本地核对页，播放器 + 时间戳按钮可直接跳原片
4. 笔记顶部徽章 —— 给"扫一眼就走"的用户：✅ 已抽核 12/15 · ⚠️ 2 条存疑
   （措辞用「抽核」不用「核对」：判定一致率尚未过验收线 + 长笔记只核抽样，
   「已核对」的隐含承诺高于实际能力——降级理由与回改条件见 DECISIONS D14）

徽章写入用 HTML 注释包裹，**幂等**：重复核对同一篇笔记只会替换标记之间的内容，
不会越插越多（"重复追加"事故的结构性规避）。

诚实边界写进报告头，不藏：核对只保证"没编造"，不保证"概括得准"；核对基于转写，
转写本身出错不在核对范围内。这句必须出现在产物里，否则用户会拿"概括不到位"来骂。
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

from .evidence_retriever import (
    _number_in_source,
    coverage,
    hard_facts,
    num_variants,
    tokenize,
)
from .models import (
    VERDICT_DOUBT,
    VERDICT_LABEL,
    VERDICT_MARK,
    VERDICT_MISMATCH,
    VERDICT_OK,
    CheckResult,
)

BEGIN_MARK = "<!-- dystill-check:begin -->"
END_MARK = "<!-- dystill-check:end -->"

# fix 的自动改写保险（见 apply_fixes）：建议式措辞不是改写文本，写进正文=篡改笔记。
# 原正则 `^建议` 只认句首、"回看原片"只认这一种说法——「月费应为 3 元，建议回看」这类
# 「指引 + 改写」混写会绕过去（2026-10-04 复查）。
_ADVICE_FIX_RE = re.compile(r"建议|回看原片|请核实|请确认")

# fix 里没有硬事实（数字/英文专名）可供校验时，靠与证据的字面重合度兜底的下限。
# 取值偏低（0.5）：中文改写能对上半数特征就算有支撑，否则宁可不改。
_FIX_MIN_COVERAGE = 0.5

BOUNDARY = ("核对边界：只保证「笔记里的话在原视频里找得到出处」，不保证「概括得准」，"
            "不评价观点对错；核对基于转写文本，转写本身出错不在核对范围内。")


def badge(result: CheckResult) -> str:
    """核对徽章：✅ 已抽核 N/M ｜ ⚠️ K 条存疑 ｜ ⏸️ 未判定 ｜ ❌ J 条不一致。

    「已抽核」是刻意选择，不是笔误——见 D14。回改「已核对」的唯一路径是验收口径 v1 达标。
    ⏸️ 单独列"调用失败未判定"：技术性失败也记 ⚠️，但和"真存疑"混在一起会让
    用户误以为内容存疑、还把数字撑高（抽检时的观察）。
    """
    ok, doubt, bad = (result.count(VERDICT_OK), result.count(VERDICT_DOUBT),
                      result.count(VERDICT_MISMATCH))
    failed = sum(1 for c in result.claims if c.failed)
    judged = result.total - failed          # 分母只算真正判定过的
    real_doubt = doubt - failed             # 真存疑 = 存疑数 - 调用失败数
    parts = [f"✅ 已抽核 {ok}/{judged}"]
    if real_doubt:
        parts.append(f"⚠️ {real_doubt} 条存疑")
    if failed:
        parts.append(f"⏸️ {failed} 条未判定（调用失败，可 --retry-failed）")
    if bad:
        parts.append(f"❌ {bad} 条不一致")
    return " · ".join(parts)


def build_json(result: CheckResult) -> dict:
    """数据模型。"""
    return {
        "key": result.key,
        "mode": result.mode,
        "badge": badge(result),
        "status": result.status,
        "created_at": result.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": result.provider,
        "counts": {
            "total": result.total,
            "ok": result.count(VERDICT_OK),
            "mismatch": result.count(VERDICT_MISMATCH),
            "doubt": result.count(VERDICT_DOUBT),
            "failed": sum(1 for c in result.claims if c.failed),
        },
        "claims": [c.to_dict() for c in result.claims],
        "skipped": result.skipped,
        "boundary": BOUNDARY,
    }


def _claim_block(c) -> str:
    mark = VERDICT_MARK.get(c.verdict, "")
    label = VERDICT_LABEL.get(c.verdict, "未判定")
    head = f"**{c.cid} {mark} {label}** ｜ {c.section} ｜ 覆盖度 {c.score:.2f}"
    body = [head, "", f"- 笔记声明：{c.text}"]
    if c.original and c.original != c.text:
        body.append(f"- 笔记原句：{c.original}")
    if c.reason:
        body.append(f"- 判定依据：{c.reason}")
    if c.miss_numbers or c.miss_names or c.miss_biz:
        miss = []
        if c.miss_numbers:
            miss.append(f"数字 {'、'.join(c.miss_numbers)}")
        if c.miss_names:
            miss.append(f"专名 {'、'.join(c.miss_names)}")
        if c.miss_biz:
            miss.append(f"对外/商业措辞 {'、'.join(c.miss_biz)}")
        body.append(f"- 原素材中查不到：{'；'.join(miss)}")
    if c.evidence:
        body.append("- 原视频证据：")
        for e in c.evidence[:3]:
            tag = "转写" if e.source == "transcript" else "画面"
            pos = f"`{e.ts}` " if e.ts else ""
            body.append(f"  - {pos}【{tag}】{e.text[:200]}")
    else:
        body.append("- 原视频证据：（未回查到字面证据）")
    if c.fix:
        body.append(f"- 修正建议：{c.fix}")
    return "<details>\n<summary>" + head + "</summary>\n\n" + "\n".join(body[2:]) + "\n</details>"


def render_markdown(result: CheckResult) -> str:
    ok, doubt, bad = (result.count(VERDICT_OK), result.count(VERDICT_DOUBT),
                      result.count(VERDICT_MISMATCH))
    failed = sum(1 for c in result.claims if c.failed)
    lines = [
        f"# 核对报告 · {result.key}",
        "",
        f"> {badge(result)} ｜ 模式：{result.mode} ｜ 生成：{result.created_at}",
        f"> {BOUNDARY}",
        "",
        "> **判定结果说明**：❌ 不是\"已确认错误\"，而是\"建议人工复查\"（实测假阳性率 40-50%）。",
        "> 常见误判：把上文的评价算到下文主体上、把\"没明说\"当\"明确否定\"、证据内部矛盾时选错边。",
        ("> **正确用法**：当\"快速定位可疑片段\"工具；打开同目录 `check.html`，"
         "看到 ❌ 后点时间戳回原片确认，不要盲目相信。"),
        "",
        "## 汇总",
        "",
        "| 判定 | 条数 | 含义 |",
        "|---|---|---|",
        f"| ✅ 一致 | {ok} | 原视频里找到出处 |",
        f"| ❌ 不一致 | {bad} | 原视频里没有，或说法/数字有出入 |",
        f"| ⚠️ 存疑 | {doubt} | 证据不足，保留但标注，建议回看原片 |",
        f"| ⏸️ 未判定 | {failed} | 模型调用失败（限流/网络），可 --retry-failed 重判 |",
        f"| 合计 | {result.total} | 原子声明数 |",
        "",
        "## 逐条明细",
        "",
    ]
    order = {VERDICT_MISMATCH: 0, VERDICT_DOUBT: 1, VERDICT_OK: 2}
    for c in sorted(result.claims, key=lambda x: (order.get(x.verdict, 3), x.cid)):
        lines.append(_claim_block(c))
    if result.skipped:
        lines += ["", "## 主动跳过的句子（透明化：没核什么，也写出来）", ""]
        lines += [f"- {s}" for s in result.skipped]
    return "\n".join(lines) + "\n"


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --ink: #1f2926;
  --muted: #68736e;
  --paper: #f4efe4;
  --card: #fffaf0;
  --line: #d8cdb9;
  --accent: #b5452e;
  --ok: #2c6953;
  --doubt: #9b6a19;
  --bad: #a32d28;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 15% 0%, rgba(181,69,46,.10), transparent 28rem),
    linear-gradient(135deg, rgba(255,255,255,.45), transparent 45%),
    var(--paper);
  font-family: "Microsoft YaHei UI", "Noto Sans SC", sans-serif;
}
.shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 38px 0 72px; }
header { display: grid; grid-template-columns: 1fr auto; gap: 28px; align-items: end;
  border-bottom: 3px solid var(--ink); padding-bottom: 22px; }
header > * { min-width: 0; }
.eyebrow { color: var(--accent); font: 700 12px/1.2 Georgia, serif; letter-spacing: .16em;
  text-transform: uppercase; }
h1 { margin: 8px 0 0; max-width: 900px; font: 700 clamp(28px, 3.7vw, 48px)/1.08 Georgia, "STSong", serif;
  text-wrap: balance; }
.meta { color: var(--muted); line-height: 1.7; text-align: right; font-size: 13px;
  overflow-wrap: anywhere; }
.summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }
.stat { background: rgba(255,250,240,.78); border: 1px solid var(--line); padding: 16px 18px; }
.stat strong { display: block; font: 700 30px/1 Georgia, serif; }
.stat span { display: block; margin-top: 7px; color: var(--muted); font-size: 13px; }
.workspace { display: grid; grid-template-columns: minmax(430px, .9fr) minmax(520px, 1.1fr); gap: 24px; }
.player-card { position: sticky; top: 16px; align-self: start; background: #15201d; color: #f7f1e7;
  padding: 12px; box-shadow: 0 18px 50px rgba(31,41,38,.18); }
video { display: block; width: 100%; max-height: 54vh; background: #000; }
.player-note { margin: 10px 4px 2px; color: #c9d1cd; font-size: 12px; line-height: 1.6; }
.missing { border: 1px dashed #80908a; padding: 22px; line-height: 1.7; color: #d9e0dc; }
.gallery-stage { position: relative; background: #0f1715; }
.gallery-stage img { display: block; width: 100%; height: min(54vh, 620px); object-fit: contain; background: #0f1715; }
.gallery-slide[hidden] { display: none; }
.gallery-bar { display: flex; justify-content: space-between; gap: 10px; align-items: center;
  padding: 9px 4px 3px; }
.gallery-bar button { background: #e9dfce; color: #1f2926; border: 0; }
.gallery-count { color: #d9e0dc; font: 700 13px/1 Georgia, serif; }
.gallery-thumbs { display: flex; gap: 6px; overflow-x: auto; padding: 9px 2px 2px; }
.gallery-thumbs button { flex: 0 0 54px; padding: 0; border: 2px solid transparent;
  background: transparent; }
.gallery-thumbs button.active { border-color: #d9b36b; }
.gallery-thumbs img { display: block; width: 50px; height: 68px; object-fit: cover; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 14px; }
button { border: 1px solid var(--line); background: var(--card); color: var(--ink); padding: 8px 12px;
  cursor: pointer; font: inherit; font-size: 13px; }
button:hover, button.active { border-color: var(--ink); background: var(--ink); color: #fff; }
.claim { background: rgba(255,250,240,.92); border: 1px solid var(--line); margin: 0 0 12px; padding: 18px; }
.claim[data-verdict="mismatch"] { border-left: 5px solid var(--bad); }
.claim[data-verdict="doubt"] { border-left: 5px solid var(--doubt); }
.claim[data-verdict="ok"] { border-left: 5px solid var(--ok); }
.claim-head { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
.cid { font: 700 13px/1 Georgia, serif; color: var(--muted); }
.verdict { font-weight: 700; }
.section { margin: 8px 0 14px; color: var(--muted); font-size: 12px; }
.claim p { line-height: 1.75; margin: 8px 0; }
.reason { border-top: 1px solid var(--line); padding-top: 12px; color: #3d4944; }
.evidence { margin-top: 12px; background: rgba(31,41,38,.045); padding: 12px; }
.evidence-title { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.evidence-row { display: grid; grid-template-columns: auto 1fr; gap: 10px; margin: 10px 0; align-items: start; }
.seek { min-width: 66px; font: 700 12px/1 Georgia, serif; background: var(--ink); color: white;
  border: 0; padding: 7px 8px; }
.ev-text { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 13px; line-height: 1.65; color: #49534e; }
.empty { padding: 28px; text-align: center; color: var(--muted); border: 1px dashed var(--line); }
footer { border-top: 1px solid var(--line); margin-top: 34px; padding-top: 16px; color: var(--muted);
  font-size: 12px; line-height: 1.8; }
@media (max-width: 820px) {
  .shell { width: min(100% - 20px, 700px); padding-top: 22px; }
  header, .workspace { grid-template-columns: 1fr; }
  .meta { text-align: left; }
  .summary { grid-template-columns: repeat(2, 1fr); }
  .player-card { position: static; }
}
@media print {
  body { background: white; }
  .shell { width: 100%; padding: 0; }
  .player-card, .toolbar, button { display: none !important; }
  .workspace { display: block; }
  .claim { break-inside: avoid; box-shadow: none; }
}
</style>
</head>
<body>
<main class="shell">
  <header>
    <div>
      <div class="eyebrow">Dystill Evidence Review</div>
      <h1>__HEADING__</h1>
    </div>
    <div class="meta">__MODE__ 模式<br>__CREATED__<br>__KEY__</div>
  </header>
  <section class="summary">__SUMMARY__</section>
  <div class="workspace">
    <aside class="player-card">__PLAYER__</aside>
    <section>
      <div class="toolbar" aria-label="筛选判定结果">
        <button class="active" data-filter="all">全部 __TOTAL__</button>
        <button data-filter="mismatch">不一致 __MISMATCH__</button>
        <button data-filter="doubt">存疑 __DOUBT__</button>
        <button data-filter="ok">一致 __OK__</button>
      </div>
      <div id="claims">__CLAIMS__</div>
    </section>
  </div>
  <footer>__BOUNDARY__<br>❌ 不是“已确认错误”，而是建议回看原片的信号。时间戳基于原素材转写与画面定位。</footer>
</main>
<script>
(() => {
  const player = document.getElementById("player");
  const seek = (seconds) => {
    if (!player || !Number.isFinite(seconds)) return;
    player.currentTime = Math.max(0, seconds);
    player.play().catch(() => {});
    player.scrollIntoView({behavior: "smooth", block: "center"});
  };
  document.querySelectorAll("[data-seek]").forEach((button) => {
    button.addEventListener("click", () => seek(Number(button.dataset.seek)));
  });
  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-filter]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      const filter = button.dataset.filter;
      document.querySelectorAll(".claim").forEach((claim) => {
        claim.hidden = filter !== "all" && claim.dataset.verdict !== filter;
      });
    });
  });
  const gallerySlides = [...document.querySelectorAll(".gallery-slide")];
  const galleryThumbs = [...document.querySelectorAll("[data-gallery-thumb]")];
  const galleryCount = document.getElementById("gallery-count");
  let galleryIndex = 0;
  const showGallery = (index) => {
    if (!gallerySlides.length) return;
    galleryIndex = Math.max(0, Math.min(index, gallerySlides.length - 1));
    gallerySlides.forEach((slide, i) => { slide.hidden = i !== galleryIndex; });
    galleryThumbs.forEach((thumb, i) => thumb.classList.toggle("active", i === galleryIndex));
    if (galleryCount) galleryCount.textContent = `第 ${galleryIndex + 1} / ${gallerySlides.length} 张`;
  };
  document.querySelectorAll("[data-gallery-index]").forEach((button) => {
    button.addEventListener("click", () => {
      showGallery(Number(button.dataset.galleryIndex));
      document.getElementById("gallery")?.scrollIntoView({behavior: "smooth", block: "center"});
    });
  });
  galleryThumbs.forEach((button) => {
    button.addEventListener("click", () => showGallery(Number(button.dataset.galleryThumb)));
  });
  document.getElementById("gallery-prev")?.addEventListener("click", () => showGallery(galleryIndex - 1));
  document.getElementById("gallery-next")?.addEventListener("click", () => showGallery(galleryIndex + 1));
})();
</script>
</body>
</html>
"""


def _ts_seconds(ts: str) -> int | None:
    """把 01:23 / 1:02:03 转成秒；格式不对返回 None，绝不猜位置。"""
    try:
        parts = [int(part) for part in (ts or "").split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _media_for_html(record_dir: Path) -> tuple[str, str]:
    """返回 (媒体 URI, 问题说明)。只接受真实存在的本地媒体。"""
    try:
        meta = json.loads((Path(record_dir) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return "", "找不到 meta.json，无法定位原片。"
    if meta.get("media_type") == "gallery":
        return "", "图文帖没有视频时间轴；请回原图核对画面证据。"
    raw = str(meta.get("media") or "").strip()
    if not raw:
        return "", "记录里没有原片路径。"
    media = Path(raw)
    if not media.is_absolute():
        media = Path(record_dir) / media
    if not media.is_file():
        return "", f"原片已不在本机：{media}。请重新下载后再点时间戳回看。"
    return media.resolve().as_uri(), ""


def _gallery_for_html(record_dir: Path) -> list[str]:
    """读取图文帖 meta.json 里真实存在的图片，返回本地 file URI 列表。"""
    try:
        meta = json.loads((Path(record_dir) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return []
    if meta.get("media_type") != "gallery":
        return []
    out = []
    for raw in meta.get("images") or []:
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path(record_dir) / path
        if path.is_file():
            out.append(path.resolve().as_uri())
    return out


def _frame_index(text: str) -> int | None:
    match = re.search(r"第\s*(\d+)\s*(?:-|~|至)?\s*(?:\d+\s*)?张", text or "")
    return int(match.group(1)) - 1 if match else None


def render_html(result: CheckResult, record_dir: Path) -> str:
    """生成零依赖本地核对页；时间戳按钮直接让播放器跳到对应秒。"""
    escaped_key = html.escape(result.key)
    counts = {
        "ok": result.count(VERDICT_OK),
        "mismatch": result.count(VERDICT_MISMATCH),
        "doubt": result.count(VERDICT_DOUBT),
    }
    media_uri, media_problem = _media_for_html(record_dir)
    gallery_uris = _gallery_for_html(record_dir)
    if media_uri:
        player = (
            '<video id="player" controls preload="metadata">'
            f'<source src="{html.escape(media_uri, quote=True)}">'
            '你的浏览器不支持本地视频播放。</video>'
            '<div class="player-note">点下方证据的时间戳，可直接跳到原片对应位置。</div>'
        )
    elif gallery_uris:
        gallery_images = "".join(
            f'<img class="gallery-slide" data-gallery-slide="{i}" '
            f'src="{html.escape(uri, quote=True)}" alt="第 {i + 1} 张图"'
            f'{" hidden" if i else ""}>'
            for i, uri in enumerate(gallery_uris)
        )
        gallery_thumbs = "".join(
            f'<button class="gallery-thumb{" active" if i == 0 else ""}" '
            f'data-gallery-thumb="{i}" aria-label="查看第 {i + 1} 张图">'
            f'<img src="{html.escape(uri, quote=True)}" alt=""></button>'
            for i, uri in enumerate(gallery_uris)
        )
        player = (
            '<div class="gallery-stage" id="gallery">'
            f'{gallery_images}'
            '<div class="gallery-bar"><button id="gallery-prev">← 上一张</button>'
            f'<span class="gallery-count" id="gallery-count">第 1 / {len(gallery_uris)} 张</span>'
            '<button id="gallery-next">下一张 →</button></div></div>'
            f'<div class="gallery-thumbs">{gallery_thumbs}</div>'
            '<div class="player-note">这是图文帖的本地看图画廊；证据里的“第 N 张图”也可以点击查看。</div>'
        )
    else:
        player = f'<div class="missing">{html.escape(media_problem)}</div>'

    summary = "".join(
        f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in (
            (result.total, "原子声明"),
            (counts["ok"], "找到出处"),
            (counts["doubt"], "存疑待看"),
            (counts["mismatch"], "不一致信号"),
        )
    )

    order = {VERDICT_MISMATCH: 0, VERDICT_DOUBT: 1, VERDICT_OK: 2}
    cards = []
    for claim in sorted(result.claims, key=lambda c: (order.get(c.verdict, 3), c.cid)):
        mark = VERDICT_MARK.get(claim.verdict, "")
        label = VERDICT_LABEL.get(claim.verdict, "未判定")
        evidence_rows = []
        for evidence in claim.evidence[:3]:
            source = "转写" if evidence.source == "transcript" else "画面"
            evidence_text = html.escape(evidence.text[:800])
            seconds = _ts_seconds(evidence.ts)
            gallery_index = _frame_index(evidence.text) if gallery_uris else None
            if gallery_index is not None and 0 <= gallery_index < len(gallery_uris):
                position = (
                    f'<button class="seek" data-gallery-index="{gallery_index}" '
                    f'aria-label="查看第 {gallery_index + 1} 张图">'
                    f'第 {gallery_index + 1} 张</button>'
                )
            elif seconds is not None and media_uri:
                position = (
                    f'<button class="seek" data-seek="{seconds}" '
                    f'aria-label="跳到 {html.escape(evidence.ts, quote=True)}">'
                    f'{html.escape(evidence.ts)}</button>'
                )
            else:
                position = f'<span class="cid">{html.escape(evidence.ts or "无时间码")}</span>'
            evidence_rows.append(
                f'<div class="evidence-row">{position}'
                f'<div><strong>{source}</strong><div class="ev-text">{evidence_text}</div></div></div>'
            )
        if not evidence_rows:
            evidence_rows.append('<div class="ev-text">未回查到字面证据。</div>')
        reason = (f'<p class="reason"><strong>判定依据：</strong>{html.escape(claim.reason)}</p>'
                  if claim.reason else "")
        fix = (f'<p class="reason"><strong>修正建议：</strong>{html.escape(claim.fix)}</p>'
               if claim.fix else "")
        cards.append(
            f'<article class="claim" data-verdict="{html.escape(claim.verdict, quote=True)}">'
            f'<div class="claim-head"><span class="cid">{html.escape(claim.cid)}</span>'
            f'<span class="verdict">{mark} {html.escape(label)}</span></div>'
            f'<div class="section">{html.escape(claim.section)}</div>'
            f'<p>{html.escape(claim.text)}</p>'
            f'{reason}'
            f'<div class="evidence"><div class="evidence-title">原视频证据</div>'
            f'{"".join(evidence_rows)}</div>{fix}</article>'
        )
    if not cards:
        cards.append('<div class="empty">没有可核对的原子声明。</div>')

    replacements = {
        "__TITLE__": f"蒸馏核对 · {escaped_key}",
        "__HEADING__": "把每一句话，放回原片里听。",
        "__MODE__": html.escape(result.mode),
        "__CREATED__": html.escape(result.created_at),
        "__KEY__": escaped_key,
        "__SUMMARY__": summary,
        "__PLAYER__": player,
        "__TOTAL__": str(result.total),
        "__MISMATCH__": str(counts["mismatch"]),
        "__DOUBT__": str(counts["doubt"]),
        "__OK__": str(counts["ok"]),
        "__CLAIMS__": "".join(cards),
        "__BOUNDARY__": html.escape(BOUNDARY),
    }
    # 单次替换：证据文本来自模型/原素材，不能让它包含 __TOKEN__ 时误改后面的模板内容。
    return re.sub(r"__[A-Z]+__", lambda match: replacements.get(match.group(0), match.group(0)),
                  _HTML_TEMPLATE)


def write_reports(record_dir: Path, result: CheckResult) -> dict:
    """落盘 check.json / check.md / check.html，返回路径。"""
    rd = Path(record_dir)
    rd.mkdir(parents=True, exist_ok=True)
    data = build_json(result)
    (rd / "check.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (rd / "check.md").write_text(render_markdown(result), encoding="utf-8")
    (rd / "check.html").write_text(render_html(result, rd), encoding="utf-8")
    return {
        "json": str(rd / "check.json"),
        "markdown": str(rd / "check.md"),
        "html": str(rd / "check.html"),
    }


def inject_badge(note_text: str, result: CheckResult) -> str:
    """把徽章插到笔记 top（frontmatter 之后）；已存在则替换标记之间内容（幂等）。"""
    line = (f"{BEGIN_MARK}\n"
            f"> {badge(result)} ｜ 模式：{result.mode} ｜ {result.created_at}\n"
            f"> 「抽核」＝抽声明回原素材找出处；⚠️/未核条目不在保证范围，"
            f"逐条见 `check.html`（可点时间戳回原片）\n"
            f"{END_MARK}\n")
    if BEGIN_MARK in note_text and END_MARK in note_text:
        head, rest = note_text.split(BEGIN_MARK, 1)
        _, tail = rest.split(END_MARK, 1)
        return head.rstrip() + "\n" + line + tail.lstrip("\n")
    if note_text.startswith("---"):
        parts = note_text.split("---", 2)
        if len(parts) == 3:
            return f"---{parts[1]}---\n\n" + line + parts[2].lstrip("\n")
    return line + "\n" + note_text


def _fix_supported(c) -> bool:
    """fix 是否有证据支撑。

    主判据：fix 里的**硬事实**（数字/英文专名）必须全部能在证据里找到。
    为什么主看硬事实、不看整体覆盖度：fix 是"修正后的说法"，与证据本就大部分重合，
    字面覆盖度拦不住"月费 100 元改成月费 3 元"这种只换一个数字的幻觉——
    而硬事实校验能：3 不在证据里就一票否决（数字是 L2 已验证的最强信号）。
    中文数字不提取（ASR 场景修正建议极少用中文数字写事实，可接受盲区）。

    兜底（2026-10-04 复查补）：fix 里**一个硬事实都没有**时上面两个 all() 对空集恒为真，
    等于没设防（README 写的"三重保险"会名不副实）——退回字面覆盖度，证据几乎不支持就
    只报告不改写。
    """
    if not c.evidence:
        return False
    src = " ".join(e.text for e in c.evidence).lower()
    nums, names = hard_facts(c.fix or "")
    if not nums and not names:
        return coverage(tokenize(c.fix or ""), tokenize(src)) >= _FIX_MIN_COVERAGE
    return (all(any(_number_in_source(v, src) for v in num_variants(n)) for n in nums)
            and all(w in src for w in names))


def apply_fixes(note_text: str, result: CheckResult, on_step=None) -> tuple[str, list[str]]:
    """把 ❌ 不一致的句子按证据改回原素材说法（**默认关闭**，需显式开启）。

    只做"原句 → 修正"的精确替换，且**三重保险，缺一不动手**——
    静默改写会破坏「笔记 vs 原片」的可信度，改错了比不改更糟：
    1. 原句能在笔记里**逐字**找到（模糊替换会误伤别处）；
    2. fix 不是"建议回看原片确认"这类**操作指引**——把指引写进正文等于篡改笔记；
    3. fix 里的硬事实（数字/英文专名）全部能在该条证据里找到——fix 是模型生成的，
       没证据兜着就可能把幻觉写进笔记；fix 里一个硬事实都没有时，退回与证据的字面重合度。
    返回 (新正文, 已改说明列表)；被跳过的改写经 on_step 报告，保持透明。
    """
    applied: list[str] = []
    out = note_text
    for c in result.claims:
        if c.verdict != VERDICT_MISMATCH or not c.fix or not c.original:
            continue
        if c.original not in out:
            continue
        if _ADVICE_FIX_RE.search(c.fix):
            why = "修正建议是操作指引而非改写文本"
        elif not _fix_supported(c):
            why = "修正建议在证据里找不到支撑（数字/专名不符或字面重合太低），不落盘"
        else:
            why = ""
        if why:
            if on_step:
                on_step(f"   [!] {c.cid} 跳过自动改写：{why}；fix：{c.fix[:40]}")
            continue
        out = out.replace(c.original, c.fix, 1)
        applied.append(f"{c.cid}: {c.original[:30]} → {c.fix[:30]}")
    return out, applied
