"""L4 核对报告生成（核对四层第四层 + 用户感知设计）。

三种交付形态，各自服务一个场景：
1. `check.json` —— note_check 记录，给下游/后续版本用（唯一机器真相源）
2. `check.md`   —— 可折叠逐条报告，给人复核用（每条带时间戳，可点回原片）
3. 笔记顶部徽章 —— 给"扫一眼就走"的用户：✅ 已抽核 12/15 · ⚠️ 2 条存疑
   （措辞用「抽核」不用「核对」：判定一致率尚未过验收线 + 长笔记只核抽样，
   「已核对」的隐含承诺高于实际能力——降级理由与回改条件见 DECISIONS D14）

徽章写入用 HTML 注释包裹，**幂等**：重复核对同一篇笔记只会替换标记之间的内容，
不会越插越多（"重复追加"事故的结构性规避）。

诚实边界写进报告头，不藏：核对只保证"没编造"，不保证"概括得准"；核对基于转写，
转写本身出错不在核对范围内。这句必须出现在产物里，否则用户会拿"概括不到位"来骂。
"""
from __future__ import annotations

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
        "> **正确用法**：当\"快速定位可疑片段\"工具，看到 ❌ 后点时间戳回原片确认，不要盲目相信。",
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


def write_reports(record_dir: Path, result: CheckResult) -> dict:
    """落盘 check.json / check.md，返回路径。"""
    rd = Path(record_dir)
    rd.mkdir(parents=True, exist_ok=True)
    data = build_json(result)
    (rd / "check.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (rd / "check.md").write_text(render_markdown(result), encoding="utf-8")
    return {"json": str(rd / "check.json"), "markdown": str(rd / "check.md")}


def inject_badge(note_text: str, result: CheckResult) -> str:
    """把徽章插到笔记 top（frontmatter 之后）；已存在则替换标记之间内容（幂等）。"""
    line = (f"{BEGIN_MARK}\n"
            f"> {badge(result)} ｜ 模式：{result.mode} ｜ {result.created_at}\n"
            f"> 「抽核」＝抽声明回原素材找出处；⚠️/未核条目不在保证范围，逐条见 `check.md`（同目录）\n"
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
