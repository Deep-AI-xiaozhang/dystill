"""L1 原子声明拆分（核对四层的第一层）。

自研取舍（与 claimify 的差别，写清楚免得被当成"没抄全"）：
claimify 把 选择 / 消歧 / 分解 三阶段全交给模型。我们只把 **消歧 + 分解** 交给模型，
**选择** 用确定性规则做——理由有三：
1. "哪些句子值得核"是产品边界问题（只核事实点，不核观点/概括/情绪），
   这是规则能精确表达的产品决策，交给模型只会每次漂移，导致基准集无法复现；
2. 规则先行 ⇒ 模型挂了也能给出可用的声明列表（**绝不空手**，与 notes.py 同一条纪律）；
3. 省一次全量调用，且报告里的"跳过原因"可解释、可对账。

规则负责：跳过（观点/情绪/概括句/"无"项/元信息）+ 分节 + 粗分类（数字/专名/步骤/引用）。
模型负责：把"它/这个/该工具"补全成自足句（消歧），把一句话里的并列事实拆成多条（分解）。
模型不可用 → 直接返回规则结果，笔记照样能出一份核对报告。

⚠️ 消歧这一段**默认关闭**（`pipeline.check_l1_disambiguate`，理由与开启条件见 D24）：
它改写/拆分声明，会改变基准集的可比性，属于要先用同日对照组量过的行为变更（D16/D20 纪律）；
2026-10-04 复查还发现接线本身是坏的——`analyze` 把判定通道的解析放在本函数**之后**，
生产路径（CLI / bench_check 都不传 chat_fn）里消歧的 chat_fn 恒为 None，从未真正执行过。
"""
from __future__ import annotations

import json
import re

from .models import (
    KIND_ENTITY,
    KIND_NUMBER,
    KIND_OTHER,
    KIND_QUOTE,
    KIND_STEP,
    Claim,
)
from .modes import CheckMode

MIN_CLAIM_CHARS = 6          # 短于此的残句并入下一句
MAX_SKIPPED_KEEP = 20        # 报告里最多展示的跳过样例数
# 消歧一次送多少条。**不能整篇一次送**：实测（2026-10-13）120 条一次送，
# 某第三方通道直接返回空 content（声明越长越明显），整篇消歧静默失效。
# 25 条时仍有约 1/5 的批返回不可解析，调到 15（单批输出更短、更不容易被截断）。
DISAMBIGUATE_BATCH = 15

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
# 上一轮核对写进笔记的徽章块：必须整块剥掉，否则第二次核对会把 `<!-- dystill-check:begin -->`
# 当成一条声明核进去（实测：会被切碎成 "<" + "，-- dystill-check:begin" 两条假声明）
_CHECK_BLOCK_RE = re.compile(r"<!--\s*dystill-check:begin\s*-->.*?<!--\s*dystill-check:end\s*-->",
                             re.DOTALL | re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_META_LINE_RE = re.compile(r"^\s*>")
_INLINE_MD_RE = re.compile(r"[*_`]+")
_TRAILING_TAG_RE = re.compile(r"^([^：:]{1,12})[：:]\s*")
# 笔记的「小节标签」写法：`- **[02:04] WorkBuddy（腾讯）**：`。时间码必须先剥掉，
# 否则 _TRAILING_TAG_RE 会在时间码内部的第一个冒号处咬断（`[02` 命中「≤12 字符 + 冒号」），
# 剩下 `04] WorkBuddy（腾讯）：` —— 含 2 位数字被判成「数字类」，连观点过滤都绕过，
# 于是每一条小节标签都变成一条待核对声明（实测 035328：14 条这类碎片，1 条还被判 ✅）。
_TIMECODE_RE = re.compile(r"^[\[\(（]?\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\]\)）]?\s*")
# 整行只是「标签：」链（可嵌套，如 `第一梯队：Codex & Claude Code：`）⇒ 栏目标记，不是声明。
# 为什么不只看“剥掉一个标签后剩不剩东西”：标签自己会带冒号、也会超过 _TRAILING_TAG_RE 的 12 字上界
# （`WorkBuddy（腾讯）：` 已 13 字、`DeepSeek Harness架构：` 19 字），那样它们仍会当成声明进核对。
# 上界 24 取实测最长标签（19 字）留余量；“每个分段都以冒号结尾才吃下整行”这条
# 让正文句不会被误吃（`优势1：上手门槛低，…` 第二段就没有冒号）。
_LABEL_LINE_RE = re.compile(r"^(?:[^：:\n]{1,24}[：:]\s*[，,、;；]?\s*)+$")
_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAIL_PUNCT = "。！？!?；;，,、"
# 全角/半角标点与符号组成的"空行"（表格分隔行的碎片会落进来）
_SYMBOL_ONLY_RE = re.compile(r"^[<>\[\]{}()（）\-—–_=+*/\\|~^.,，。、;；:：!！?？\"'“”‘’\s]+$")

# 观点/建议/情绪：没有硬事实点时跳过（边界——不核对概括是否到位、不评价观点）
_OPINION_RE = re.compile(
    r"(我认为|我觉得|个人觉得|值得|推荐大家|建议大家|建议你|最好|显然|可以说|无疑是|"
    r"令人|非常|其实|大概|或许|应该是|感觉|体验很好|香|踩坑|真香|yyds)"
)
# 概括性栏目：里面的句子本质是"总结"，只有带硬事实点时才核
_SUMMARY_SECTIONS = ("结论先行", "总结与适用场景", "一句话")

_NUMBER_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|％|元|块|万|千|亿|倍|个|款|条|张|次|天|周|月|年|小时|分钟|秒|GB|MB|KB|token|k\b))"
    r"|(\d{2,})|([一二三四五六七八九十百千万两]\s*(?:款|个|条|步|种|天|周|月|年|小时|分钟|倍|成|万|元))"
)
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.+-][A-Za-z0-9]+)*")
_QUOTE_RE = re.compile(r"(原话|他说|她说|作者说|提到|表示|引用|据称)|[「『“\"].{2,}[」』”\"]")
_STEP_RE = re.compile(r"(点击|打开|输入|粘贴|安装|配置|设置|选择|运行|执行|命令|步骤|流程|注册|登录|导出|导入|填|拖)")

# 工具链名词黑名单：蒸馏管道自己的名词不该被当成"视频里的专名"来核对
_PIPELINE_WORDS = {"faster-whisper", "faster_whisper", "PySceneDetect", "scenedetect",
                   "douyin-downloader", "whisper", "ffmpeg", "yt-dlp", "OpenAI"}


def strip_frontmatter(md: str) -> str:
    """剥掉 frontmatter 与**上一轮写入的核对徽章块**——这两块都不是笔记正文。"""
    s = _FRONTMATTER_RE.sub("", md or "", count=1)
    return _CHECK_BLOCK_RE.sub("", s)


def iter_blocks(md: str):
    """按标题分节，产出 (章节名, 原文, 子标签)。子标签来自 `- **关键概念**：xxx` 这类字段。"""
    section, buf = "正文", []

    def flush():
        if buf:
            yield section, "\n".join(buf).strip()
        buf.clear()

    for raw in strip_frontmatter(md).splitlines():
        m = _HEADING_RE.match(raw)
        if m:
            yield from flush()
            section = m.group(1).strip()      # 取标题正文，不带 ### 前缀（此前会把"### xxx"当章节名）
            continue
        if _META_LINE_RE.match(raw):      # 顶部元信息（蒸馏日期/类型/主题）不是声明来源
            continue
        buf.append(raw)
    yield from flush()


def _clean(line: str) -> str:
    s = _HTML_COMMENT_RE.sub("", line)
    s = _INLINE_MD_RE.sub("", s)
    s = _BULLET_RE.sub("", s)
    return s.strip()


def _table_cells(row: str) -> list[str] | None:
    """markdown 表格行 → 单元格列表；分隔行（|---|）返回 []；非表格行返回 None。

    实测（2026-09-17 真实笔记）：不处理表格时，`|------|------|` 会被当成一条声明、
    整行 `| 场景1 | 调研竞品 |` 会被当成一条"含多个事实的长声明"——两样都是噪声。
    按单元格拆开后，每个格子各自过闸，语义单元才对得上原视频。
    """
    s = row.strip()
    if not s.startswith("|"):
        return None
    body = s.strip("|")
    if not body.strip() or set(body) <= set("|-: \t"):
        return []
    return [c.strip() for c in body.split("|")]


def split_sentences(text: str) -> list[str]:
    """按行 + 句末标点切句；表格行按单元格拆开；过短残句并入下一句。

    残句合并**不得跨越表格行**（2026-09-30 实测真 bug，历史考点视频 6 条 ❌ 里 3 条是它产的）：
    单元格被拉平后，上一行末尾的短格（如「南北面官制」5 字 < 6 字阈值）会并进下一行首格
    （「西夏」），制造出笔记里根本不存在的「南北面官制，西夏」跨政权假声明——
    L2 拿黑板板书一查，配对错误，引擎把它们全判 ❌，笔记被冤枉。

    边界必须**双向**（2026-10-04 复查 D18 修复时发现只堵了单向，同型假声明换个方向又来）：
    - 表格行**首格**不许并入上一行（原修复，堵「行尾短格 + 下一行首格」）；
    - 表格行**末格**不许被下一行并入（补漏：`| 辽 | 契丹 |` + 换行 + `元昊建立西夏`
      原来并成「辽，契丹，元昊建立西夏」）。
    行内格间合并照旧保留：「党项，元昊」是这一行自己的配对，合法声明。

    表格**表头行**（后面紧跟 `|---|---|` 分隔行）整行丢弃：它是列名导航不是断言。
    实测（2026-10-04）公布验收集里 3 条伪声明（`政权，建立时间`/`民族，建立者`/
    `定都，核心制度`）就是表头行被格间合并产出的，还被人标成 ok 计进了命中。
    """
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text or "") if ln.strip()]
    # 表头行 = 表格行且下一行是纯分隔行（markdown 标准写法，不会误伤真数据行）
    headers = {i for i, raw in enumerate(lines)
               if _table_cells(raw) and i + 1 < len(lines) and _table_cells(lines[i + 1]) == []}

    # (片段, 不许并入前一片段, 不许被后一片段并入)
    frags: list[tuple[str, bool, bool]] = []
    for i, raw in enumerate(lines):
        if i in headers:
            continue
        cells = _table_cells(raw)
        parts = [c for c in (_clean(c) for c in (cells or [])) if c] if cells is not None \
            else [_clean(raw)]
        last = len(parts) - 1
        for j, part in enumerate(parts):
            pieces = [p.strip().strip(_TAIL_PUNCT).strip() for p in _SPLIT_RE.split(part)]
            pieces = [p for p in pieces if p]
            for k, piece in enumerate(pieces):
                # 行首格的第一句：自己起一组；行末格的最后一句：别让下一行并进来。
                # 边界只加在**格的两端**，格内按标点切出的短句仍可互相合并
                # （| Claude | 体验最好；操作流程… | 里的「体验最好」要能并回去）。
                back = cells is not None and j == 0 and k == 0
                blocks_fwd = cells is not None and j == last and k == len(pieces) - 1
                frags.append((piece, back, blocks_fwd))

    merged: list[str] = []
    may_extend = True            # 当前合并组的末片段还能不能被后面的短片段并入
    for f, back, blocks_fwd in frags:
        if merged and not back and may_extend and len(merged[-1]) < MIN_CLAIM_CHARS:
            merged[-1] = f"{merged[-1]}，{f}"
        else:
            merged.append(f)
        may_extend = not blocks_fwd
    return [m for m in merged if len(m) >= MIN_CLAIM_CHARS]


def classify_kind(text: str) -> str:
    """粗分类：数字 > 引用 > 专名 > 步骤 > 其他（顺序即优先级，数字最可信）。"""
    if _NUMBER_RE.search(text):
        return KIND_NUMBER
    if _QUOTE_RE.search(text):
        return KIND_QUOTE
    words = {w for w in _EN_WORD_RE.findall(text) if len(w) >= 2 and w not in _PIPELINE_WORDS}
    if words:
        return KIND_ENTITY
    if _STEP_RE.search(text):
        return KIND_STEP
    return KIND_OTHER


def is_checkable(text: str) -> tuple[bool, str]:
    """是否值得核对。返回 (可核, 跳过原因)。理由串直接进报告，保证"没核"是透明的。"""
    t = _HTML_COMMENT_RE.sub("", text).strip().strip(_TAIL_PUNCT).strip()
    if len(t) < MIN_CLAIM_CHARS:
        return False, "过短"
    if t in ("无", "暂无", "略") or re.fullmatch(r"(无|暂无|略)[。.]?", t):
        return False, "笔记自己标了「无」"
    if _SYMBOL_ONLY_RE.match(t):
        return False, "符号/标记行（markdown 表格分隔、HTML 注释碎片等）"
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t):
        return False, "无内容"
    # 小节标题（2026-09-30 基准实测 FP 来源）：`**【关于男女思维差异的底层逻辑】**` 这类
    # 整条只剩【】包裹的内容（剥掉后连标点引号都不剩），是导航标记不是断言，
    # 引擎对"标题"只能判 ok（标题词总能在视频里对上）→ 稳定产假 ✅
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", re.sub(r"【[^【】]*】", "", t)):
        return False, "小节标题（整条只剩【】标题，无断言内容）"
    kind = classify_kind(t)
    # 只看无硬事实点的句子：带数字/专名/步骤的句子哪怕夹着"推荐"也必须核（可能推荐错了工具名）
    if kind == KIND_OTHER and _OPINION_RE.search(t):
        return False, "观点/情绪句（不核对观点）"
    return True, ""


def cap_balanced(claims: list[Claim], cap: int) -> tuple[list[Claim], bool]:
    """超上限时按章节轮转取样，尽量让每一节都有代表进入核对。

    为什么不用 head 截断（`claims[:cap]`）：实测真实笔记 164 条候选 / 上限 120，
    head 截断会把**后半篇整节**（数据基准、对你的启发）一条都不核——用户看到
    "已核对 120 条"根本不知道尾部有 44 条连看都没看。轮转取样保证"每节都覆盖一些"，
    并在报告里写明截断事实。返回 (取样结果, 是否发生截断)。
    """
    if len(claims) <= cap:
        return claims, False
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(claims):
        groups.setdefault(c.section, []).append(i)
    picked: list[int] = []
    round_no = 0
    while len(picked) < cap:
        added = False
        for idxs in groups.values():
            if round_no < len(idxs) and len(picked) < cap:
                picked.append(idxs[round_no])
                added = True
        if not added:
            break
        round_no += 1
    return [claims[i] for i in sorted(picked)], True


def extract_by_rules(md: str, mode: CheckMode) -> tuple[list[Claim], list[str]]:
    """纯规则拆分：分节 → 切句 → 过滤 → 分类。永不空手，也永不抛错。"""
    claims: list[Claim] = []
    skipped: list[str] = []
    for section, block in iter_blocks(md):
        if not block:
            continue
        in_summary = any(k in section for k in _SUMMARY_SECTIONS)
        for sent in split_sentences(block):
            head = _TIMECODE_RE.sub("", sent).strip()
            # 栏目标记先拦：它限定的是**它下面的**声明，自己没有被核的内容。
            # 原实现的兜底 `body = sent` 正是噪声来源——剥完标签什么都不剩时
            # 又把整行当声明收回来，于是 `[02:04] WorkBuddy（腾讯）：` 还因含 2 位数字
            # 被判成「数字类」绕过观点过滤（实测 4 条视频共 29 条这类噪声）。
            if _LABEL_LINE_RE.fullmatch(head):
                if len(skipped) < MAX_SKIPPED_KEEP:
                    skipped.append(f"{sent[:40]}〔栏目标记（只有标签没有正文）〕")
                continue
            sub = _TRAILING_TAG_RE.match(head)
            body = head[sub.end():].strip() if sub else head
            # 字段标签（"注意事项：无"）先摘掉再判定——否则"无"会被当成一条声明核进去
            if not body:
                if len(skipped) < MAX_SKIPPED_KEEP:
                    skipped.append(f"{sent[:40]}〔栏目标记（只有标签没有正文）〕")
                continue
            ok, reason = is_checkable(body)
            kind = classify_kind(body)
            if ok and in_summary and kind == KIND_OTHER:
                ok, reason = False, "概括句（核对只保证没编造，不判概括是否到位）"
            if not ok:
                if len(skipped) < MAX_SKIPPED_KEEP:
                    skipped.append(f"{sent[:40]}〔{reason}〕")
                continue
            label = section if not sub else f"{section}/{sub.group(1)}"
            claims.append(Claim(cid="", text=body, section=label,
                                kind=kind, original=sent))
    if mode.kinds:
        claims = [c for c in claims if c.kind in mode.kinds]
    claims, capped = cap_balanced(claims, mode.max_claims)
    if capped:
        # 截断必须写进报告：否则用户以为"全都核过了"，实际只是核了一部分
        skipped.append(f"（候选声明超过上限 {mode.max_claims}，已按章节轮转取样——"
                       f"未核到的条目请调大 pipeline.check_max_claims 或分段核对）")
    return claims, skipped


_DISAMBIGUATE_PROMPT = """你是中文视频笔记的核对助手。下面是按顺序编号的**待核对声明**（来自一篇视频蒸馏笔记）。

请对每一条做两件事：
1. **消歧**：把"它/这个/该工具/前者"这类指代补全成具体名称（用上下文里出现过的名字）；补不出就保留原样，不要编。
2. **拆分**：一句话里若含多个可独立验证的事实（多个数字、多个工具、多个步骤、多个并列项），
   拆成多条**原子句**，每条只讲一个事实点——原子句才好逐条回原素材核对。

硬约束：
- **只能改写表达，绝不能增加原文没有的信息**；补不出的指代保持原样。
- **逐条返回、编号原样对应**：不许跳过、不许合并、不许新增或改动编号。
  （曾因让模型自己编号导致错位、改写内容串到别的声明上——49/120 条章节与内容对不上。）
- 每条返回一个 `parts` 数组：不需要拆分时，数组里就放一个元素（原句或消歧后的句子）。
- 只返回 JSON 数组，不要解释、不要 markdown 围栏。格式：
  [{{"i": 原始编号, "parts": ["原子句一", "原子句二"]}}]

待核对声明：
{items}
"""


def _fmt_items(claims: list[Claim]) -> str:
    return "\n".join(f"{i}. [{c.section}] {c.text}" for i, c in enumerate(claims, 1))


def parse_disambiguation(raw: str, n: int) -> dict[int, list[str]]:
    """解析 [{i, parts:[...]}]（兼容旧格式 [{i, text}]），返回 {编号: [原子句...]}。

    **任何编号异常都返回 {} 让整批退回原文**：错位回填比不拆分危险得多
    （实测 2026-10-13：模型自己编号时 49/120 条内容串到了别的声明上）。
    """
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    out: dict[int, list[str]] = {}
    for row in data if isinstance(data, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        raw_parts = row.get("parts")
        parts = ([str(p).strip() for p in raw_parts] if isinstance(raw_parts, list)
                 else [str(row.get("text") or "").strip()])      # 兼容旧格式
        parts = [p for p in parts if len(p) >= MIN_CLAIM_CHARS]
        if not (1 <= i <= n) or not parts:
            continue
        if i in out:
            return {}                    # 重复编号 = 模型编号乱了，整批作废
        out[i] = parts
    return out


def disambiguate(claims: list[Claim], chat_fn, on_step=None) -> list[Claim]:
    """用模型做消歧/拆分；分批送（整篇一次送会返回空），单批失败该批原样保留。

    不抛错、不空手：任何一批失败都只是让那批退回规则原文，其余批次照常。
    拆出的多条**由代码按输入编号展开**（不让模型自己编号）——这是错位事故的根治：
    模型只回答"第 i 条拆成哪几段"，映射由我们维护，不可能串行。
    """
    if not claims or chat_fn is None:
        return claims
    out: list[Claim] = []
    total_batches = (len(claims) + DISAMBIGUATE_BATCH - 1) // DISAMBIGUATE_BATCH
    for bi, start in enumerate(range(0, len(claims), DISAMBIGUATE_BATCH), 1):
        batch = claims[start:start + DISAMBIGUATE_BATCH]
        mapping: dict[int, list[str]] = {}
        err = ""
        # 重试一次：模型偶尔把 JSON 包进解释或截断，重问一次往往就好
        # （实测 2026-10-13：约 1/7 批首次不可解析）。
        for _attempt in (1, 2):
            try:
                raw = chat_fn([{"role": "user",
                                "content": _DISAMBIGUATE_PROMPT.format(items=_fmt_items(batch))}])
                mapping = parse_disambiguation(getattr(raw, "content", raw) or "", len(batch))
                err = ""
            except Exception as e:  # noqa: BLE001 —— 消歧失败不该毁掉整次核对
                err = f"{type(e).__name__}: {e}"
            if mapping:
                break
        if not mapping and on_step:
            why = f"调用失败（{err}）" if err else "返回不可解析"
            on_step(f"   [!] 消歧第 {bi}/{total_batches} 批重试后仍{why}，该批改用规则原文")
        for j, c in enumerate(batch, 1):
            parts = mapping.get(j)
            if not parts:
                out.append(c)                     # 该条退回规则原文
                continue
            for k, p in enumerate(parts):
                # original 只给第一条：--rewrite 靠它在笔记里做逐字替换，
                # 拆出的多条共享同一个 original 会导致同一句被反复替换
                out.append(Claim(cid="", text=p, section=c.section, kind=c.kind,
                                 original=c.original if k == 0 else ""))
    return out


def assign_cids(claims: list[Claim]) -> list[Claim]:
    """给声明编号（报告里可引用）。放在最后一步，保证任何分支产出的声明都有 cid。"""
    for i, c in enumerate(claims, 1):
        c.cid = f"C{i:02d}"
    return claims


def extract_claims(md: str, mode: CheckMode, chat_fn=None, on_step=None) -> tuple[list[Claim], list[str]]:
    """L1 主入口：规则选择 + 模型消歧/分解。返回 (声明列表, 跳过原因样例)。"""
    claims, skipped = extract_by_rules(md, mode)
    if on_step:
        on_step(f"   L1 规则选出 {len(claims)} 条候选声明（跳过 {len(skipped)} 句）")
    return assign_cids(disambiguate(claims, chat_fn, on_step)), skipped
