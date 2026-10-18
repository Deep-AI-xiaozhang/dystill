"""知识库：把蒸馏出的笔记汇成可检索、可归档的索引，并做完整性自检。

设计红线（本项目定位）：本模块只做「汇总与自检」，不带任何领域倾向。
- 不预设分类表：主题标签全部从笔记自身内容里长出来（作者写的 #话题 标签 +
  语料内高频词），换任何题材的用户都能用，不会被绑到某个垂直方向。
- 不评价内容：不判断"值不值得看""能不能赚钱"，只做归拢、去重、找缺件。
- 分类轴只有一个：笔记自己带的 类型（实操教程 / 观点认知），由笔记层产出。

产物：
  1. 知识库.md —— 索引总览（类型分组 + 主题簇 + 逐篇一句话 + 全量索引表）
  2. 可选（--frontmatter）：给每篇笔记补 type / topic / tags 三个 YAML 字段，
     供 Obsidian 等工具按标签检索。幂等：只增改托管字段，不动用户手写内容。

自检（--check）：缺身份键 / 键格式异常 / 重复 / 空壳 / 有素材无笔记 /
失效引用 / 索引过期。只报事实，不做价值判断。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

INDEX_NAME = "知识库.md"

# 类型轴：笔记层产出的两个值（notes.py）。未知值原样保留，不猜。
KNOWN_TYPES = ("实操教程", "观点认知")
UNCLASSIFIED = "未分类"
UNMATCHED_GROUP = "（未命中任何主题）"

# 虚词边界：以这些字开头或结尾的候选词是切词碎片（"视频的""与使用"），不是主题。
# 纯字符层面的过滤，与领域无关。
_PARTICLE = set("的了与和及或是在对从把被为就也都而但以之其")

_FM_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_TYPE_RE = re.compile(r"类型[：:]\s*([^｜|\n]+)")
_TOPIC_RE = re.compile(r"主题[：:]\s*([^｜|\n]+)")
_HASHTAG_RE = re.compile(r"#([^\s#，。；、｜|]{2,20})")
_CONCLUSION_RE = re.compile(r"##\s*一[、.]\s*结论先行\s*\n(.*?)(?:\n##|\Z)", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*[-*]\s*(.+)$", re.MULTILINE)

# 词频统计丢弃的词：中文功能词 + 笔记结构词。这是语言层面的过滤，不是领域倾向。
_STOP = {"的", "了", "是", "在", "和", "与", "及", "也", "都", "就", "还", "但", "而", "或", "被", "把", "对", "从", "到", "用", "让", "会", "能", "要", "有", "没", "不", "这", "那", "你", "我", "他", "她", "它", "一个", "一些", "这个", "那个", "什么", "怎么", "如何", "可以", "因为", "所以", "但是", "如果", "已经", "需要", "进行", "通过", "使用", "这些", "那些", "我们", "他们", "视频", "内容", "方法", "方式", "时候", "问题", "情况", "地方", "东西", "事情", "上面", "下面", "里面", "后面", "前面", "之后", "之前", "然后", "现在", "就是", "不是", "标题", "笔记", "蒸馏", "要点", "口播", "画面", "步骤", "结论", "先行", "总结", "适用", "场景", "知识", "核心", "视频核心", "核心结论", "核心观点", "结论是", "本视频", "这条视频", "本次", "主要", "介绍", "讲解", "分享", "提到", "包括", "以及", "相关", "完成", "实现", "提供", "支持", "这样", "那样", "演示", "展示", "说明", "强调", "指出", "认为", "表示", "给出", "建议", "注意", "目的", "目标", "为何", "为什么", "哪些", "第一", "第二", "第三", "一步", "两步", "三种", "四种", "五个", "六个", "自己", "自己的", "一定要", "一定", "真的", "其实", "直接", "最后", "开始", "没有"}


@dataclass
class NoteMeta:
    path: Path
    key: str = ""
    title: str = ""
    date: str = ""
    source: str = ""
    ntype: str = UNCLASSIFIED
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    core: str = ""
    body_chars: int = 0

    @property
    def valid_key(self) -> bool:
        """身份键格式：<source>:<id>（identity.py 约定）。空键视为无效。"""
        return bool(re.fullmatch(r"[a-z]+:[A-Za-z0-9_-]+", self.key))


def normalize_type(raw: str) -> str:
    """把类型字段归一成 KNOWN_TYPES 之一。

    实测笔记里的脏值有三类（都由生成层产生，索引层只能容错）：
      `<实操教程>`        模板占位符的尖括号没删干净
      `**观点认知**`      Markdown 加粗被原样写进字段
      `<观点认知+实操指南>` 占位符没删 + 自造组合值
    只要值里含已知类型就归到那个类型；否则原样保留（不猜、不吞信息）。
    """
    s = (raw or "").strip().strip("*").strip("`").strip()
    s = s.strip("<>").strip("《》").strip()
    for t in KNOWN_TYPES:
        if t in s:
            return t
    return s or UNCLASSIFIED


def parse_note(path: Path) -> NoteMeta:
    """从笔记文件解析索引字段。缺字段留空，不抛异常（坏文件也要能进自检报告）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = NoteMeta(path=path)

    fm = _FM_RE.match(text)
    if fm:
        for line in fm.group(1).splitlines():
            if ":" not in line or line.startswith(" "):
                continue
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "key":
                meta.key = v
            elif k == "title" and not meta.title:
                meta.title = v
            elif k == "date":
                meta.date = v
            elif k == "source":
                meta.source = v

    if not meta.title:
        m = _H1_RE.search(text)
        if m:
            meta.title = re.sub(r"^蒸馏笔记\s*·?\s*", "", m.group(1).strip())

    m = _TYPE_RE.search(text)
    if m:
        meta.ntype = normalize_type(m.group(1))
    m = _TOPIC_RE.search(text)
    if m:
        meta.topic = m.group(1).strip()

    seen, tags = set(), []
    for raw in _HASHTAG_RE.findall(meta.title + " " + meta.topic):
        t = raw.strip()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)
    meta.tags = tags

    m = _CONCLUSION_RE.search(text)
    if m:
        b = _BULLET_RE.search(m.group(1))
        if b:
            core = b.group(1)
        else:
            first = m.group(1).strip().splitlines()
            core = first[0] if first else ""
        meta.core = " ".join(core.split())[:100]

    body = re.sub(r"\A---\n.*?\n---\n?", "", text, flags=re.DOTALL)
    meta.body_chars = len(re.sub(r"\s+", "", body))
    return meta


def find_notes(data_dir: Path, notes_dir: Path | None = None) -> list[NoteMeta]:
    """扫描所有笔记。笔记默认落在 data/<record_key>/note.md；
    notes_dir 若配置且存在，也一并收录（兼容用户把笔记另存一处）。

    必须排除索引自身（INDEX_NAME）——否则第二次运行会把上一次生成的索引
    当成一篇笔记收进去，越滚越大（实测踩到）。
    """
    paths: list[Path] = []
    if data_dir.is_dir():
        paths += sorted(data_dir.glob("*/note.md"))
    if notes_dir and notes_dir.is_dir() and notes_dir.resolve() != data_dir.resolve():
        paths += sorted(notes_dir.glob("*.md"))
        paths += sorted(notes_dir.glob("*/note.md"))
    seen, out = set(), []
    for p in paths:
        if p.name == INDEX_NAME:
            continue
        r = p.resolve()
        if r in seen:
            continue
        seen.add(r)
        out.append(parse_note(p))
    return out


def _ngrams(text: str, n_min: int = 2, n_max: int = 4) -> list[str]:
    """切出候选词：连续汉字取 2~4 字滑窗；英文/数字整词。"""
    out = []
    for m in re.finditer(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9+#.-]{1,}", text):
        seg = m.group(0)
        if re.match(r"[A-Za-z]", seg):
            out.append(seg.lower())
            continue
        for n in range(n_min, n_max + 1):
            for i in range(len(seg) - n + 1):
                out.append(seg[i:i + n])
    return out


def _topic_text(meta: NoteMeta) -> str:
    """用于主题统计的文本：标题 + 主题。

    刻意不含正文：正文是模型生成的话术，结构词（"视频核心""结论是"）会霸榜。
    标题与主题是作者/视频自己的话，主题信号最干净。
    """
    return f"{meta.title} {meta.topic}"


def top_keywords(metas: list[NoteMeta], limit: int = 20) -> list[tuple[str, int]]:
    """语料内高频词。完全从笔记内容统计，无预设词表——换任何题材都成立。

    按「出现在多少篇笔记里」计数，而不是出现次数，避免单篇长文霸榜。
    只取标题与主题两个字段：正文是模型写的话术，结构词多、噪声大（实测过）。
    """
    if not metas:
        return []
    doc_hits: dict[str, int] = {}
    for meta in metas:
        for g in set(_ngrams(_topic_text(meta))):
            if g in _STOP or len(g) < 2:
                continue
            # 虚词边界：切词碎片（"视频的""与使用"）不是主题
            if g[0] in _PARTICLE or g[-1] in _PARTICLE:
                continue
            doc_hits[g] = doc_hits.get(g, 0) + 1
    threshold = 2 if len(metas) >= 3 else 1
    items = [(g, n) for g, n in doc_hits.items() if n >= threshold]
    items.sort(key=lambda x: (-x[1], -len(x[0])))
    kept: list[tuple[str, int]] = []
    for g, n in items:
        if any(g != k and g in k for k, _ in kept):
            continue
        kept.append((g, n))
        if len(kept) >= limit:
            break
    return kept


def group_by_topic(metas: list[NoteMeta], keywords: list[tuple[str, int]]) -> dict[str, list[NoteMeta]]:
    """按主题词归组：**命中即收录，一篇可出现在多个主题下**。

    刻意不做"每篇只归一类"——那会让没命中任何主题词的笔记掉进一个「其它」黑洞
    （实测 24 篇里有 9 篇掉进去，等于白归）。多标签归组的好处是每个主题组都是完整的，
    代价是跨主题的笔记会重复出现，对索引来说这个代价可以接受。

    没命中任何主题词的笔记单独成组，保证没有任何一篇被静默吞掉。
    """
    groups: dict[str, list[NoteMeta]] = {}
    for kw, _ in keywords:
        # 大小写不敏感：英文关键词统计时已转小写，标题里是 "AI"，直接 in 会漏（实测踩到）
        hit = [m for m in metas if kw in _topic_text(m).lower()]
        if hit:
            groups[kw] = hit
    matched = {id(m) for items in groups.values() for m in items}
    rest = [m for m in metas if id(m) not in matched]
    if rest:
        groups[UNMATCHED_GROUP] = rest
    return groups


def build_index(metas: list[NoteMeta], out_path: Path, generated: str) -> str:
    """生成索引 Markdown（返回纯字符串，便于单测）。"""
    metas = sorted(metas, key=lambda m: (m.date or "9999", m.title))
    keywords = top_keywords(metas)
    by_type: dict[str, list[NoteMeta]] = {}
    for m in metas:
        by_type.setdefault(m.ntype, []).append(m)
    groups = group_by_topic(metas, keywords)

    dates = [m.date for m in metas if m.date]
    span = f"{min(dates)} ~ {max(dates)}" if dates else "—"

    lines = [
        "# 知识库 · 索引",
        "",
        f"> 生成时间：{generated} ｜ 收录 **{len(metas)}** 篇 ｜ 时间跨度 {span}",
        "> 本文件由 dystill kb 自动生成，重复运行结果一致（幂等）。",
        "> 分类来自笔记自身内容（类型 + 高频词），不含任何预设领域词表。",
        "",
        "---",
        "",
        "## 一、总览",
        "",
        "| 类型 | 篇数 | 占比 |",
        "|---|---:|---:|",
    ]
    total = max(1, len(metas))
    order = list(KNOWN_TYPES) + sorted(k for k in by_type if k not in KNOWN_TYPES)
    for t in order:
        items = by_type.get(t) or []
        if not items:
            continue
        lines.append(f"| {t} | {len(items)} | {len(items) / total * 100:.0f}% |")

    if keywords:
        lines += ["", "## 二、高频主题（从笔记内容自动统计）", "", "| 主题词 | 出现篇数 |", "|---|---:|"]
        lines += [f"| {kw} | {n} |" for kw, n in keywords]

    lines += ["", "## 三、按主题归组", "",
              "> 一篇笔记命中多个主题时会重复出现（多标签归组），每个主题组都是完整的。", ""]
    for kw, items in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        lines.append(f"### {kw}（{len(items)} 篇）")
        lines.append("")
        for m in items:
            lines.append(f"- **{m.title or '(无标题)'}** — {m.core or '（无一句话核心）'}")
        lines.append("")

    lines += ["---", "", "## 四、全量索引", "",
              "| 日期 | 标题 | 类型 | 主题 | 核心 |", "|---|---|---|---|---|"]
    for m in metas:
        title = (m.title or "(无标题)").replace("|", "｜")
        topic = (m.topic or "—").replace("|", "｜")
        core = (m.core or "—").replace("|", "｜")
        lines.append(f"| {m.date or '—'} | {title} | {m.ntype} | {topic} | {core} |")
    lines += ["", f"*共 {len(metas)} 篇 ｜ 自动生成：{generated}*", ""]
    return "\n".join(lines)


def upsert_frontmatter(path: Path, meta: NoteMeta) -> bool:
    """给笔记补 type/topic/tags（幂等，只动托管字段，保留用户手写字段）。

    返回 True 表示文件有变化。已有 frontmatter 时逐字段更新；没有则新建。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    managed = {
        "type": meta.ntype,
        "topic": meta.topic,
        "tags": "[" + ", ".join(meta.tags) + "]",
    }
    fm = _FM_RE.match(text)
    if fm:
        keys_seen = set()
        new_lines = []
        for ln in fm.group(1).splitlines():
            k = ln.split(":", 1)[0].strip() if ":" in ln else ""
            if k in managed:
                new_lines.append(f"{k}: {managed[k]}")
                keys_seen.add(k)
            else:
                new_lines.append(ln)
        for k, v in managed.items():
            if k not in keys_seen:
                new_lines.append(f"{k}: {v}")
        new_text = "---\n" + "\n".join(new_lines) + "\n---\n" + text[fm.end():]
    else:
        block = "---\n" + "\n".join(f"{k}: {v}" for k, v in managed.items()) + "\n---\n\n"
        new_text = block + text
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


@dataclass
class Issue:
    level: str      # ERROR / WARN
    kind: str
    detail: str


def check_kb(metas: list[NoteMeta], data_dir: Path, index_path: Path,
             min_body_chars: int = 200) -> list[Issue]:
    """知识库完整性自检。只报事实，不做价值判断。"""
    issues: list[Issue] = []

    for m in metas:
        if not m.key:
            issues.append(Issue("WARN", "缺身份键", f"{m.path.parent.name}/note.md 没有 key，无法去重"))
        elif not m.valid_key:
            issues.append(Issue("WARN", "身份键格式异常", f"{m.path.parent.name}: key={m.key!r}"))

    seen: dict[str, int] = {}
    for m in metas:
        if m.key:
            seen[m.key] = seen.get(m.key, 0) + 1
    for k, n in seen.items():
        if n > 1:
            issues.append(Issue("ERROR", "重复笔记", f"key={k} 出现 {n} 次"))

    for m in metas:
        if m.body_chars < min_body_chars:
            issues.append(Issue("ERROR", "空壳笔记",
                                f"{m.path.parent.name}: 正文仅 {m.body_chars} 字（<{min_body_chars}）"))

    if data_dir.is_dir():
        dirs = {d.name for d in data_dir.iterdir() if d.is_dir()}
        note_dirs = {m.path.parent.name for m in metas
                     if m.path.parent.parent.resolve() == data_dir.resolve()}
        for name in sorted(dirs - note_dirs):
            d = data_dir / name
            if (d / "transcript.txt").exists() or (d / "vlm.txt").exists():
                issues.append(Issue("WARN", "有素材无笔记", f"{name} 有转写/画面但没生成 note.md"))
        for name in sorted(note_dirs - dirs):
            issues.append(Issue("ERROR", "失效引用", f"{name} 的笔记存在但素材目录已不在"))

    if not index_path.exists():
        issues.append(Issue("WARN", "索引未生成", f"{index_path.name} 不存在，跑一次 dystill kb"))
    else:
        txt = index_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"收录\s*\*\*(\d+)\*\*\s*篇", txt)
        if m and int(m.group(1)) != len(metas):
            issues.append(Issue("WARN", "索引过期",
                                f"索引收录 {m.group(1)} 篇，实际 {len(metas)} 篇，需重新生成"))
    return issues
