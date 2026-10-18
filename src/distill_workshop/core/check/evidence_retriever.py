"""L2 证据回查（核对四层第二层）。

与 guard 的差别：他们的 retriever 去网页/Wikipedia 找证据，**我们的证据只来自原视频本身**
（转写 + 画面理解 + timecodes.json）——这是"核对"不是"求真"：判据是"笔记里的话在原素材里有没有出处"，
不是"这话在互联网上对不对"。所以只做同素材内部检索，不接外部知识源。

技术选型（M1 不引 embedding，见 modes.py 注释）：中文无分词依赖的前提下，用**字/词 n-gram 字面覆盖度**：
- 强项：数字、专名、工具名这类"硬事实点"字面几乎不变，命中即高可信；
- 弱项：同义改写（"四款工具" vs "四种 AI 产品"）会漏召回 → 由 L3 判定时容忍，
  且漏召回只会让声明落到 ⚠️ 存疑，**不会**被误判成 ✅（宁存疑不误判，方向是有意选的）。

时间戳是视频笔记独有的：转写按 `[mm:ss]` 分段，画面证据经 timecodes.json 把"第 N 张图"映射回秒，
所以报告里每条声明都能点回原片位置。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Evidence

# ── 证据类型标注（2026-10-16 改进：给 L3 判定提供证据优先级提示）──
# opening: 视频开头定位陈述（前 30 秒）
# contrast: 对比结构（"之前...现在""没有X之前...用了X就"）
# spoken: 口语表达（"多/左右/大概/约"等模糊词）
# visual: 画面证据
# neutral: 普通证据

_CONTRAST_PATTERNS = [
    r'(之前|以前|原来|过去|旧版|老版|曾经|before|used to|previously|old).{0,50}(现在|目前|新版|用了|有了|后来|now|currently|with|after)',
    r'(没有|不用|缺少|without|no).{0,30}(就|得|要|需要|need|have to).{0,50}(有了|用了|装了|加了|with|using).{0,30}(就|才|能|can|able)',
]
_SPOKEN_MARKERS = [
    '多', '左右', '大概', '约', '差不多', '大约', '接近', '上下', '几乎',
    'about', 'around', 'approximately', 'roughly', 'nearly'
]


def _parse_ts_seconds(ts: str) -> float:
    """解析时间戳到秒数，如 "01:23" -> 83.0，失败返回 -1.0"""
    if not ts:
        return -1.0
    parts = ts.split(':')
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return -1.0


def _classify_evidence_type(text: str, source: str, ts: str) -> str:
    """根据证据特征标注类型，供 L3 判定时按优先级排序。"""
    # 画面证据优先级高（画面文字比转写可靠）
    if source == 'vlm':
        return 'visual'
    # 开头定位（前30秒）
    sec = _parse_ts_seconds(ts)
    if 0 <= sec <= 30:
        return 'opening'
    # 对比结构
    for pat in _CONTRAST_PATTERNS:
        if re.search(pat, text or ''):
            return 'contrast'
    # 口语表达
    if any(marker in (text or '').lower() for marker in _SPOKEN_MARKERS):
        return 'spoken'
    return 'neutral'


_TS_LINE_RE = re.compile(r"^\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)$")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+-]*")
_NUM_RE = re.compile(r"\d+")
_IMG_ORD_RE = re.compile(r"第\s*([一二三四五六七八九十百\d]+)\s*(?:张|幅|帧|个图|张图)")
# 声明/栏目里直接写了"第 5-6 张""第11-13张"时，那几帧才是它的证据来源
_FRAME_REF_RE = re.compile(r"第\s*([一二三四五六七八九十百\d]+)\s*(?:[-—~至]\s*([一二三四五六七八九十百\d]+)\s*)?张")
# 老工具（distill.step_vision）写的是 `--- 帧 3/18 ---`，没有"第"字；
# 认它就能让新证据层直接吃下老库里的 200+ 条 vlm.txt，并拿到正确的帧号→时间戳
_LEGACY_FRAME_RE = re.compile(r"帧\s*(\d+)\s*/\s*\d+")
_MD_MARK_RE = re.compile(r"[*_`]+")
_MIN_SEG_CHARS = 2

# L2 召回宽度与单条声明的证据预算（首轮基准 42.5% 的直接修正：见 retrieve 注释）
#
# top_k 4 -> 6（2026-10-14）：交付证据 +3%（75840 -> 78044 字符 / 39 条声明），
# 是三个可调参数里影响最小的一个——因为 EVIDENCE_BUDGET 已经卡住近半数声明
# （实测 19/39 条触顶）。**准确率上的效果无法测量**：同一条件重跑两遍，
# 25 条里就有 3~5 条判定漂移，噪声大于信号。故这只是一次方向性的小调整
# （多召回比少召回安全），不是已验证的改进；要下结论需新视频 + 人标真值。
DEFAULT_TOPK = 6
EVIDENCE_BUDGET = 1800
PIN_BUDGET_FACTOR = 2      # 钉住的证据可以超出常规预算到这个倍数，但也不是无限（防长笔记撑爆提示词）
PIN_MIN_CHARS = 1500       # 钉住证据的保底字数：预算被调得很小时，钉住的也不许被裁掉（钉了就得不掉）

_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def cn_number(s: str) -> int | None:
    """"十一"→11、"二十"→20、"3"→3。解析不了返回 None（不猜）。"""
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十") and len(s) == 2 and s[1] in _CN_DIGITS:
        return 10 + _CN_DIGITS[s[1]]
    if s.endswith("十") and len(s) == 2 and s[0] in _CN_DIGITS:
        return _CN_DIGITS[s[0]] * 10
    if len(s) == 1:
        return _CN_DIGITS.get(s)
    return None


_CN_SINGLE = "零一二三四五六七八九"


def _cn_int(n: int, liang: bool = False) -> str:
    """整数 → 中文念法。liang=True 时百/千位的 2 用"两"（口语"两百/两千"）。"""
    if n == 0:
        return ""
    if n < 10:
        return "两" if (liang and n == 2) else _CN_SINGLE[n]
    if n < 20:
        return "十" + (_CN_SINGLE[n - 10] if n > 10 else "")
    if n < 100:
        return f"{_CN_SINGLE[n // 10]}十{_CN_SINGLE[n % 10] if n % 10 else ''}"   # 20 念"二十"，不用"两十"
    if n < 1000:
        head = "两" if (liang and n // 100 == 2) else _CN_SINGLE[n // 100]
        return f"{head}百{_cn_int(n % 100)}"
    if n < 10000:
        head = "两" if (liang and n // 1000 == 2) else _CN_SINGLE[n // 1000]
        return f"{head}千{_cn_int(n % 1000)}"
    return str(n)


def _int_variants(n: int) -> set[str]:
    """整数的可能写法：阿拉伯 + 中文念法（含"两百/二百"这类等义写法）。"""
    out = {str(n)}
    if 0 < n < 10000:
        out.add(_cn_int(n))
        if 200 <= n < 300 or 2000 <= n < 3000:
            out.add(_cn_int(n, liang=True))
    return {s for s in out if s}


def num_variants(tok: str) -> set[str]:
    """一个数字的所有可能写法（阿拉伯/中文/带小数点）。用于"数字在不在原素材里"的判定。"""
    tok = (tok or "").strip()
    if not tok:
        return set()
    if "." in tok:
        a, b = tok.split(".", 1)
        out = {tok, f"{a}点{b}"}
        if a.isdigit() and b.isdigit() and len(a) == 1 and len(b) == 1:
            out.add(f"{_CN_SINGLE[int(a)]}点{_CN_SINGLE[int(b)]}")
        return out
    return _int_variants(int(tok)) if tok.isdigit() else {tok}


def tokenize(text: str) -> set[str]:
    """特征集：英文词（小写）+ 数字（# 前缀防与词混）+ 汉字 2-gram / 3-gram。"""
    out: set[str] = set()
    t = text or ""
    for w in _EN_RE.findall(t):
        if len(w) >= 2:
            out.add(w.lower())
    for n in _NUM_RE.findall(t):
        out.add("#" + n)
    for run in _CJK_RE.findall(t):
        if len(run) == 1:
            out.add(run)
            continue
        for i in range(len(run) - 1):
            out.add(run[i:i + 2])
        for i in range(len(run) - 2):
            out.add(run[i:i + 3])
    return out


def coverage(query: set[str], doc: set[str]) -> float:
    """声明特征被文档覆盖的比例。0 = 原文里几乎找不到字面支撑。"""
    if not query:
        return 0.0
    return len(query & doc) / len(query)


# ── 硬事实命中检测（确定性信号，不花 token） ────────────────
# 实测（2026-09-17 真实笔记）：纯 bigram 覆盖度太钝——中文常用 2-gram 满地都是，
# 导致"证据不相关"和"证据相关"分数差不多，逼着模型一律判 ⚠️ 存疑。
# 而**数字**是最硬的事实点：转写会把工具名听错（ZCode→dcode、Codex→codes），
# 但数字很少听错，所以"数字在原素材里 0 命中"≈ 这个数字是编的或抄错的。
_HARD_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_HARD_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.+-][A-Za-z0-9]+)*")
_EN_STOP = {"the", "and", "for", "app", "com", "www", "ok", "ai"}   # 太泛的词不给权重
# 中文数字的组成字符（数字边界判定用）
_CN_DIGIT_CHARS = "零一二三四五六七八九十百千万两"


def hard_facts(text: str) -> tuple[set[str], set[str]]:
    """拆出声明的硬事实点：(数字集合, 英文专名集合)。数字归一化去掉无意义前导零。

    **帧号不是事实数字**：画面小节常写「第 3 张图显示…」——那个 3 是画面序号不是声明内容，
    若把它当硬事实，会触发「数字在原素材里查不到」闸门，把本该 ✅ 的声明压成 ⚠️（实测踩到）。
    所以先剥掉「第 N 张/幅/帧/个图」里的帧号再提数字。
    """
    t = _FRAME_REF_RE.sub("", _IMG_ORD_RE.sub("", text or ""))
    nums = {n.lstrip("0") or "0" for n in _HARD_NUM_RE.findall(t)}
    names = {w.lower() for w in _HARD_EN_RE.findall(text or "")
             if len(w) >= 2 and w.lower() not in _EN_STOP}
    return nums, names


def _number_in_source(v: str, src: str) -> bool:
    """数字变体 v 是否在原素材 src 里出现——**带边界**，防「200」误中「2000/1200」。

    无边界时，声明里的 200 会被原素材里的 2000 当成「查得到」，编造数字这道闸就形同虚设。
    边界 = 前后都不是数字字符（阿拉伯 0-9 + 汉字数字），保证：
    - 阿拉伯 200 不命中 2000 / 1200 / 3.200
    - 中文「四十」不命中「四十万」，「二百」不命中「一千二百五」
    """
    if not v:
        return False
    pat = rf"(?<![0-9{_CN_DIGIT_CHARS}]){re.escape(v)}(?![0-9{_CN_DIGIT_CHARS}])"
    return re.search(pat, src) is not None


def hard_miss(text: str, source: str) -> tuple[list[str], list[str]]:
    """哪些硬事实点在**整条原素材**里一个都没出现。返回 (缺失数字, 缺失专名)。

    专名缺失只当提示（ASR 音译常出错），数字缺失才是强信号——由 L3 分别处置。
    数字判定走 num_variants：口播念"两百"而笔记写"200"不算缺失；
    且带边界匹配：200 与 2000 是两个不同的数，200 不能在 2000 里找到出处。
    """
    nums, names = hard_facts(text)
    src = (source or "").lower()
    miss_nums = sorted(n for n in nums
                       if not any(_number_in_source(v, src) for v in num_variants(n)))
    miss_names = sorted(w for w in names if w not in src)
    return miss_nums, miss_names


# 商业化/对外化措辞：笔记把它当事实写、原素材里却连这个词都没有时，就是"推理性改写"。
# 词表刻意收得**极窄**（高精度优先）：只收"对外提供/买卖关系"这一层意思的词。
# 故意不收「提供」「交付」「收费」「付费」——它们太日常：
#   · 「提供图形界面」是普通动词，会把大量正常声明误伤成存疑
#   · 「收费」「付费」在原素材里以别的意思出现过（"操作我已经下载了这款收费的阅读器"、
#     "用户付费的点"），单独出现不足以说明商业化
# 实测依据（2026-09-17，40 条基准）：只有 1 条声明命中本词表，而它正是那条唯一假阳性；
# 其余 39 条一次都没触发——精度 100%，代价 0。
_BIZ_TERMS = (
    "服务", "定制", "对外", "客户", "甲方", "接单", "承接", "外包", "代做",
    "报价", "变现", "商业模式", "定价", "售卖", "销售", "供给", "卖给", "收款",
)


def business_leap(text: str, source: str) -> list[str]:
    """声明里用了哪些"对外提供/买卖关系"措辞，而**整条原素材**里一个都没有。

    对应最容易被骗过的一类假 ✅：原素材说的是"我自己用它做这三件事"，
    笔记写成"可基于它提供这三项服务"。硬事实（工具名、场景名）全都在源里，
    所以逐条证据匹配看着完全支持——**错的是推理方向，不是事实**。
    """
    src = source or ""
    return sorted(t for t in _BIZ_TERMS if t in (text or "") and t not in src)


def frame_refs(text: str) -> set[int]:
    """声明里点名了哪些帧（"第 5-6 张""第11-13张""第五张"）→ 帧号集合。

    实测踩到（2026-09-17）：笔记的画面描述是 `第11-13张：Qoder IDE 选择 Computer Use 工具…`，
    但检索按字面重合度选段，喂的是**第 8 张图**（ZCode 的界面）——模型据此判"不符"。
    声明自己已经说明白了"看哪几帧"，检索就得听它的，别自作主张按字面分数挑。
    """
    out: set[int] = set()
    for m in _FRAME_REF_RE.finditer(text or ""):
        lo = cn_number(m.group(1))
        hi = cn_number(m.group(2)) if m.group(2) else lo
        if lo is None or hi is None:
            continue
        if hi < lo:
            lo, hi = hi, lo
        out |= set(range(lo, hi + 1))
    return out


@dataclass
class Segment:
    text: str
    ts: str = ""
    frame: int = 0            # 画面证据的帧序号（1-based），转写为 0
    tokens: set[str] = field(default_factory=set)

    def make(self) -> Segment:
        self.tokens = tokenize(self.text)
        return self


def parse_transcript(text: str) -> list[Segment]:
    """转写 → 按时间戳分段。无时间戳的行并入上一段（分段转写拼接后可能丢标签）。"""
    segs: list[Segment] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = _TS_LINE_RE.match(line)
        if m:
            segs.append(Segment(text=m.group(2).strip(), ts=m.group(1)).make())
        elif segs:
            segs[-1].text = f"{segs[-1].text} {line.strip()}".strip()
            segs[-1].make()
        else:
            segs.append(Segment(text=line.strip()).make())
    # 丢掉时间戳但没内容的空段
    return [s for s in segs if len(s.text) >= _MIN_SEG_CHARS]


def parse_vlm(text: str) -> list[Segment]:
    """画面理解 → 按空行分块；块内的"第 N 张图"决定它对应第几帧。

    实测格式（真机 vlm.txt）：`**第一张图**` 常常自己独占一个块，正文在下一块——
    所以只含序号的短块必须与其后一块合并，否则"第 N 张图"会变成一条没有内容的假证据。

    **帧号要「继承」而不是「顺序补齐」**（2026-09-17 实测踩到）：一帧的正文自己就含空行，
    会被切成好几块；只有第一块带标记，后面几块若按顺序编号，就会把第 1 帧的内容
    标成第 2、3、4 帧 —— 画面证据的时间戳会全错。所以块内没有明确序号时，
    沿用上一个明确序号；整份文件都没有序号（老格式）时才退回顺序编号。
    """
    raw = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    segs: list[Segment] = []
    pending: int | None = None      # 上一个"只有序号"的标题块下标
    current: int | None = None      # 最近一次明确出现的帧号（供后续无标记块继承）
    for b in raw:
        flat = _MD_MARK_RE.sub("", b).strip()
        m = _IMG_ORD_RE.search(flat) or _LEGACY_FRAME_RE.search(flat)
        n = cn_number(m.group(1)) if m else None
        if n:
            current = n
        if m and len(flat) <= 12:                       # 纯标题块：先占位，等正文
            segs.append(Segment(text=b, frame=n or 0))
            pending = len(segs) - 1
            continue
        if pending is not None:                          # 正文块：并进上一个标题块
            segs[pending].text = f"{segs[pending].text}\n{b}"
            pending = None
            continue
        segs.append(Segment(text=b, frame=n or current or 0))
    for i, s in enumerate(segs, 1):
        if not s.frame:
            s.frame = i
        s.make()
    return [s for s in segs if len(s.text) >= _MIN_SEG_CHARS]


def frame_timecodes(frames_dir: Path) -> dict[str, float]:
    p = Path(frames_dir) / "timecodes.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in dict(data).items()}
    except (json.JSONDecodeError, TypeError, ValueError, OSError):
        return {}


def fmt_ts(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


@dataclass
class SourceIndex:
    """一条记录的可检索素材。构建一次，L2 复用（同一篇里几十条声明不必重复解析）。"""
    transcript: list[Segment] = field(default_factory=list)
    vlm: list[Segment] = field(default_factory=list)
    frame_ts: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, transcript_text: str, vlm_text: str, frames_dir: Path | None = None) -> SourceIndex:
        return cls(transcript=parse_transcript(transcript_text),
                   vlm=parse_vlm(vlm_text),
                   frame_ts=frame_timecodes(frames_dir) if frames_dir else {})

    @property
    def empty(self) -> bool:
        return not self.transcript and not self.vlm

    def full_text(self) -> str:
        """全素材拼文本：硬事实命中检测的判据是"整条视频里有没有出现过"，不分段。"""
        return "\n".join(s.text for s in self.transcript) + "\n" + \
               "\n".join(s.text for s in self.vlm)

    def _vlm_ts(self, frame: int) -> str:
        if not self.frame_ts:
            return ""
        names = sorted(self.frame_ts)
        if 1 <= frame <= len(names):
            return fmt_ts(self.frame_ts[names[frame - 1]])
        return ""

    def retrieve(self, claim_text: str, top_k: int = DEFAULT_TOPK, pin: bool = True,
                 budget: int = EVIDENCE_BUDGET, window: int = 2,
                 context: str = "") -> tuple[float, list[Evidence]]:
        """回查单条声明：打分选段 → 邻域补全 → 硬事实/帧号钉住 → 相邻合并 → 按预算裁剪。

        参数各自修一个实测出来的毛病（2026-09-17，见 scripts/recall_audit.py）：
        - `context`：声明所在的**栏目名**（"测试任务"/"会占用用户鼠标"/"第5-6张"）。
          它既是检索线索也是限定语境，必须一起参与打分（详见 Claim.context 的注释）。
        - `window`：**笔记的要点常由好几句话拼成**（如"设置里找到电脑控制→勾选→授权屏幕录制权限"
          跨转写 04:16/04:24/04:28 三行）。这种声明对**任何单段**的覆盖度都不高，
          按覆盖度选段会一段都选不中 → 交付的证据集覆盖不到声明。
          所以选中最高分那段后，把它的前后各 window 段一并纳入，再做合并。
        - `pin`：含声明硬事实的段必须入选（哪怕覆盖度排不进前几）——判据是"原素材里有没有这个数字/专名"。
        - `budget`：单条声明的证据字数上限，防止扩召回把提示词撑爆。
        """
        query = f"{context} {claim_text}".strip() if context else claim_text
        q = tokenize(query)
        if not q:
            return 0.0, []
        nums, names = hard_facts(query)
        variants = {v.lower() for n in nums for v in num_variants(n)} | {n.lower() for n in names}
        refs = frame_refs(query)     # 声明点名了"看哪几帧"时，那几帧是硬性证据

        picked: list[Evidence] = []
        for segs, source in ((self.transcript, "transcript"), (self.vlm, "vlm")):
            if not segs:
                continue
            scored: list[tuple[float, float, float, int, Segment]] = []
            for i, seg in enumerate(segs):
                sc = coverage(q, seg.tokens)
                if variants:
                    low = seg.text.lower()
                    hf = sum(1 for v in variants if v and v in low) / len(variants)
                else:
                    hf = 0.0
                score = 0.6 * sc + 0.4 * hf          # 硬事实优先于"通顺的常用词重合"
                if sc > 0 or hf > 0:
                    scored.append((score, sc, hf, i, seg))
            if not scored:
                continue
            ranked = sorted(scored, key=lambda t: (-t[0], t[3]))
            # ── 贪心「覆盖增益」选段，替代「按单段分数取前 top_k」──
            # 单段分数只看这一段盖住声明的多少，不看它跟已选段是否重复：概括类声明的
            # 支撑常摊在好几段、每段单看分都不高，取前几名会拿到一堆互相重复的段，
            # 声明的其余部分一段都盖不住。改成按「新增覆盖的声明特征数」贪心选，
            # 直接对齐交付覆盖度这个目标。实测（2026-09-29，346 条已标注基准）：
            # 过度存疑组交付覆盖度 0.352 vs 全素材上界 0.449，缺口均值 0.097、最大 0.53。
            # 锚点固定是分最高的那段：返回值 top_cov（单段最高覆盖度）是报告与
            # 深度模式阈值读的口径，不能因选段方式改变。
            chosen: set[int] = set()
            covered: set[str] = set()
            hit_var: set[str] = set()
            anchor = ranked[0][3]
            chosen.add(anchor)
            covered |= segs[anchor].tokens
            if variants:
                hit_var = {v for v in variants if v and v in segs[anchor].text.lower()}
            while len(chosen) < top_k:
                best_i, best_gain, best_new_var = -1, 0, set()
                # ranked 顺序 = 分高→低、平手取段号小；增益严格更大才替换，保证确定性
                for _, _, _, i, seg in ranked:
                    if i in chosen:
                        continue
                    gain = len((q & seg.tokens) - covered)
                    new_var = ({v for v in variants if v and v in seg.text.lower()}
                               - hit_var) if variants else set()
                    if gain + len(new_var) > best_gain:
                        best_i = i
                        best_gain = gain + len(new_var)
                        best_new_var = new_var
                if best_i < 0:        # 没有任何段还能带来新增覆盖 → 不硬凑
                    break
                chosen.add(best_i)
                covered |= segs[best_i].tokens
                hit_var |= best_new_var
            # 钉住的段：含全部硬事实的，或声明点名的那几帧
            pin_idx = {t[3] for t in scored if t[2] >= 1.0} if pin else set()
            if refs and source == "vlm":
                # 声明点名了"第 N 张"：那几帧是它自己的证据来源，按字面分数挑会挑错帧
                pin_idx |= {i for i, seg in enumerate(segs) if seg.frame in refs}
            chosen |= pin_idx
            if window > 0:
                chosen |= {i for i in range(anchor - window, anchor + window + 1)
                           if 0 <= i < len(segs)}

            # 相邻段合并：同一段话被时间戳切碎时，报告里要给完整上下文
            groups: list[list[int]] = []
            for i in sorted(chosen):
                if groups and i - groups[-1][-1] <= 1:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            cand: list[Evidence] = []
            for g in groups:
                cov = max((t[1] for t in scored if t[3] in g), default=0.0)
                hf = max((t[2] for t in scored if t[3] in g), default=0.0)
                text = " ".join(segs[i].text for i in g)
                ts = segs[g[0]].ts or (self._vlm_ts(segs[g[0]].frame) if source == "vlm" else "")
                evi_type = _classify_evidence_type(text, source, ts)
                cand.append(Evidence(source=source, text=text, ts=ts,
                                       score=round(cov, 4), fact_hit=round(hf, 4),
                                       pinned=bool(set(g) & pin_idx),
                                       evi_type=evi_type))
            # 强制包含开头证据（2026-10-16 改进）：如果存在前 30 秒的转写证据，至少保留一条
            opening_evi = [e for e in cand if e.evi_type == 'opening']
            if opening_evi and not any(e.pinned for e in opening_evi):
                # 取得分最高的一条开头证据，标记为 pinned
                opening_evi[0].pinned = True

              # 预算裁剪：先留钉住的，再按硬事实命中率/覆盖度补
            # （钉住的段必须**豁免**预算——否则"钉住"只是句空话：它们 fact_hit 可能为 0、
            #   按分数排在最后，一超预算就先被裁掉。实测踩到，见 DECISIONS D10）
            cand.sort(key=lambda e: (not e.pinned, -e.fact_hit, -e.score))
            pin_cap = max(budget * PIN_BUDGET_FACTOR, PIN_MIN_CHARS)
            kept, used, pin_used = [], 0, 0
            for e in cand:
                if e.pinned:
                    if pin_used + len(e.text) <= pin_cap:
                        kept.append(e)
                        pin_used += len(e.text)
                        used += len(e.text)
                    continue
                if kept and used + len(e.text) > budget:
                    continue
                kept.append(e)
                used += len(e.text)
            picked += kept

        picked.sort(key=lambda e: (-e.fact_hit, -e.score))
        top_cov = max((e.score for e in picked), default=0.0)
        return top_cov, picked


def retrieve_all(claims, index: SourceIndex, top_k: int = DEFAULT_TOPK, on_step=None) -> None:
    """就地给每条声明挂证据、分数与硬事实缺失清单（L3 直接用）。"""
    source = index.full_text()
    for c in claims:
        score, ev = index.retrieve(c.text, top_k=top_k, context=c.context)
        c.score = round(score, 4)
        c.evidence = ev
        c.miss_numbers, c.miss_names = hard_miss(c.text, source)
        c.miss_biz = business_leap(c.text, source)
    if on_step:
        hit = sum(1 for c in claims if c.evidence)
        pinned = sum(1 for c in claims if any(e.fact_hit >= 1.0 for e in c.evidence))
        miss = sum(1 for c in claims if c.miss_numbers)
        biz = sum(1 for c in claims if c.miss_biz)
        on_step(f"   L2 回查：{hit}/{len(claims)} 条找到字面证据（其中 {pinned} 条拿到"
                f"「含全部硬事实」的决定性片段）；{miss} 条含「原素材里查不到的数字」"
                f"{f'；{biz} 条含「原素材没有的对外/商业措辞」' if biz else ''}")
