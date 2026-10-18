"""核对引擎数据模型（note_check 记录的代码化）。

一处真相源：L1/L2/L3/L4 与报告渲染全部只用这里的结构，避免各层各写一套 dict。
判定只有三种取值，且**默认值必须是存疑而不是一致**——
"没核到"和"核过了一致"是两回事，默认 ✅ 等于把没核的当核过的（本工具最怕的假阳性）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

VERDICT_OK = "ok"            # ✅ 一致：能在原素材里找到出处
VERDICT_MISMATCH = "mismatch"  # ❌ 不一致：原素材里没有，或说反了
VERDICT_DOUBT = "doubt"      # ⚠️ 存疑：证据不足/互相矛盾，保留但标注

VERDICT_MARK = {VERDICT_OK: "✅", VERDICT_MISMATCH: "❌", VERDICT_DOUBT: "⚠️"}
VERDICT_LABEL = {VERDICT_OK: "一致", VERDICT_MISMATCH: "不一致", VERDICT_DOUBT: "存疑"}

# 声明类型：决定快速模式核哪些、深度模式看多严
KIND_NUMBER = "number"    # 数字/金额/时长/比例
KIND_ENTITY = "entity"    # 专有名词/工具名/模型名/平台名
KIND_STEP = "step"        # 可照做的步骤/指令/配置
KIND_QUOTE = "quote"      # 引用原话/他人观点转述
KIND_OTHER = "other"      # 其余事实性表述


@dataclass
class Evidence:
    """一条回查到的证据。ts 为原片时间戳（如 "01:23"），画面证据来自 timecodes.json。"""
    source: str            # transcript | vlm
    text: str
    ts: str = ""
    score: float = 0.0
    fact_hit: float = 0.0  # 该段命中了声明里多少比例的硬事实（1.0 = 决定性片段，优先给模型看）
    pinned: bool = False   # 钉住（硬事实全中 / 声明点名的帧）：不受字数预算裁剪
    evi_type: str = ""     # 证据类型：opening（视频开头定位）| contrast（对比结构）| spoken（口语表达）| visual（画面精确数据）| neutral（普通证据）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    """一条可独立验证的原子声明（L1 产物），后面被 L2/L3 逐层填上判定。"""
    cid: str                       # C01 / C02 ...（报告内可引用）
    text: str                      # 消歧后的自足句（"它/这个"已补全指代）
    section: str = ""              # 笔记里的章节名，报告按它分组
    kind: str = KIND_OTHER
    original: str = ""             # 笔记里的原句（用于"把 ❌ 改回原素材说法"的精确替换）
    verdict: str = ""              # 空 = 尚未判定（绝不预设 ok）
    reason: str = ""
    fix: str = ""                  # 修正建议（❌/⚠️ 才有）
    evidence: list[Evidence] = field(default_factory=list)
    score: float = 0.0             # L2 最强证据的覆盖度
    miss_numbers: list[str] = field(default_factory=list)   # 原素材里 0 命中的数字（强信号）
    miss_names: list[str] = field(default_factory=list)     # 0 命中的英文专名（弱信号，ASR 会听错）
    miss_biz: list[str] = field(default_factory=list)       # 0 命中的"对外提供/买卖"措辞（推理性改写）
    failed: bool = False           # 该条所在的判定批次调用失败（限流/网络）→ 可 --retry-failed 重判

    @property
    def ts(self) -> str:
        return self.evidence[0].ts if self.evidence else ""

    @property
    def context(self) -> str:
        """笔记里这条声明所在的**栏目名**（`section` 的最后一段，如"测试任务"/"会占用用户鼠标"/"第5-6张"）。

        为什么必须有（2026-09-17 实测踩到）：笔记写的是
        `- **测试任务**：找到下载目标栏并上传 pio 文件`，L1 把栏目名剥掉后只剩后半句，
        模型就把它读成"断言上传成功了"，于是对着"上传失败"的原素材判 mismatch —— 一条假 ❌。
        栏目名是这句声明的**限定语境**，检索和判定都得带上。
        """
        sec = self.section or ""
        return sec.rsplit("/", 1)[-1].strip() if "/" in sec else ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mark"] = VERDICT_MARK.get(self.verdict, "")
        return d


@dataclass
class CheckResult:
    """一次核对的全部产出，落盘即note_check 记录。"""
    key: str
    mode: str
    claims: list[Claim] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # 主动跳过的句子（观点/情绪/无项）
    created_at: str = ""
    provider: str = ""

    def count(self, verdict: str) -> int:
        return sum(1 for c in self.claims if c.verdict == verdict)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def status(self) -> str:
        """checked = 全部有判定；partial = 部分未判定；unchecked = 无声明或被中断。"""
        if not self.claims:
            return "unchecked"
        if all(c.verdict for c in self.claims):
            return "checked"
        return "partial"
