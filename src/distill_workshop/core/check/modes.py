"""三种核对模式（吸收 guard 的 mode 概念，自研参数化）。

- 快速：只核数字/专名/引用三类硬事实，省 token，用于批量与低价值视频
- 标准（默认）：全部原子声明逐条核对
- 深度：标准 + 双信号（LLM 判定 ∧ L2 证据覆盖度达标），任一不满足降级为 ⚠️ 存疑

双信号为什么是"覆盖度"而不是 embedding：本阶段不引入 embedding，
而 L2 的字面覆盖度是**同素材内部**的确定性指标，不额外花钱、可离线、结果可复现——
它不能替代 embedding，但足以承担"第二只眼睛"的职责：LLM 说一致、原文却几乎找不到字面支撑时，
这条就该标存疑，而不是随 LLM 一路 ✅。
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import KIND_ENTITY, KIND_NUMBER, KIND_QUOTE

MODES: dict[str, CheckMode] = {}

# batch_size 是**质量参数，不是速度旋钮**——实测（2026-10-13，同一篇 120 条声明、
# 同一批证据，只改批量）：
#   批量 5  -> 900s，✅101 ⚠️18 ❌1
#   批量 15 -> 593s，✅55  ⚠️65 ❌0   ← 快 34%，但 ✅ 掉了 46 条
# 同批声明越多，模型越倾向判"存疑"（信息过载 + 批内互相干扰）。
# 别为了提速调大它：质量掉得比时间省得多。


@dataclass(frozen=True)
class CheckMode:
    name: str
    label: str
    batch_size: int          # 每批判定的声明条数（批量省 token，且同批可互相参照）
    kinds: tuple[str, ...]   # 只核这些类型；空元组 = 全部
    double_signal: bool      # 是否启用双信号
    min_score: float         # 双信号要求的 L2 覆盖度下限
    max_claims: int          # 单篇最多核对条数（防长笔记把 token 打爆）

    def __post_init__(self) -> None:
        # setdefault 而非赋值：`dataclasses.replace()` 派生出的副本不该覆盖全局表
        # （否则按 config 调大 max_claims 会污染后续所有调用）
        MODES.setdefault(self.name, self)


CheckMode("fast", "快速", batch_size=8,
          kinds=(KIND_NUMBER, KIND_ENTITY, KIND_QUOTE),
          double_signal=False, min_score=0.0, max_claims=40)
CheckMode("standard", "标准", batch_size=5,
          kinds=(), double_signal=False, min_score=0.0, max_claims=120)
CheckMode("deep", "深度", batch_size=5,
          kinds=(), double_signal=True, min_score=0.18, max_claims=120)

DEFAULT_MODE = "standard"


def get_mode(name: str | None) -> CheckMode:
    """取模式；未知名字直接报错而不是悄悄退回默认——配错模式会静默改变核对强度。"""
    key = (name or DEFAULT_MODE).strip().lower()
    if key not in MODES:
        raise ValueError(f"未知核对模式: {name}（可选：{'/'.join(MODES)}）")
    return MODES[key]
