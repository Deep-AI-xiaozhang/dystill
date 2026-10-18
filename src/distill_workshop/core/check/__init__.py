"""核对引擎（产品灵魂）。

四层：L1 原子声明拆分 → L2 证据回查 → L3 逐条判定 → L4 报告与徽章。
对外只暴露 run_check 与数据模型；四层可单测、可替换（模型调用一律走注入的 chat_fn）。
"""
from .engine import analyze, find_note, resolve_record_dir, run_check
from .models import (
    VERDICT_DOUBT,
    VERDICT_LABEL,
    VERDICT_MARK,
    VERDICT_MISMATCH,
    VERDICT_OK,
    CheckResult,
    Claim,
    Evidence,
)
from .modes import MODES, CheckMode, get_mode
from .report import BOUNDARY, badge, build_json, render_markdown, write_reports

__all__ = [
    "BOUNDARY",
    "MODES",
    "VERDICT_DOUBT",
    "VERDICT_LABEL",
    "VERDICT_MARK",
    "VERDICT_MISMATCH",
    "VERDICT_OK",
    "CheckMode",
    "CheckResult",
    "Claim",
    "Evidence",
    "analyze",
    "badge",
    "build_json",
    "find_note",
    "get_mode",
    "render_markdown",
    "resolve_record_dir",
    "run_check",
    "write_reports",
]
