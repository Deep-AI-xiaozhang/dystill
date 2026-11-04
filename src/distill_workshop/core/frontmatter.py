"""frontmatter 标量转义（笔记层与知识库层共用）。

为什么单独一个模块：`notes.py` 写笔记、`kb.py` 补字段，两边都要把**模型产出的文字**
写进 YAML frontmatter。放在任一方的模块里，另一方就得反向依赖（kb → notes →
providers.llm），纯为借一个纯函数不值当。
"""
from __future__ import annotations

import re

# YAML 指示符：这些字符开头的裸标量会被解释成注释/序列/锚点/标签，直接解析失败或变义
_INDICATORS = r"[-?:,\[\]{}#&*!|>'\"%@`]"
_NUMBER_RE = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?", re.IGNORECASE)
_RESERVED = {"true", "false", "yes", "no", "on", "off", "null", "~"}


def yaml_scalar(value: str) -> str:
    """把任意字符串写成 frontmatter 里**安全**的单行 YAML 标量（必要时加引号）。

    为什么必须做（实测）：抖音标题带 `#话题` 是常态，`title: #AI新星计划 …` 会被 YAML
    当成注释 → Obsidian/解析器读到的标题是 None；`- 破折号开头` 直接 ScannerError；
    纯数字标题会被转成 int。这些都不会在生成时暴露，只在别的工具读笔记时才炸。

    **只在真会歧义时才加引号**：绝大多数中文标题原样写更好读、diff 也更稳，
    所以能裸写就裸写（指示符开头、": " 冒号空格、行内 " #"、纯数字/布尔/null 才加）。
    """
    v = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    needs_quote = bool(
        not v
        or re.match(_INDICATORS, v)          # 指示符开头
        or re.search(r":\s", v)              # "a: b" 会被当成映射
        or re.search(r"\s#", v)              # 行内注释起点
        or v.lower() in _RESERVED
        or _NUMBER_RE.fullmatch(v)           # 纯数字会被转成 int/float
    )
    if not needs_quote:
        return v
    # 单引号最省事：只有 ' 需要转义，且不会像双引号那样把 \n 之类当转义序列解释。
    return "'" + v.replace("'", "''") + "'"


def unquote_scalar(raw: str) -> str:
    """读回 frontmatter 标量：剥掉 yaml_scalar 加的单引号（'It''s' → It's）。

    读取端必须和写入端配套：`title: '#AI新星计划 …'` 若只 strip("'")，开头那个引号没了、
    结尾的也一并吃掉，正文里带引号的标题会被削掉一半。
    """
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s
