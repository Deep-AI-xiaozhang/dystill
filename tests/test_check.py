"""核对引擎测试（F2 / M1）。

重点覆盖三件最容易出事故的事：
1. 该跳过的句子有没有被跳过（观点/概括/"无"项被核进报告 = 噪声，用户会反感）
2. 判定闸门方向对不对（零证据、低覆盖度只能把 ✅ 往下压，绝不能把 ⚠️ 抬成 ✅）
3. 模型失败时不许"看起来成功了"（全批记存疑 + 报告照出）
"""
import json
import re
from pathlib import Path

import pytest

from distill_workshop.config import AppConfig
from distill_workshop.core.check import (
    badge,
    build_json,
    get_mode,
    render_markdown,
    run_check,
)
from distill_workshop.core.check.claim_extractor import (
    classify_kind,
    extract_by_rules,
    extract_claims,
    is_checkable,
    parse_disambiguation,
    split_sentences,
)
from distill_workshop.core.check.engine import resolve_record_dir
from distill_workshop.core.check.evidence_retriever import (
    SourceIndex,
    cn_number,
    coverage,
    hard_facts,
    hard_miss,
    num_variants,
    parse_transcript,
    parse_vlm,
    tokenize,
)
from distill_workshop.core.check.models import (
    KIND_ENTITY,
    KIND_NUMBER,
    KIND_STEP,
    VERDICT_DOUBT,
    VERDICT_MISMATCH,
    VERDICT_OK,
    CheckResult,
    Claim,
    Evidence,
)
from distill_workshop.core.check.report import apply_fixes, inject_badge
from distill_workshop.core.check.verifier import (
    apply_gates,
    build_items,
    parse_verdicts,
    verify_claims,
)

NOTE = """---
title: 测试视频
key: aweme:1
---

# 蒸馏笔记 · 测试视频

> 蒸馏日期：2026-09-17 ｜ 类型：实操教程 ｜ 主题：四款 AI 工具 Computer Use 功能横评

## 一、结论先行

视频对比了 4 款 AI 工具的 Computer Use 功能，其中 Codex 操作最精准。
个人觉得这类工具值得大家优先尝试。

## 二、视频内容

- 口播要点：作者用统一测试对象横向对比了四款工具的实际表现
- 画面/操作要点：屏幕上出现任务列表和设置面板

## 三、知识要点（核心）

- **关键概念**：Computer Use 是让 AI 直接操控电脑完成操作的功能
- **实操步骤**：先打开 Lumen Reader，再点击设置里的连接按钮完成授权
- **工具与资源**：视频评测了 Codex、ZCode、Qoder、Orca 四款工具
- **对比与数据**：Codex 有独立鼠标指针，操作精准度明显更高
- **注意事项**：无

## 四、总结与适用场景

适合做竞品调研和软件测试的人使用，也可以用于办公自动化。
"""

TRANSCRIPT = """[00:00]来讲 computer use这个功能 很多AI工具都有了 本期视频测评我常用的四款AI工具
[00:12]Codex 的体验是最好的 有独立的鼠标指针 操作非常精准
[00:25]ZCode 虽然有小窗口 但看不到鼠标 准确度比较低
[00:40]这些工具也能用来做竞品调研、软件测试和办公自动化
[00:52]先打开 Lumen Reader 再点击设置里的连接按钮完成授权
"""

VLM = """第一张图

画面文字：是什么？正文提到 Computer Use 是让 AI 操控电脑的功能。

第二张图

画面显示任务列表和设置面板，标题栏写着 Codex。
"""


def make_app(tmp_path: Path) -> AppConfig:
    return AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                     data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")


AWEME = "7678251600959376681"


def make_record(tmp_path: Path, note: str = NOTE) -> Path:
    rd = tmp_path / "data" / f"aweme_{AWEME}"
    rd.mkdir(parents=True)
    (rd / "note.md").write_text(note, encoding="utf-8")
    (rd / "transcript.txt").write_text(TRANSCRIPT, encoding="utf-8")
    (rd / "vlm.txt").write_text(VLM, encoding="utf-8")
    (rd / "meta.json").write_text(json.dumps({"key": f"aweme:{AWEME}", "source": "douyin"}),
                                  encoding="utf-8")
    frames = rd / "frames"
    frames.mkdir()
    (frames / "timecodes.json").write_text(json.dumps({"001.jpg": 0.0, "002.jpg": 12.0}), encoding="utf-8")
    return rd


class _Resp:
    def __init__(self, content, provider="fake"):
        self.content = content
        self.provider = provider


def fake_chat(messages, **_kw):
    """按提示词类型分流：L1 消歧返回原样，L3 判定一律说 ok（用于验证闸门真的在拦）。"""
    prompt = messages[0]["content"]
    if "待核对声明：" in prompt:
        tail = prompt.rsplit("待核对声明：", 1)[-1]
        n = len(re.findall(r"^\d+\. \[", tail, re.MULTILINE))
        return _Resp(json.dumps([{"i": i, "text": f"消歧后的声明 {i}"} for i in range(1, n + 1)],
                                ensure_ascii=False))
    if "事实核对员" in prompt:
        n = prompt.count("声明：")
        return _Resp(json.dumps([{"i": i, "verdict": "ok", "reason": "证据支持", "fix": ""}
                                 for i in range(1, n + 1)], ensure_ascii=False))
    return _Resp("[]")


# ── L1 ────────────────────────────────────────────────
def test_kind_classification():
    assert classify_kind("月费 200 元") == KIND_NUMBER
    assert classify_kind("视频评测了 Codex 和 Qoder") == KIND_ENTITY
    assert classify_kind("先打开设置再点击连接按钮") == KIND_STEP


def test_split_sentences_merges_short_fragments():
    out = split_sentences("步骤：\n点击设置\n再点击连接按钮完成授权")
    assert out and all(len(s) >= 6 for s in out)


def test_split_sentences_handles_markdown_tables():
    md = """| 场景 | 说明 |
|---|---|
| 竞品调研 | 用截图分析竞品页面 |
| 办公自动化 | 每天定时查看数据 |
"""
    out = split_sentences(md)
    assert not any(set(s) <= set("|-: \t") for s in out)      # 分隔行不再是声明
    assert any("竞品调研" in s for s in out)
    assert any("办公自动化" in s for s in out)
    assert all("|" not in s for s in out)                     # 单元格已拆开，管道符不进声明


def test_split_sentences_table_rows_do_not_merge_across_rows():
    """跨行合并真 bug（2026-09-30 实测）：上一行末尾的短格并进下一行首格，
    造出笔记里不存在的「南北面官制，西夏」跨政权假声明 → 引擎对着板书判假 ❌。"""
    md = """| 辽 | 916年 | 上京 | 南北面官制 |
| 西夏 | 1038年 | 兴庆府 | 仿宋制 |
| 金 | 1115年 | 会宁府 | 猛安谋克制 |
"""
    out = split_sentences(md)
    assert not any("南北面官制，西夏" in s for s in out)
    assert not any("仿宋制，金" in s for s in out)
    assert not any("猛安谋克制，元" in s for s in out)
    # 行内短格合并保留：行自己的配对（如「西夏，1038年」）是合法声明
    assert any("西夏，1038年" in s or s == "西夏，1038年" for s in out)


def test_split_sentences_table_row_end_does_not_merge_forward():
    """D18 的边界只堵了单向（2026-10-04 复查发现）：表格行**末格**还会被下一行并进来，
    同型假声明换个方向又来（`| 辽 | 契丹 |` + `元昊建立西夏` → 「辽，契丹，元昊建立西夏」）。"""
    out = split_sentences("| 辽 | 契丹 |\n元昊建立西夏。")
    assert not any("契丹，元昊" in s for s in out), out
    assert "元昊建立西夏" in out
    # 列表行同型（表格行末尾的短格也不许并进下一行的列表项）
    out2 = split_sentences("| 辽 | 契丹 |\n- 元昊建立西夏")
    assert not any("契丹，元昊" in s for s in out2), out2


def test_split_sentences_skips_table_header_row():
    """表头行（后面紧跟 `|---|` 分隔行）是列名导航不是断言，整行丢弃。
    实测（2026-10-04）：公布验收集里 3 条伪声明（政权，建立时间 / 民族，建立者 /
    定都，核心制度）就是表头行被格间合并产出的。"""
    md = ("| 政权 | 建立时间 | 民族 | 建立者 | 定都 | 核心制度 |\n"
          "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
          "| 辽 | 916年 | 契丹族 | 耶律阿保机 | 上京 | 南北面官制 |\n")
    out = split_sentences(md)
    for junk in ("建立时间", "建立者", "核心制度"):
        assert not any(junk in s for s in out), out
    assert any("916年" in s for s in out)          # 数据行照旧核


def test_is_checkable_skips_opinion_and_none():
    assert is_checkable("无")[0] is False
    ok, why = is_checkable("个人觉得这类工具值得大家优先尝试")
    assert ok is False and "观点" in why
    # 带硬事实点的句子即使夹着"推荐"也必须核（可能推荐错了工具名）
    assert is_checkable("推荐 Codex 和 ZCode 这两款工具")[0] is True


def test_is_checkable_skips_bracket_section_titles():
    """【】小节标题基准实测产假 ✅（2026-09-30 D19）：标题词总能"在视频里对上"。"""
    for title in ("【关于男女思维差异的底层逻辑】", "”，【关于婚后沟通红线】",
                  "【关于具体问题的应对话术（实操部分）】"):
        ok, why = is_checkable(title)
        assert ok is False and "小节标题" in why, title
    # 【】只是引用记号、括号外有实质内容的声明必须照常核
    assert is_checkable("视频中提到【猛安谋克制】是金国制度")[0] is True


def test_extract_by_rules_skips_meta_and_summary_generalization():
    mode = get_mode("standard")
    claims, skipped = extract_by_rules(NOTE, mode)
    texts = [c.text for c in claims]
    assert any("Codex" in t for t in texts)
    assert not any("蒸馏日期" in t for t in texts)          # 元信息不核
    assert not any("值得大家优先尝试" in t for t in texts)   # 观点句不核
    assert not any(t.strip().startswith("无") for t in texts)
    assert any("概括句" in s for s in skipped)              # 跳过原因是透明的


def test_extract_claims_falls_back_when_model_broken():
    """消歧调用炸了也要给出声明列表（绝不空手）。"""
    def boom(_messages, **_kw):
        raise RuntimeError("503")

    claims, _ = extract_claims(NOTE, get_mode("fast"), chat_fn=boom, on_step=None)
    assert claims and all(c.cid for c in claims)


def test_parse_disambiguation_ignores_bad_rows():
    raw = '前言 ```json\n[{"i":1,"text":"有效声明内容"},{"i":99,"text":"越界"},{"junk":1}]\n```'
    assert parse_disambiguation(raw, 3) == {1: ["有效声明内容"]}


def test_fast_mode_keeps_only_hard_facts():
    fast, std = get_mode("fast"), get_mode("standard")
    f_claims, _ = extract_by_rules(NOTE, fast)
    s_claims, _ = extract_by_rules(NOTE, std)
    assert len(f_claims) < len(s_claims)
    assert all(c.kind in fast.kinds for c in f_claims)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        get_mode("quick")


def test_recheck_ignores_previous_badge_block():
    """第二次核对时，上一轮写进去的徽章块与 HTML 注释不能变成声明（实测踩到过）。"""
    from distill_workshop.core.check.report import inject_badge

    r = CheckResult(key="k", mode="standard", created_at="2026-09-17 10:00:00",
                    claims=[Claim("C01", "a", verdict=VERDICT_OK)])
    checked = inject_badge(NOTE, r)
    claims, _ = extract_by_rules(checked, get_mode("standard"))
    texts = [c.text for c in claims]
    assert not any("dystill-check" in t for t in texts)
    assert not any(t.strip() in ("<", ">") for t in texts)
    assert len(texts) == len(extract_by_rules(NOTE, get_mode("standard"))[0])


def test_section_name_has_no_heading_marks():
    claims, _ = extract_by_rules(NOTE, get_mode("standard"))
    assert claims
    for c in claims:
        assert not c.section.startswith("#"), c.section
    assert any(c.section.startswith("三、知识要点") for c in claims)


def test_symbol_only_fragments_rejected():
    from distill_workshop.core.check.claim_extractor import is_checkable
    for junk in ("|------|------|", "<!-- x -->", "---|---", "> < >", "==="):
        ok, why = is_checkable(junk)
        assert ok is False, junk
        assert "符号" in why or "过短" in why or "无内容" in why
    # 反过来：带真实内容的行不能被误杀
    assert is_checkable("视频对比了 4 款工具的 Computer Use 功能")[0] is True


def test_section_label_lines_are_not_claims():
    """栏目标记（`- **[02:04] WorkBuddy（腾讯）**：`）不是声明。

    原实现在剥完标签后兜底 `body = sent`，每条栏目标记于是都变成一条待核对声明；
    时间码还会在它内部第一个冒号处被 _TRAILING_TAG_RE 咬断成 `04] WorkBuddy（腾讯）：`，
    含 2 位数字被判成「数字类」、连观点过滤都绕过（035328 实测 14 条，其中 1 条被判 ✅）。
    """
    note = ("## 二、视频内容\n\n### 口播要点（按时间顺序）\n"
            "- **[00:10] 概念区分**：\n"
            "    - Agent = 脑子 + 手 + 脚。\n"
            "- **[02:04] WorkBuddy（腾讯）**：\n"
            "    - 优势1：上手门槛低，像正常软件一样使用。\n"
            "- **[00:00] 开场**：画面展示6个App图标。\n")
    claims, skipped = extract_by_rules(note, get_mode("standard"))
    texts = [c.text for c in claims]
    assert texts == ["Agent = 脑子 + 手 + 脚", "上手门槛低，像正常软件一样使用", "画面展示6个App图标"]
    # 时间码与被咬断的残片都不许进声明正文（`04] …` 就是旧实现的指纹）
    assert not any("[02:04]" in t or "04]" in t or "[00:10]" in t or "10]" in t for t in texts)
    assert any("栏目标记" in s for s in skipped)
    # 标记要当上下文用：带正文的那条拿到标签 `开场`（旧实现拿到的是被咬断的 `[02`）
    assert [c.context for c in claims] == ["", "优势1", "开场"]


def test_nested_label_chain_is_a_label():
    """相邻标签行会被短句合并成 `口播要点：，用户分类与对策：`——整行只是标签链，同样不是声明。"""
    note = ("## 二、视频内容\n\n- **口播要点**：\n- **用户分类与对策**：\n"
            "    - 新手上手更快。\n")
    claims, skipped = extract_by_rules(note, get_mode("standard"))
    assert [c.text for c in claims] == ["新手上手更快"]
    assert any("栏目标记" in s for s in skipped)


def test_label_strip_keeps_real_claims_with_colons():
    """真声明里也有「字段名：正文」这种冒号，不能连它一起当栏目标记吃掉。"""
    note = ("## 二、视频内容\n\n"
            "- **核心公式**：Agent = Model + Harness。\n"
            "- **机制**：接到任务目标 -> 自动拆解任务。\n"
            "- **核心理念**：Everything is a plugin（一切皆插件）。\n")
    claims, _ = extract_by_rules(note, get_mode("standard"))
    assert [c.text for c in claims] == [
        "Agent = Model + Harness", "接到任务目标 -> 自动拆解任务",
        "Everything is a plugin（一切皆插件）"]
    assert [c.context for c in claims] == ["核心公式", "机制", "核心理念"]


# ── L2 ────────────────────────────────────────────────
def test_parse_transcript_keeps_timestamps():
    segs = parse_transcript(TRANSCRIPT)
    assert segs[0].ts == "00:00" and "computer use" in segs[0].text
    assert segs[1].ts == "00:12"


def test_cn_number_and_vlm_frame_mapping():
    assert (cn_number("一"), cn_number("十一"), cn_number("二十"), cn_number("x")) == (1, 11, 20, None)
    segs = parse_vlm(VLM)
    assert segs[0].frame == 1 and "Computer Use" in segs[0].text
    assert segs[1].frame == 2


def test_vlm_frame_number_is_inherited_not_sequenced():
    """一帧的正文含空行会被切成多块；后几块必须沿用该帧号，不能按顺序编号。

    真机实测：18 帧解析出 127 段，第 1 帧的正文裂成 9 块——按顺序编号会把它们
    标成第 1..9 帧，画面证据的时间戳全错。这条锁住"继承"语义。
    """
    txt = ("--- 第 1 张图（001.jpg） ---\n第一帧正文\n\n补充说明一\n\n补充说明二\n\n"
           "--- 第 2 张图（002.jpg） ---\n第二帧正文\n\n补充说明三")
    segs = parse_vlm(txt)
    assert [s.frame for s in segs] == [1, 1, 1, 2, 2]


def test_vlm_legacy_frame_marker_supported():
    """老工具写的是 `--- 帧 3/18 ---`（没有"第"字）；认它才能给老库素材做核对。"""
    txt = "--- 帧 1/3 ---\n开场画面\n\n--- 帧 2/3 ---\n第二帧\n\n--- 帧 3/3 ---\n结尾"
    assert [s.frame for s in parse_vlm(txt)] == [1, 2, 3]


def test_vlm_without_any_marker_falls_back_to_sequence():
    assert [s.frame for s in parse_vlm("第一段内容\n\n第二段内容")] == [1, 2]


def test_retrieve_scores_and_timestamp():
    idx = SourceIndex.build(TRANSCRIPT, VLM)
    score, ev = idx.retrieve("Codex 有独立鼠标指针，操作精准度明显更高")
    assert score > 0 and ev
    assert any(e.source == "transcript" and e.ts for e in ev)
    zero, _no_ev = idx.retrieve("这条内容原视频里压根没提过两百块的月费")
    assert zero < 0.5


def test_greedy_pick_covers_claim_parts_not_repeated_parts():
    """选段按「新增覆盖」贪心，不按单段分取前几名。

    概括类声明的支撑摊在两处：前两段互相重复（单段分都高、说的是同一件事），
    第三段只盖住声明的后半（单段分低）。按单段分取前 top_k 会拿到两段重复的、
    把后半丢掉——346 条基准上过度存疑组的召回缺口（交付 0.352 vs 上界 0.449）
    就是这么来的：证据没送到，模型基于残缺证据判存疑，锅却记在 L3 头上。
    """
    md = ("[00:01]OpenClaw 开源 运行在本地设备 可接入社交软件后台 由社区维护\n"
          "[00:09]OpenClaw 开源 运行在本地设备 可接入社交软件后台 由社区维护 还内置工具市场\n"
          "[00:17]可以连接模型和聊天自动化 平台能力都能扩展\n")
    claim = "OpenClaw 开源，运行在本地设备，可接入社交软件后台，由社区维护，还能连接模型和聊天自动化"
    idx = SourceIndex.build(md, "")
    # 窗口与钉事实都关掉，单独考察「选段」这一步
    score, ev = idx.retrieve(claim, top_k=2, pin=False, window=0)
    texts = " ".join(e.text for e in ev)
    assert "连接模型和聊天自动化" in texts          # 后半段不许被重复段挤掉
    assert "社区维护" in texts                       # 前半也要在
    got = set()
    for e in ev:
        got |= tokenize(e.text)
    assert coverage(tokenize(claim), got) >= 0.7     # 交付覆盖度明显高于只取重复段
    assert score > 0                                 # top_cov 口径（单段最高覆盖度）不变


def test_coverage_and_tokenize():
    assert coverage(tokenize("Codex 操作精准"), tokenize("Codex 的体验是最好的 操作非常精准")) > 0.4
    assert coverage(tokenize("ab"), set()) == 0.0


def test_gate_missing_number_blocks_ok():
    """第三道闸：声明里的数字在原素材里查不到，不许判一致（编造数字最危险）。"""
    c = Claim(cid="C01", text="月费 200 元", verdict=VERDICT_OK,
              evidence=[Evidence(source="transcript", text="月费多少没说", score=0.9)],
              score=0.9, miss_numbers=["200"])
    apply_gates(c, get_mode("standard"))
    assert c.verdict == VERDICT_DOUBT and "查不到" in c.reason


def test_num_variants_and_cn_number_matching():
    assert "两百" in num_variants("200") and "二百" in num_variants("200")
    assert "5点3" in num_variants("5.3")
    # 口播念"四款"、笔记写"4 款" → 不算缺失（否则会凭空多出一堆存疑）
    nums, _names = hard_miss("评测了 4 款工具", "本期视频测评我常用的四款AI工具")
    assert nums == []
    # 反过来：原素材里根本没有的数字必须被抓出来
    nums2, _ = hard_miss("用了通义千问3.8 Max", "接下来我们看一下阿里的这个ai变成工具code")
    assert nums2 == ["3.8"]


def test_hard_miss_ignores_common_words():
    nums, names = hard_facts("the app AI ok 工具 Computer Use")
    assert nums == set()
    assert names == {"computer", "use"}


def test_hard_miss_number_boundary_not_substring():
    """数字命中必须带边界：200 不能从 2000 / 1200 里找到出处（B1）。"""
    # 声明 200，原素材只有 2000 → 必须判缺失（此前子串匹配会漏判）
    nums, _ = hard_miss("这款工具收费 200 元", "视频里说价格是2000元起")
    assert nums == ["200"]
    nums2, _ = hard_miss("200 元", "起步价1200元")
    assert nums2 == ["200"]
    # 边界不该误伤：原素材真有 200 时不算缺失
    nums3, _ = hard_miss("这款工具收费 200 元", "视频里说价格是200元")
    assert nums3 == []
    # 中文数字同理：「四十」不命中「四十万」（口播量级与笔记数字是两回事）
    nums4, _ = hard_miss("总量 40", "视频说总量四十万")
    assert nums4 == ["40"]


def test_hard_facts_ignores_frame_numbers():
    """画面小节里的「第 N 张图」是帧号不是事实数字（H1）。"""
    nums, names = hard_facts("第 3 张图显示：Qoder 是阿里的产品")
    assert "3" not in nums
    assert "qoder" in names
    nums2, _ = hard_facts("第11-13张：Qoder 界面")
    assert "11" not in nums2 and "13" not in nums2
    # 帧号剔除不能误伤真实数字：同一句里的价格仍要被抓
    nums3, _ = hard_facts("第 2 张图：收费 200 元")
    assert "200" in nums3


# ── L3 ────────────────────────────────────────────────
def test_parse_verdicts_tolerates_fence_and_aliases():
    raw = '```json\n[{"i":1,"verdict":"一致"},{"i":2,"verdict":"MISMATCH","fix":"改"},{"i":7,"verdict":"ok"}]\n```'
    out = parse_verdicts(raw, 3)
    assert out[1]["verdict"] == VERDICT_OK
    assert out[2]["verdict"] == VERDICT_MISMATCH and out[2]["fix"] == "改"
    assert 7 not in out


def test_gate_zero_evidence_downgrades_ok():
    """核心闸门：没有证据就没有一致——模型的 ✅ 必须被压回存疑。"""
    c = Claim(cid="C01", text="月费 200 元", verdict=VERDICT_OK)
    apply_gates(c, get_mode("standard"))
    assert c.verdict == VERDICT_DOUBT and "没回查到出处" in c.reason


def test_gate_double_signal_only_in_deep():
    weak_ok = Claim(cid="C01", text="x", verdict=VERDICT_OK,
                    evidence=[Evidence(source="transcript", text="y", score=0.05)], score=0.05)
    stable = Claim(cid="C01", text="x", verdict=VERDICT_OK,
                   evidence=[Evidence(source="transcript", text="y", score=0.05)], score=0.05)
    apply_gates(weak_ok, get_mode("standard"))
    apply_gates(stable, get_mode("deep"))
    assert weak_ok.verdict == VERDICT_OK          # 标准模式不启用双信号
    assert stable.verdict == VERDICT_DOUBT        # 深度模式压回存疑


# ── 第四道闸：推理性改写（D9）────────────────────────────

def test_business_leap_detects_unsupported_service_framing():
    """笔记把"我自己用的功能"写成"对外提供的服务"——原素材里连'服务'这个词都没有。"""
    from distill_workshop.core.check.evidence_retriever import business_leap

    src = "我经常用 Computer Use 做几个场景的功能：1、调研竞品 2、测试软件 3、办公自动化"
    claim = "可基于 Computer Use 功能提供竞品调研服务、软件测试服务、办公自动化定制服务"
    assert business_leap(claim, src) == ["定制", "服务"]


def test_business_leap_quiet_when_source_has_the_word():
    """原素材自己就说了"收费/变现"时不该误伤（词表故意不收 收费/付费，另测普通动词）。"""
    from distill_workshop.core.check.evidence_retriever import business_leap

    src = "这个工具是收费的，我给别人提供服务收 200 一次"
    assert business_leap("提供竞品调研服务", src) == []


def test_business_leap_ignores_generic_verbs():
    """「提供/交付/支持」这类日常动词不进词表——否则「提供图形界面」会被误伤成存疑。"""
    from distill_workshop.core.check.evidence_retriever import business_leap

    assert business_leap("该工具提供图形界面和快捷键支持", "它的界面和快捷键都很好用") == []


def test_gate_business_leap_downgrades_ok():
    c = Claim(cid="C01", text="可基于它提供调研服务", verdict=VERDICT_OK,
              evidence=[Evidence(source="transcript", text="我用它调研竞品", score=0.8)],
              miss_biz=["服务"])
    apply_gates(c, get_mode("fast"))
    assert c.verdict == VERDICT_DOUBT
    assert "对外" in c.reason and "推论" in c.reason
    assert c.fix


def test_gate_business_leap_not_triggered_without_missing_terms():
    c = Claim(cid="C01", text="我用它调研竞品", verdict=VERDICT_OK,
              evidence=[Evidence(source="transcript", text="我用它调研竞品", score=0.8)])
    apply_gates(c, get_mode("fast"))
    assert c.verdict == VERDICT_OK


# ── 栏目名 / 帧号钉住 / 漏返补问（D10）────────────────────

def test_claim_context_is_the_field_label():
    """栏目名是声明的限定语境：「测试任务」不是说"任务成功了"。"""
    assert Claim("C01", "找到下载目标栏并上传 pio 文件",
                 section="口播要点（按时间顺序）/测试任务").context == "测试任务"
    assert Claim("C01", "x", section="正文").context == ""          # 没有子栏目时不给噪声


def test_frame_refs_parses_ranges_and_cn_numbers():
    from distill_workshop.core.check.evidence_retriever import frame_refs

    assert frame_refs("第5-6张：在聊天界面选择 computer use") == {5, 6}
    assert frame_refs("第11-13张") == {11, 12, 13}
    assert frame_refs("第五张") == {5}
    assert frame_refs("没有任何帧号") == set()


def test_frame_referenced_evidence_is_pinned_and_survives_budget():
    """声明点名"第 3-4 张"时，那两帧的画面证据必须进证据集，且不被字数预算裁掉。

    实测踩到的两个坑：① 按字面分数会挑到别的帧；② 钉住后仍被预算裁掉（钉了等于没钉）。
    """
    VLM = ("**第一张图**\n" + "甲" * 60 + "\n\n"
           "**第二张图**\n" + "乙" * 60 + "\n\n"
           "**第三张图**\n画面文字：终端里 npm install -g @openai/codex 正在安装\n\n"
           "**第四张图**\n画面文字：安装完成 already installed\n\n"
           "**第五张图**\n" + "戊" * 60)
    idx = SourceIndex.build("", VLM)
    _score, ev = idx.retrieve("终端安装 Codex", context="第3-5张", budget=1)   # 预算故意极小
    pinned = [e for e in ev if e.pinned]
    assert pinned, "点名帧号后必须有钉住的证据"
    assert any("第三张图" in e.text for e in pinned), "钉住的必须是声明点名的那几帧"


def test_pinned_flag_marks_hard_fact_segments():
    VLM = "**第一张图**\n画面文字：Lumen Reader 导入书籍\n\n**第二张图**\n无关画面内容描述"
    idx = SourceIndex.build("", VLM)
    _, ev = idx.retrieve("Lumen Reader 导入书籍")
    assert any(e.pinned and e.fact_hit >= 1.0 for e in ev)


def test_missing_verdict_triggers_one_extra_ask():
    """模型漏返某条（机械失败）时，单独补问一次，而不是直接记 ⚠️ 白丢一条。"""
    claims = [Claim(cid="C01", text="声明甲", evidence=[Evidence("transcript", "甲", score=0.9)]),
              Claim(cid="C02", text="声明乙", evidence=[Evidence("transcript", "乙", score=0.9)])]
    calls = []

    def chat(messages, **_kw):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return _Resp('```json\n[{"i":1,"verdict":"ok","reason":"支持","fix":""}]\n```')  # 漏返第 2 条
        return _Resp('```json\n[{"i":1,"verdict":"ok","reason":"补问支持","fix":""}]\n```')

    verify_claims(claims, get_mode("fast"), chat, on_step=None)
    assert len(calls) == 2, "应当只有一次补问，不是整批重跑"
    assert claims[0].verdict == VERDICT_OK
    assert claims[1].verdict == VERDICT_OK and claims[1].reason == "补问支持"


def test_missing_verdict_stays_doubt_when_reask_fails():
    claims = [Claim(cid="C01", text="声明甲", evidence=[Evidence("transcript", "甲", score=0.9)])]
    verify_claims(claims, get_mode("fast"), lambda m, **k: _Resp("[]"), on_step=None)
    assert claims[0].verdict == VERDICT_DOUBT        # 绝不空手：仍是存疑 + 有理由


def test_build_items_shows_field_label():
    c = Claim(cid="C01", text="找到下载目标栏并上传 pio 文件", section="口播要点/测试任务",
              evidence=[Evidence("transcript", "看一下它能不能找到下载目标栏并且上传", ts="09:51")])
    out = build_items([c])
    assert "测试任务" in out and "找到下载目标栏并上传 pio 文件" in out


def test_verify_failure_marks_all_doubt():
    claims = [Claim(cid="C01", text="a", evidence=[Evidence(source="transcript", text="a", score=1.0)])]

    def boom(_messages, **_kw):
        raise RuntimeError("network down")

    verify_claims(claims, get_mode("standard"), boom, on_step=None, retry_pause=0)
    assert all(c.verdict == VERDICT_DOUBT for c in claims)
    assert "调用失败" in claims[0].reason        # 失败原因写进报告，不吞掉


def test_verify_empty_array_still_doubt():
    claims = [Claim(cid="C01", text="a", evidence=[Evidence(source="transcript", text="a", score=1.0)])]
    verify_claims(claims, get_mode("standard"), lambda m, **k: _Resp("[]"), on_step=None)
    assert claims[0].verdict == VERDICT_DOUBT


def test_verify_retries_once_on_rate_limit():
    """限流（429）时先等一会儿重试一次：重试成功就不该把整批判成存疑。"""
    calls = {"n": 0}

    def flaky(messages, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return _Resp(json.dumps([{"i": 1, "verdict": "ok", "reason": "证据支持", "fix": ""}],
                                ensure_ascii=False))

    c = Claim(cid="C01", text="a", evidence=[Evidence(source="transcript", text="a", score=1.0)])
    verify_claims([c], get_mode("standard"), flaky, on_step=None, retry_pause=0)
    assert calls["n"] == 2 and c.verdict == VERDICT_OK


# ── L4 ────────────────────────────────────────────────
def test_badge_and_json_shape():
    r = CheckResult(key="aweme:1", mode="standard", created_at="2026-09-17 10:00:00",
                    claims=[Claim("C01", "a", verdict=VERDICT_OK),
                            Claim("C02", "b", verdict=VERDICT_DOUBT),
                            Claim("C03", "c", verdict=VERDICT_MISMATCH)])
    assert badge(r) == "✅ 已抽核 1/3 · ⚠️ 1 条存疑 · ❌ 1 条不一致"   # 措辞降级见 D14
    data = build_json(r)
    assert data["status"] == "checked" and data["counts"]["total"] == 3
    md = render_markdown(r)
    assert "核对边界" in md and "C03" in md


def test_inject_badge_is_idempotent():
    r = CheckResult(key="k", mode="standard", created_at="2026-09-17 10:00:00",
                    claims=[Claim("C01", "a", verdict=VERDICT_OK)])
    once = inject_badge(NOTE, r)
    twice = inject_badge(once, r)
    assert once.count("dystill-check:begin") == 1 and once.count("dystill-check:end") == 1
    assert twice.count("dystill-check:begin") == 1 and twice.count("dystill-check:end") == 1
    assert twice.count("已抽核") == once.count("已抽核") == 1     # 重复核对不会越插越多
    assert "标题" not in once.split("dystill-check:begin")[0].split("\n\n")[-1]
    assert twice.split("---", 2)[2].lstrip("\n").startswith("<!-- dystill-check:begin")


def test_apply_fixes_requires_exact_match():
    r = CheckResult(key="k", mode="standard", claims=[
        # fix 必须有证据的字面支撑才会被写进笔记（幻觉 fix 比不改更糟）
        Claim("C01", "x", original="月费 200 元", verdict=VERDICT_MISMATCH, fix="月费 100 元",
              evidence=[Evidence(source="transcript", text="这个月费是 100 元一个月")]),
        Claim("C02", "y", original="不存在的一句", verdict=VERDICT_MISMATCH, fix="z"),
    ])
    out, applied = apply_fixes("笔记里写着月费 200 元，别的不动。", r)
    assert "月费 100 元" in out and len(applied) == 1


def test_apply_fixes_skips_advice_and_unsupported_fixes():
    """三重保险：建议式 fix 不落盘；证据里没影子的 fix 不落盘；只有有支撑的才动笔。"""
    r = CheckResult(key="k", mode="standard", claims=[
        Claim("C01", "x", original="它提供竞品调研服务", verdict=VERDICT_MISMATCH,
              fix="建议回看原片确认"),                                   # 操作指引，不是改写
        Claim("C02", "y", original="月费 999 元", verdict=VERDICT_MISMATCH,
              fix="月费 3 元",                                            # 证据里没有 3 元 → 幻觉
              evidence=[Evidence(source="transcript", text="月费是 200 元")]),
        Claim("C03", "z", original="有证据的原句", verdict=VERDICT_MISMATCH,
              fix="月费 100 元",
              evidence=[Evidence(source="transcript", text="月费是 100 元一个月")]),
    ])
    skipped: list[str] = []
    out, applied = apply_fixes("笔记：它提供竞品调研服务 / 月费 999 元 / 有证据的原句。", r,
                               on_step=lambda m: skipped.append(m))
    assert len(applied) == 1 and applied[0].startswith("C03")
    assert "月费 999 元" in out and "月费 100 元" in out      # 只有 C03 被改
    assert len(skipped) == 2                                  # 跳过原因透明上报


# ── 端到端 ─────────────────────────────────────────────
def test_run_check_end_to_end(tmp_path: Path):
    app = make_app(tmp_path)
    rd = make_record(tmp_path)
    out = run_check(rd, app, mode="standard", chat_fn=fake_chat, on_step=None)

    assert out["status"] == "checked" and out["counts"]["total"] > 0
    assert Path(out["json"]).is_file() and Path(out["markdown"]).is_file()
    # 徽章已注入笔记顶部（frontmatter 之后）
    note = (rd / "note.md").read_text(encoding="utf-8")
    assert "dystill-check:begin" in note and "已抽核" in note
    # 台账留痕
    from distill_workshop.core.ledger import read_events
    assert any(e.get("event") == "check" for e in read_events(app))


def test_run_check_uses_actual_evidence_for_gate(tmp_path: Path):
    """端到端验证闸门真的在跑：假模型对所有条目都说 ok，没证据的必须落存疑。"""
    app = make_app(tmp_path)
    rd = make_record(tmp_path)
    out = run_check(rd, app, mode="standard", chat_fn=fake_chat, on_step=None)
    data = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
    assert data["counts"]["doubt"] >= 1
    for c in data["claims"]:
        if c["verdict"] == "ok":
            assert c["evidence"], f"{c['cid']} 判了 ok 却没有证据"


def test_run_check_without_provider_all_doubt(tmp_path: Path):
    app = make_app(tmp_path)          # 无 providers
    rd = make_record(tmp_path)
    out = run_check(rd, app, mode="standard", on_step=None)
    assert out["counts"]["ok"] == 0 and out["counts"]["doubt"] == out["counts"]["total"]
    md = Path(out["markdown"]).read_text(encoding="utf-8")
    assert "未配置模型供应商" in md


def test_run_check_rejects_invalid_note(tmp_path: Path):
    app = make_app(tmp_path)
    rd = make_record(tmp_path, note="# 空壳\n")
    with pytest.raises(ValueError):
        run_check(rd, app, chat_fn=fake_chat, on_step=None)


def test_cap_balanced_covers_every_section():
    """超上限时按章节轮转取样：不能让后半篇整节一条都不核（实测真实笔记踩到过）。"""
    from distill_workshop.core.check.claim_extractor import cap_balanced
    claims = [Claim(cid=f"C{i}", text=f"声明内容 {i}", section="前段" if i < 30 else "后段")
              for i in range(60)]
    picked, capped = cap_balanced(claims, 10)
    assert capped and len(picked) == 10
    assert {c.section for c in picked} == {"前段", "后段"}
    assert picked == sorted(picked, key=lambda c: claims.index(c))   # 保持原顺序


def test_cap_balanced_noop_when_under_limit():
    from distill_workshop.core.check.claim_extractor import cap_balanced
    claims = [Claim(cid="C01", text="x", section="s")]
    picked, capped = cap_balanced(claims, 5)
    assert picked == claims and capped is False


def test_mode_override_does_not_pollute_global_modes(tmp_path: Path):
    """config 调大上限不能污染全局模式表（否则影响后续所有核对）。"""
    from distill_workshop.core.check.modes import get_mode
    app = make_app(tmp_path)
    app.pipeline = {"check_max_claims": 7}
    rd = make_record(tmp_path)
    from distill_workshop.core.check import analyze
    note = (rd / "note.md").read_text(encoding="utf-8")
    analyze(note, TRANSCRIPT, VLM, mode="fast", app=app, on_step=None)
    assert get_mode("fast").max_claims == 40


def test_resume_reuses_previous_verdicts(tmp_path: Path):
    """断点续跑：已判定的条目不再送进 L3 判定（低配额通道被限流打断后的恢复手段）。"""
    from distill_workshop.core.check import analyze
    from distill_workshop.core.check.engine import load_previous
    app = make_app(tmp_path)
    rd = make_record(tmp_path)
    note = (rd / "note.md").read_text(encoding="utf-8")

    first = run_check(rd, app, mode="fast", chat_fn=fake_chat, on_step=None)
    assert first["counts"]["total"] > 0
    l3_asks = []

    def counting_chat(messages, **_kw):
        if "事实核对员" in messages[0]["content"]:
            l3_asks.append(1)
        return fake_chat(messages)

    res = analyze(note, TRANSCRIPT, VLM, mode="fast", chat_fn=counting_chat,
                  previous=load_previous(rd), resume=True, on_step=None)
    assert l3_asks == []                          # L3 一次都没调（L1 消歧仍会跑）
    assert all(c.verdict for c in res.claims)
    assert res.count("ok") == first["counts"]["ok"]


def test_retry_failed_only_rejudges_failed(tmp_path: Path):
    """--retry-failed 只重判「调用失败」的条目，成功的不重复花钱。"""
    from distill_workshop.core.check import analyze
    app = make_app(tmp_path)
    rd = make_record(tmp_path)
    note = (rd / "note.md").read_text(encoding="utf-8")
    run_check(rd, app, mode="fast", chat_fn=fake_chat, on_step=None)

    # 手工把上一轮报告改造成"只有第 1 条失败"
    prev = load_previous_path(rd)
    first_text = next(iter(prev))
    prev[first_text] = {"text": first_text, "verdict": "doubt",
                        "reason": "该批模型调用失败（限流/网络），未能判定", "fix": ""}
    l3_batches = []

    def chat(messages, **_kw):
        if "事实核对员" in messages[0]["content"]:
            l3_batches.append(messages[0]["content"].count("声明："))
        return fake_chat(messages)

    analyze(note, TRANSCRIPT, VLM, mode="fast", chat_fn=chat,
                  previous=prev, only_failed=True, on_step=None)
    assert sum(l3_batches) == 1        # 只有那 1 条被重新送进模型
    assert len(l3_batches) == 1


def test_resume_still_applies_new_gates(tmp_path: Path):
    """复用旧判定时必须重新过闸：判定标准升级后，旧的 ✅ 不能被当成免检。"""
    from distill_workshop.core.check import analyze
    make_app(tmp_path)
    note_extra = NOTE.replace("## 二、视频内容",
                              "## 二、视频内容\n\n- **对比与数据**：月费 200 元，响应时间 3 秒\n")
    rd = make_record(tmp_path, note=note_extra)
    note = (rd / "note.md").read_text(encoding="utf-8")
    stale = {"月费 200 元，响应时间 3 秒":
             {"text": "月费 200 元，响应时间 3 秒", "verdict": "ok", "reason": "证据支持", "fix": ""}}

    res = analyze(note, TRANSCRIPT, VLM, mode="fast", chat_fn=None,
                  previous=stale, resume=True, on_step=None)
    hit = [c for c in res.claims if "200" in c.text]
    assert len(hit) == 1
    assert "200" in hit[0].miss_numbers        # 原素材里确实没有 200
    assert hit[0].verdict != "ok"              # 第三道闸把旧的 ✅ 压下来了


def load_previous_path(rd: Path) -> dict:
    import json as _json
    data = _json.loads((rd / "check.json").read_text(encoding="utf-8"))
    return {c["text"]: dict(c) for c in data["claims"]}


def test_find_note_ignores_own_report(tmp_path: Path):
    """核对产物 check.md 不能被当成笔记——否则第二次核对会直接报"没有笔记"（实测踩到）。"""
    from distill_workshop.core.check.engine import find_note
    rd = tmp_path / "data" / "aweme_1"
    rd.mkdir(parents=True)
    (rd / "note-旧营销版备份.md").write_text(NOTE, encoding="utf-8")
    (rd / "check.md").write_text("# 核对报告\n" + "报告正文" * 100, encoding="utf-8")
    assert find_note(rd).name == "note-旧营销版备份.md"
    # note.md 存在时优先它，且永远不选 check.md
    (rd / "note.md").write_text(NOTE, encoding="utf-8")
    assert find_note(rd).name == "note.md"
    (rd / "note.md").unlink()
    (rd / "check.md").unlink()
    (rd / "note-旧营销版备份.md").unlink()
    (rd / "check.md").write_text("# 核对报告", encoding="utf-8")
    assert find_note(rd) is None       # 只剩自己的报告 → 明确"没有笔记"，不能拿报告去核


def test_recheck_same_record_twice(tmp_path: Path):
    """端到端：同一记录连续核对两次都要成功（第一次产出 check.md 不能挡住第二次）。"""
    app = make_app(tmp_path)
    rd = make_record(tmp_path)
    first = run_check(rd, app, mode="fast", chat_fn=fake_chat, on_step=None)
    second = run_check(rd, app, mode="fast", chat_fn=fake_chat, on_step=None)
    assert first["status"] == "checked" and second["status"] == "checked"
    assert (rd / "check.md").is_file()


def test_resolve_record_dir_by_aweme_and_keyword(tmp_path: Path):
    app = make_app(tmp_path)
    make_record(tmp_path)
    assert resolve_record_dir(f"https://www.douyin.com/video/{AWEME}", app).name == f"aweme_{AWEME}"
    assert resolve_record_dir(f"aweme_{AWEME}", app).name == f"aweme_{AWEME}"
    assert resolve_record_dir("7678251600959", app).name == f"aweme_{AWEME}"
    with pytest.raises(FileNotFoundError):
        resolve_record_dir("https://www.douyin.com/video/9999999999999999999", app)


def test_default_chat_honors_judge_provider(tmp_path: Path):
    """判定通道与笔记通道分离（D21）：judge_provider 指定名字只切判定，配错要报错不静默。"""
    from distill_workshop.config import ProviderConfig
    from distill_workshop.core.check.engine import _default_chat

    app = make_app(tmp_path)
    app.providers = [
        ProviderConfig(name="a", api_base="https://a/v1", api_key="k", model="m1"),
        ProviderConfig(name="b", api_base="https://b/v1", api_key="k", model="m2"),
    ]
    _chat_fn, hint = _default_chat(app)
    assert hint == "a"                                   # 缺省走降级链第一家

    app.pipeline["judge_provider"] = "b"
    _chat_fn, hint = _default_chat(app)
    assert hint == "b"                                   # 判定通道切到指定家

    # 逗号分隔多个 = 判定专用降级链（D21 追记：不同供应商互为备份，防降级窗口报废整轮）
    app.providers.append(ProviderConfig(name="c", api_base="https://c/v1", api_key="k", model="m3"))
    app.pipeline["judge_provider"] = "b, a, b"
    _chat_fn, hint = _default_chat(app)
    assert hint == "b"                                   # 链首为 b，重复名去重

    app.pipeline["judge_provider"] = "不存在"
    try:
        _default_chat(app)
        assert False, "配错名字应该报错而不是静默退回全链"
    except ValueError as e:
        assert "不存在" in str(e)

    app.pipeline["judge_provider"] = "b, 不存在"
    try:
        _default_chat(app)
        assert False, "链中任何一个未知名字都要报错"
    except ValueError as e:
        assert "不存在" in str(e)

    # 名字校验要排在「有没有就绪 provider」**之前**（2026-10-04 复查盲区）：
    # 否则「名字写错 + 没配 provider」报的是「未配置模型供应商」，把人引去查 key
    empty = make_app(tmp_path)                    # make_app 没有 providers
    empty.pipeline["judge_provider"] = "b"
    try:
        _default_chat(empty)
        assert False, "指定了通道名却没有就绪 provider 时，要报出名字而不是笼统说没配模型"
    except ValueError as e:
        assert "b" in str(e) and "provider" in str(e)
    empty.pipeline["judge_provider"] = ""            # 没名字 + 没 provider = 正常空手
    assert _default_chat(empty) == (None, "")


def test_l1_disambiguation_is_opt_in(tmp_path: Path, monkeypatch):
    """L1 模型消歧默认**不开**（见 D24），且必须显式设 pipeline.check_l1_disambiguate 才生效。

    原实现把判定通道的解析放在 extract_claims 之后，而 CLI/bench_check 都不传 chat_fn，
    消歧的 chat_fn 恒为 None——ROADMAP 记作已落地，实际从未执行过（2026-10-04 复查）。
    这个测试钉子住「开/关都真的生效」，而不只是代码里看得见接线。
    """
    from distill_workshop.config import ProviderConfig
    from distill_workshop.core.check import analyze, engine

    app = make_app(tmp_path)
    app.providers = [ProviderConfig(name="text", api_base="https://x/v1",
                                    api_key="k", model="m")]
    seen: list[str] = []

    def spy(_providers, _timeout=120):   # judge_timeout 进入 _chat_fn 签名（D21 追记二）
        def fn(messages):
            content = messages[0]["content"]
            seen.append(content)
            if "消歧" in content:                  # L1 消歧提示词（L3 提示词里没有这两个字）
                return _Resp('[{"i": 1, "text": "L1 消歧后的声明"}]')
            return fake_chat(messages)              # L3 照旧
        return fn

    monkeypatch.setattr(engine, "_chat_fn", spy)

    for off in (False, "false", ""):              # 没设 / 字符串假值都不开
        seen.clear()
        app.pipeline["check_l1_disambiguate"] = off
        analyze(NOTE, TRANSCRIPT, VLM, mode="fast", app=app, on_step=None)
        assert not any("消歧" in p for p in seen), off

    # 开关是**唯一决定者**：关着时即使显式传 chat_fn 也不开——
    # 否则 run_check(chat_fn=…) 这类只想换 L3 通道的调用方会被动改掉 L1。
    seen.clear()
    app.pipeline["check_l1_disambiguate"] = False

    def l3_only(messages):
        seen.append(messages[0]["content"])
        return fake_chat(messages)

    analyze(NOTE, TRANSCRIPT, VLM, mode="fast", app=app, chat_fn=l3_only, on_step=None)
    assert not any("消歧" in p for p in seen)

    seen.clear()
    app.pipeline["check_l1_disambiguate"] = True
    res = analyze(NOTE, TRANSCRIPT, VLM, mode="fast", app=app, on_step=None)
    assert any("消歧" in p for p in seen), "开了就要真的把模型接进 L1"
    assert any(c.text == "L1 消歧后的声明" for c in res.claims), \
        "消歧结果要真的进入声明列表"


def test_verify_failure_reason_carries_real_error():
    """D21 追记说「失败日志透传末次错误原文」，但产物里只写「限流/网络」——
    「余额不足」和「限流」处置完全不同，吞掉真实死因会把人引向错误排查方向。"""
    from distill_workshop.core.check.engine import prev_failed

    c = Claim(cid="C01", text="a",
              evidence=[Evidence(source="transcript", text="a", score=1.0)])
    steps: list[str] = []

    def boom(messages, **_kw):
        raise RuntimeError("HTTP 429 code=1113 余额不足或无可用资源包")

    verify_claims([c], get_mode("fast"), boom, on_step=steps.append, retry_pause=0)
    assert c.failed is True
    assert "余额不足" in c.reason and "调用失败" in c.reason
    assert any("余额不足" in s for s in steps)
    assert prev_failed({"reason": c.reason})       # --retry-failed 仍能认出这条


def test_supplement_failure_keeps_error_and_is_retryable():
    """补问那一批也抛错时，错误摘要不能吞、而且要打上 failed 供 --retry-failed 恢复。
    原代码把 raised/err 拿了不用（死在这里），补问失败在产物里只留一句「保持存疑」。"""
    calls: list[int] = []

    def flaky(messages, **_kw):
        calls.append(1)
        if len(calls) == 1:                        # 整批：漏返第 2 条（机械失败）
            return _Resp('[{"i": 1, "verdict": "ok", "reason": "有证据", "fix": ""}]')
        raise RuntimeError("HTTP 500 上游超时")      # 补问：直接抛错

    claims = [Claim(cid="C01", text="甲句里有事实",
                    evidence=[Evidence(source="transcript", text="甲句里有事实", score=1.0)]),
              Claim(cid="C02", text="乙句里有事实",
                    evidence=[Evidence(source="transcript", text="乙句里有事实", score=1.0)])]
    steps: list[str] = []
    verify_claims(claims, get_mode("fast"), flaky, on_step=steps.append, retry_pause=0)
    assert len(calls) == 2                         # 只补问一次，不整批重跑
    assert claims[0].reason == "有证据" and claims[0].failed is False
    assert claims[1].failed is True
    assert "上游超时" in claims[1].reason
    assert any("上游超时" in s for s in steps)

def test_disambiguate_retries_once_when_unparsable():
    """某批首次返回不可解析时会重问一次；第二次成功就采用第二次的。"""
    from distill_workshop.core.check.claim_extractor import disambiguate
    from distill_workshop.core.check.models import Claim

    calls = []

    def fake_chat(messages):
        calls.append(1)
        if len(calls) == 1:
            return type("R", (), {"content": "抱歉，我无法按要求处理这个请求。"})()
        return type("R", (), {"content": '[{"i": 1, "text": "补全后的自足句"}]'})()

    out = disambiguate([Claim(cid="C01", text="它很好用", original="它很好用")], fake_chat)
    assert len(calls) == 2                    # 失败后重试了一次
    assert out[0].text == "补全后的自足句"


def test_disambiguate_gives_up_after_two_tries():
    """两次都不可解析时退回规则原文，不抛错、不空手。"""
    from distill_workshop.core.check.claim_extractor import disambiguate
    from distill_workshop.core.check.models import Claim

    calls = []

    def fake_chat(messages):
        calls.append(1)
        return type("R", (), {"content": "不是 JSON"})()

    out = disambiguate([Claim(cid="C01", text="原文保留", original="原文保留")], fake_chat)
    assert len(calls) == 2
    assert out[0].text == "原文保留"

def test_disambiguation_rejects_duplicate_indices():
    """模型返回重复编号（编号乱了）时整批退回原文——绝不能错位回填。"""
    from distill_workshop.core.check.claim_extractor import parse_disambiguation

    dup = '[{"i": 1, "text": "改写后的自足句甲"}, {"i": 1, "text": "改写后的自足句乙"}]'
    assert parse_disambiguation(dup, 2) == {}

    ok = '[{"i": 1, "parts": ["改写后的自足句甲"]}, {"i": 2, "parts": ["改写后的自足句乙"]}]'
    got = parse_disambiguation(ok, 2)
    assert got == {1: ["改写后的自足句甲"], 2: ["改写后的自足句乙"]}


def test_disambiguation_splits_by_index_not_by_order():
    """拆分由输入编号映射：第 i 条拆出的段落回到第 i 条，绝不会串到别条。

    这是 2026-10-13 错位事故的回归测试（当时模型自己编号，49/120 条内容串行）。
    """
    from distill_workshop.core.check.claim_extractor import disambiguate
    from distill_workshop.core.check.models import Claim

    claims = [
        Claim(cid="", text="甲甲甲甲甲甲", section="S1", original="甲甲甲甲甲甲"),
        Claim(cid="", text="乙乙乙乙乙乙", section="S2", original="乙乙乙乙乙乙"),
    ]

    def fake_chat(messages):
        # 第 2 条拆成两段；第 1 条不拆。故意让返回顺序与输入一致但内容交错命名，
        # 用来验证映射靠编号而非顺序。
        return type("R", (), {"content": (
            '[{"i": 1, "parts": ["甲甲甲甲甲甲"]},'
            ' {"i": 2, "parts": ["乙乙乙乙乙乙", "丙丙丙丙丙丙"]}]')})()

    out = disambiguate(claims, fake_chat)
    assert len(out) == 3
    assert out[0].text == "甲甲甲甲甲甲" and out[0].section == "S1"
    assert out[1].text == "乙乙乙乙乙乙" and out[1].section == "S2"
    assert out[2].text == "丙丙丙丙丙丙" and out[2].section == "S2"   # 跟着乙条走
    assert out[1].original == "乙乙乙乙乙乙"      # 拆分首条保留原句（供 --rewrite）
    assert out[2].original == ""                  # 拆出的次条不重复携带原句
