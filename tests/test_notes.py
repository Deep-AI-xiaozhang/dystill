import json
from pathlib import Path

from distill_workshop.config import AppConfig
from distill_workshop.core.notes import (
    build_prompt,
    check_note_quality,
    chunk_text,
    default_note_fn,
    generate_note,
)

GOOD_NOTE = """# 蒸馏笔记 · 测试视频

> 蒸馏日期：2026-09-16 ｜ 类型：实操教程 ｜ 主题：用平台任务接单的完整流程拆解

## 一、结论先行
这条视频讲了一个完整的平台任务接单流程，核心是把任务拆解成选品、报价、交付三步执行。
作者展示了任务后台的单价与收入数据，流程清晰，值得学习。
## 二、视频内容
- 口播要点：作者先展示后台数据截图建立信任，再逐步拆解接单流程，强调先跑通最小闭环再放大投入，最后给出新手最容易踩的两个坑。
- 画面/操作要点：屏幕录制显示任务列表、报价表与结算记录，关键数字用红框标注。
## 三、知识要点（核心）
- **关键概念**：平台任务接单是指按平台发布的任务报价并交付的工作方式。
- **实操步骤**：选品、报价、交付三步，先小单验证再放大。
- **工具与资源**：表格清单与模板库，单价 200 元起，周结算到账。
- **对比与数据**：月收入三万元，投入时间每天两小时，转化率约一成。
- **注意事项**：新手最容易踩的两个坑是报价过低和交付延期。
## 四、总结与适用场景
适合有一技之长、想通过平台任务获得稳定收入的人入门，先小单验证再放大投入。
"""

SHORT_NOTE = "这条视频讲了点东西。"
POLLUTED_NOTE = GOOD_NOTE.replace("主题：用平台任务接单的完整流程拆解", "主题：douyin-downloader → faster-whisper")
PLACEHOLDER_NOTE = GOOD_NOTE.replace("主题：用平台任务接单的完整流程拆解", "主题：<一句话概括视频主题；禁止写蒸馏工具链>")
TRUNCATED_NOTE = GOOD_NOTE.split("## 四、")[0]


def make_app(tmp_path: Path, providers=None) -> AppConfig:
    return AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                     data_dir=tmp_path / "data", cache_dir=tmp_path / "cache",
                     providers=providers or [])


def test_check_note_quality_good():
    assert check_note_quality(GOOD_NOTE) == []
    assert len(GOOD_NOTE) >= 400  # 标准模板正文下限


def test_check_note_quality_too_short():
    assert any("正文过短" in p for p in check_note_quality(SHORT_NOTE))


def test_check_note_quality_pollution_and_placeholder():
    assert any("蒸馏工具链" in p for p in check_note_quality(POLLUTED_NOTE))
    assert any("占位符" in p for p in check_note_quality(PLACEHOLDER_NOTE))


def test_check_note_quality_truncation():
    assert any("截断" in p for p in check_note_quality(TRUNCATED_NOTE))
    # 旧营销模板产物（情绪/变现分析栏目）在纯知识模板下必须被拦下
    emo = "# 蒸馏笔记 · 情绪\n\n> 类型：情绪内容\n\n## 这个视频是啥\n某人晒收益。\n"
    assert any("截断" in p for p in check_note_quality(emo))  # 缺第四板块
    assert any("结论先行" in p for p in check_note_quality(emo))  # 结构跑偏


def test_prompt_is_pure_knowledge_template():
    prompt = build_prompt("测试视频", "转写", "画面")
    for sec in ("一、结论先行", "二、视频内容", "三、知识要点", "四、总结与适用场景"):
        assert sec in prompt
    # 营销栏目只能在「禁止出现」那句原则里提到一次，不能当成模板栏目
    for banned in ("变现", "转化链路", "内容公式"):
        assert prompt.count(banned) == 1


def test_chunk_text_cap_rebalance():
    text = "\n".join(f"第{i}行内容" * 10 for i in range(30))
    chunks = chunk_text(text, max_chars=200, cap=5)
    assert 1 <= len(chunks) <= 5
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


class _Resp:
    def __init__(self, content):
        self.content = content


def test_generate_note_quality_retry(tmp_path: Path):
    app = make_app(tmp_path, providers=[type("P", (), {"name": "main", "api_base": "x",
                                                      "api_key": "k", "model": "m",
                                                      "max_attempts": 3,
                                                      "ready": True})()])  # duck-typing provider
    calls = []

    def fake_chat(messages, providers_, max_tokens=None, **kw):
        calls.append(max_tokens)
        return _Resp(SHORT_NOTE if len(calls) == 1 else GOOD_NOTE)

    out = generate_note("短转写", "画面略", "测试视频", app, chat_fn=fake_chat)
    assert out == GOOD_NOTE
    assert len(calls) == 2  # 第一版自检不通过 → 带定向指令重试一次


def test_generate_note_fallback_without_providers(tmp_path: Path):
    out = generate_note("转写原文", "画面原文", "测试视频", make_app(tmp_path))
    assert "草稿" in out and "转写原文" in out


def test_generate_note_handles_gallery_without_transcript(tmp_path: Path):
    """图文帖没有语音：空 transcript 不能炸，提示词里要写明"这是图文帖"。"""
    app = make_app(tmp_path, providers=[type("P", (), {
        "name": "t", "api_base": "x", "api_key": "k", "model": "m",
        "max_attempts": 1, "ready": True})()])
    captured = {}

    def fake_chat(messages, providers, max_tokens=None, **kw):
        captured["prompt"] = messages[0]["content"]
        return type("R", (), {"content": GOOD_NOTE})()

    out = generate_note("", "图片里写着部署流程与回滚方案", "图文帖", app, chat_fn=fake_chat)
    assert out == GOOD_NOTE
    assert "图文帖" in captured["prompt"]   # 空转写被换成说明，不是留空让人猜


def test_default_note_fn_writes_frontmatter(tmp_path: Path):
    # Mock provider and chat to generate real note (not draft)
    mock_prov = type('P', (), {'name': 't', 'api_base': 'x', 'api_key': 'k', 'model': 'm', 'max_attempts': 1, 'ready': True})()
    app = make_app(tmp_path, providers=[mock_prov])
    rd = tmp_path / "data" / "fp_abc"
    rd.mkdir(parents=True)
    (rd / "transcript.txt").write_text("足够长的有效转写正文，用来通过闸门的四十个字下限要求，包含具体步骤与数字。" * 2, encoding="utf-8")
    (rd / "vlm.txt").write_text("画面中出现演示界面，字幕提到关键步骤。" * 3, encoding="utf-8")
    (rd / "meta.json").write_text(json.dumps(
        {"key": "fp_abc", "source": "local", "media": "2026-09-16_测试视频_7412345678901234567.mp4"}), encoding="utf-8")

    # Patch generate_note to return GOOD_NOTE via mock chat
    import distill_workshop.core.notes as notes_mod
    orig_gen = notes_mod.generate_note
    def mock_gen(trans, vlm, title, app_, **kw):
        return orig_gen(trans, vlm, title, app_, chat_fn=lambda m, p, **k: type('R', (), {'content': GOOD_NOTE})(), **kw)
    notes_mod.generate_note = mock_gen
    try:
        default_note_fn(app)(rd, {}, app, lambda m: None)
    finally:
        notes_mod.generate_note = orig_gen
    note = (rd / "note.md").read_text(encoding="utf-8")
    assert note.startswith("---") and "title: 测试视频" in note
    assert "7412345678901234567" not in note.split("\n\n")[0]  # 尾部 id 已从标题清洗
