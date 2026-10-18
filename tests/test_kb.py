"""知识库模块：解析、归一化、索引生成、自检。

重点覆盖三个实测踩到的坑：
  1. 类型字段被模板占位符/Markdown 加粗污染（`<实操教程>` / `**观点认知**`）
  2. 索引文件被自己当成笔记收录（第二次运行越滚越大）
  3. 关键词统计混入"视频核心""结论是"这类模板结构词
"""
from pathlib import Path

from distill_workshop.core.kb import (
    INDEX_NAME,
    UNMATCHED_GROUP,
    build_index,
    check_kb,
    find_notes,
    group_by_topic,
    normalize_type,
    parse_note,
    top_keywords,
    upsert_frontmatter,
)

NOTE_TMPL = """---
title: {title}
key: {key}
source: douyin
date: {date}
---
# 蒸馏笔记 · {title}

> 蒸馏日期：{date} ｜ 类型：{ntype} ｜ 主题：{topic}

## 一、结论先行
- {core}

## 二、视频内容
{body}
"""


def _write(p: Path, title="示例标题 #AI #工具", key="aweme:123", date="2026-10-01",
           ntype="实操教程", topic="一句话主题", core="核心结论一句",
           body="正文内容" * 60) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(NOTE_TMPL.format(title=title, key=key, date=date, ntype=ntype,
                                  topic=topic, core=core, body=body), encoding="utf-8")
    return p


def test_normalize_type_strips_placeholder_and_bold():
    assert normalize_type("<实操教程>") == "实操教程"
    assert normalize_type("**观点认知**") == "观点认知"
    assert normalize_type("`实操教程`") == "实操教程"
    assert normalize_type("<观点认知+实操指南>") == "观点认知"   # 含已知类型即归位
    assert normalize_type("  实操教程  ") == "实操教程"


def test_normalize_type_keeps_unknown_verbatim():
    """不认识的值原样保留——不猜、不吞信息（索引层不做价值判断）。"""
    assert normalize_type("带货口播") == "带货口播"
    assert normalize_type("") == "未分类"


def test_parse_note_fields(tmp_path):
    p = _write(tmp_path / "aweme_1" / "note.md")
    m = parse_note(p)
    assert m.key == "aweme:123"
    assert m.title == "示例标题 #AI #工具"
    assert m.date == "2026-10-01"
    assert m.source == "douyin"
    assert m.ntype == "实操教程"
    assert m.topic == "一句话主题"
    assert m.core == "核心结论一句"
    assert m.tags == ["AI", "工具"]
    assert m.valid_key


def test_parse_note_survives_broken_file(tmp_path):
    """坏文件不能抛异常——它要能进自检报告。"""
    p = tmp_path / "aweme_bad" / "note.md"
    p.parent.mkdir(parents=True)
    p.write_text("# 蒸馏笔记 · 视频\n\n> 类型：<实操教程>\n", encoding="utf-8")
    m = parse_note(p)
    assert m.title == "视频"
    assert m.key == ""
    assert not m.valid_key
    assert m.ntype == "实操教程"


def test_find_notes_excludes_index_itself(tmp_path):
    """索引文件不能被当成笔记——否则第二次运行会把上次的索引收进去。"""
    data = tmp_path / "data"
    notes = tmp_path / "notes"
    _write(data / "aweme_1" / "note.md")
    notes.mkdir(parents=True)
    (notes / INDEX_NAME).write_text("# 知识库 · 索引\n", encoding="utf-8")
    metas = find_notes(data, notes)
    assert len(metas) == 1
    assert all(m.path.name != INDEX_NAME for m in metas)


def test_find_notes_dedupes_same_file(tmp_path):
    data = tmp_path / "data"
    _write(data / "aweme_1" / "note.md")
    metas = find_notes(data, data)      # notes_dir == data_dir 不应重复收录
    assert len(metas) == 1


def test_top_keywords_drops_structural_words(tmp_path):
    """模板结构词（视频核心/结论是）与功能词不能进主题榜。"""
    metas = []
    for i, kw in enumerate(("开源工具", "开源项目", "开源社区"), start=1):
        metas.append(parse_note(_write(tmp_path / f"aweme_{i}" / "note.md",
                                       title=f"{kw} 视频核心", topic="结论是 这个")))
    words = [w for w, _ in top_keywords(metas)]
    assert "开源" in words or "开源工具" in words
    assert not any(w in ("视频核心", "结论是", "这个") for w in words)


def test_top_keywords_needs_cross_document_hits(tmp_path):
    """只在单篇出现的词不算主题（阈值 2 篇）。"""
    metas = [
        parse_note(_write(tmp_path / "aweme_1" / "note.md", title="苹果手机", topic="香蕉")),
        parse_note(_write(tmp_path / "aweme_2" / "note.md", title="橙子果汁", topic="葡萄")),
        parse_note(_write(tmp_path / "aweme_3" / "note.md", title="西瓜田地", topic="草莓")),
    ]
    assert top_keywords(metas) == []


def test_group_by_topic_falls_back_to_other(tmp_path):
    m = parse_note(_write(tmp_path / "aweme_1" / "note.md", title="无关标题", topic="无关主题"))
    groups = group_by_topic([m], [("命中词", 5)])
    assert list(groups) == [UNMATCHED_GROUP]


def test_group_by_topic_multi_membership(tmp_path):
    """一篇笔记命中多个主题时应在各组都出现——不做"只归一类"（否则会掉进黑洞）。"""
    m = parse_note(_write(tmp_path / "aweme_1" / "note.md", title="开源工具 自动化", topic="开源"))
    groups = group_by_topic([m], [("开源", 2), ("自动化", 2)])
    assert groups["开源"] == [m]
    assert groups["自动化"] == [m]
    assert UNMATCHED_GROUP not in groups


def test_group_by_topic_is_case_insensitive(tmp_path):
    """英文关键词统计时转了小写，标题里常是 "AI"——匹配必须大小写不敏感。"""
    m = parse_note(_write(tmp_path / "aweme_1" / "note.md", title="热门 AI Agent 全对比", topic=""))
    groups = group_by_topic([m], [("ai", 2)])
    assert groups["ai"] == [m]
    assert UNMATCHED_GROUP not in groups


def test_top_keywords_drops_particle_fragments(tmp_path):
    """切词碎片（"视频的""与使用"）不是主题词。"""
    metas = [parse_note(_write(tmp_path / f"aweme_{i}" / "note.md",
                                 title="视频的核心 安装与使用方法", topic="与使用"))
             for i in range(1, 4)]
    words = [w for w, _ in top_keywords(metas)]
    assert not any(w.startswith("与") or w.endswith("的") for w in words)


def test_build_index_sections_and_count(tmp_path):
    metas = [parse_note(_write(tmp_path / f"aweme_{i}" / "note.md",
                                 title=f"开源工具 {i}", key=f"aweme:{i}"))
             for i in range(1, 4)]
    out = build_index(metas, tmp_path / INDEX_NAME, "2026-10-14")
    assert "收录 **3** 篇" in out
    for sec in ("## 一、总览", "## 三、按主题归组", "## 四、全量索引"):
        assert sec in out
    assert "实操教程" in out


def test_check_kb_reports_missing_key_and_shell(tmp_path):
    data = tmp_path / "data"
    _write(data / "aweme_1" / "note.md", key="")                      # 缺键
    _write(data / "aweme_2" / "note.md", key="aweme:2", body="短")    # 空壳
    metas = find_notes(data)
    issues = check_kb(metas, data, tmp_path / INDEX_NAME)
    kinds = {i.kind for i in issues}
    assert "缺身份键" in kinds
    assert "空壳笔记" in kinds
    assert any(i.level == "ERROR" for i in issues)


def test_check_kb_reports_duplicate_and_material_without_note(tmp_path):
    data = tmp_path / "data"
    _write(data / "aweme_1" / "note.md", key="aweme:same")
    _write(data / "aweme_2" / "note.md", key="aweme:same")
    orphan = data / "aweme_9"
    orphan.mkdir(parents=True)
    (orphan / "transcript.txt").write_text("有转写没笔记", encoding="utf-8")
    metas = find_notes(data)
    issues = check_kb(metas, data, tmp_path / INDEX_NAME)
    kinds = {i.kind for i in issues}
    assert "重复笔记" in kinds
    assert "有素材无笔记" in kinds


def test_check_kb_reports_stale_index(tmp_path):
    data = tmp_path / "data"
    _write(data / "aweme_1" / "note.md")
    _write(data / "aweme_2" / "note.md", key="aweme:2")
    idx = tmp_path / INDEX_NAME
    idx.write_text("> 收录 **1** 篇\n", encoding="utf-8")
    issues = check_kb(find_notes(data), data, idx)
    assert any(i.kind == "索引过期" for i in issues)


def test_upsert_frontmatter_is_idempotent_and_keeps_user_fields(tmp_path):
    p = _write(tmp_path / "aweme_1" / "note.md")
    m = parse_note(p)
    # 用户手写的字段
    text = p.read_text(encoding="utf-8").replace("source: douyin", "source: douyin\nmy_note: 手写内容")
    p.write_text(text, encoding="utf-8")
    m = parse_note(p)

    assert upsert_frontmatter(p, m) is True
    once = p.read_text(encoding="utf-8")
    assert upsert_frontmatter(p, m) is False        # 第二次无变化
    assert p.read_text(encoding="utf-8") == once
    assert "my_note: 手写内容" in once              # 用户字段保留
    assert "type: 实操教程" in once
    assert "topic: 一句话主题" in once
    assert "tags: [AI, 工具]" in once


def test_upsert_frontmatter_creates_block_when_absent(tmp_path):
    p = tmp_path / "aweme_1" / "note.md"
    p.parent.mkdir(parents=True)
    p.write_text("# 蒸馏笔记 · 标题\n\n正文\n", encoding="utf-8")
    m = parse_note(p)
    assert upsert_frontmatter(p, m) is True
    assert p.read_text(encoding="utf-8").startswith("---\n")
