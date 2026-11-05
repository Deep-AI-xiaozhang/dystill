"""自媒体爆款衍生模块测试。"""
from distill_workshop.config import AppConfig
from distill_workshop.core.creative import STYLES, rewrite_note


def test_styles_defined():
    assert "xiaohongshu" in STYLES
    assert "weitoutiao" in STYLES
    assert "script" in STYLES

def test_rewrite_note_unknown_style():
    app = AppConfig(root=None, notes_dir=None, data_dir=None, cache_dir=None)
    try:
        rewrite_note("some note", "unknown_style", app)
        assert False, "应当抛出 ValueError"
    except ValueError as e:
        assert "未知风格" in str(e)
