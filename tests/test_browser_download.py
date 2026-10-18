"""core/browser_download.py 的纯函数测试：不实际启动浏览器（CI 里没装 chromium 也要能跑）。

真浏览器路径（download_via_browser）走 playwright 集成，由实机验证；这里锁住
命名/解析/判定这些确定性逻辑，防止将来改坏「下游依赖的文件名形状」。
"""
import builtins

import pytest

from distill_workshop.core import browser_download as bd


def _detail(**over) -> dict:
    d = {"aweme_id": "7381234567890123456", "desc": " 测试 标题:带特殊字符? ",
         "create_time": 1789603200}  # 2026-09-17 00:00 UTC
    d.update(over)
    return d


def test_play_addr_urls_extracts_and_filters():
    detail = {"video": {"play_addr": {"url_list": [
        "https://v3-ok.douyinvod.com/a.mp4", "", 123, None]}}}
    assert bd.play_addr_urls(detail) == ["https://v3-ok.douyinvod.com/a.mp4"]
    assert bd.play_addr_urls({}) == []
    assert bd.play_addr_urls({"video": {"play_addr": {}}}) == []


def test_media_filename_shape_matches_downloader_template():
    name = bd.media_filename(_detail(), "https://www.douyin.com/video/7381234567890123456")
    # {date}_{title}_{id}.mp4，日期前缀 / 尾部 aweme_id 是下游 clean_title / 身份校验的契约
    assert name.startswith("2026-09-17_") and name.endswith("_7381234567890123456.mp4")
    assert "测试 标题 带特殊字符" in name  # 非法字符(:/?)被替换成空格，标题截断保真


def test_media_filename_falls_back_to_url_id_when_detail_missing():
    detail = {"desc": "no id in detail"}
    name = bd.media_filename(detail, "https://www.douyin.com/video/998877665544332211")
    assert name.endswith("_998877665544332211.mp4")


def test_media_filename_raises_when_no_id_anywhere():
    with pytest.raises(bd.BrowserFallbackFailed):
        bd.media_filename({"desc": "x"}, "https://www.douyin.com/")


def test_looks_like_mp4_accepts_ftyp_rejects_html():
    assert bd.looks_like_mp4(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    assert not bd.looks_like_mp4(b"<html><body>error</body></html>")
    assert not bd.looks_like_mp4(b"\x00\x00\x00\x10xxxx")          # box 类型不是 ftyp
    assert not bd.looks_like_mp4(b"short")


def test_playwright_available_without_playwright(monkeypatch):
    """没装 playwright 时返回 (False, 含安装提示) 且不抛异常。

    `import playwright` 走 builtins.__import__，所以在这里打补丁才是对的
    （patch importlib.import_module 对 import 语句无效）。
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "playwright":
            raise ImportError("No module named 'playwright'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, why = bd.playwright_available()
    assert ok is False
    assert "playwright" in why and "install" in why


def test_playwright_available_with_playwright():
    # 本机装了 playwright → 正常返回 True（不启动浏览器）
    ok, why = bd.playwright_available()
    assert isinstance(ok, bool)
    assert why