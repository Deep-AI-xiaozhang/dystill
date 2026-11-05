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


def test_browser_response_json_errors_are_silenced():
    class DeadResponse:
        def json(self):
            raise RuntimeError("Target page, context or browser has been closed")

    assert bd.response_json_or_none(DeadResponse()) is None


def test_gallery_image_urls_filters_ui_squares_and_dedupes():
    items = [
        {"src": "https://p.douyinpic.com/avatar/1.jpg", "naturalWidth": 1200, "naturalHeight": 1200},
        {"src": "https://p.douyinpic.com/note/ok1.jpg", "naturalWidth": 1792, "naturalHeight": 2400},
        {"src": "https://p.douyinpic.com/note/icon.png", "naturalWidth": 512, "naturalHeight": 512},
        {"src": "https://p3.douyinpic.com/note/ok1.jpg?x=2", "naturalWidth": 1792, "naturalHeight": 2400},
        {"src": "https://p.douyinpic.com/note/ok2.jpg", "naturalWidth": 1792, "naturalHeight": 2400},
    ]
    got = bd.gallery_image_urls(items)
    assert len(got) == 2
    assert got[0].endswith("ok1.jpg")
    assert got[1].endswith("ok2.jpg")


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


# ── _download_gallery_images 告警测试：失败不再静默 ──────────────

class _FakeRequest:
    """模拟 ctx.request.get，behavior 是响应对象或要抛的异常。"""
    def __init__(self, behaviors: list):
        self._behaviors = list(behaviors)
        self._i = 0

    def get(self, url, timeout=None):
        if self._i >= len(self._behaviors):
            raise OSError("mock: 超出预期调用")
        b = self._behaviors[self._i]
        self._i += 1
        if isinstance(b, Exception):
            raise b
        return b   # 直接返回响应对象，不再包装


class _FakeResponse:
    def __init__(self, body: bytes, ok: bool = True, status: int = 200):
        self._body = body
        self.ok = ok
        self.status = status

    def body(self):
        return self._body


class _FakeCtx:
    def __init__(self, behaviors):
        self.request = _FakeRequest(behaviors)

    def cookies(self):
        return []


class _FakePage:
    def __init__(self, items_result):
        self._items = items_result

    def evaluate(self, _):
        if isinstance(self._items, Exception):
            raise self._items
        return self._items


def test_download_gallery_images_warns_on_all_failure(tmp_path):
    """所有候选图都下载失败时，应通过 on_step 汇总告警，而不是静默返回空列表。"""
    items = [{"src": f"https://p.douyinpic.com/note/ok{i}.jpg",
              "naturalWidth": 1792, "naturalHeight": 2400} for i in range(3)]
    ctx = _FakeCtx([OSError("连接超时")] * 3)
    page = _FakePage(items)
    warnings = []
    result = bd._download_gallery_images(
        ctx, page, tmp_path,
        "https://www.douyin.com/note/7123456789012345678",
        on_step=warnings.append)
    assert result == []
    assert len(warnings) == 4   # 3 张各自告警 + 1 条汇总
    assert any("全部下载失败" in w for w in warnings)


def test_download_gallery_images_warns_on_per_image_http_error(tmp_path):
    """单张图 HTTP 错误时应有对应告警，但不影响其余图片下载。"""
    items = [{"src": "https://p.douyinpic.com/note/bad.jpg", "naturalWidth": 1792, "naturalHeight": 2400},
             {"src": "https://p.douyinpic.com/note/good.jpg", "naturalWidth": 1792, "naturalHeight": 2400}]

    ctx = _FakeCtx([_FakeResponse(b"", ok=False, status=400),
                    _FakeResponse(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 2048)])
    page = _FakePage(items)
    warnings = []
    result = bd._download_gallery_images(
        ctx, page, tmp_path,
        "https://www.douyin.com/note/7123456789012345678",
        on_step=warnings.append)
    assert len(result) == 1                       # 第二张成功
    assert any("HTTP 400" in w for w in warnings)


def test_download_gallery_images_silent_on_page_closed(tmp_path):
    """页面已关闭时静默返回（正常路径），不当作错误喷告警。"""
    page = _FakePage(RuntimeError("Target page has been closed."))
    warnings = []
    result = bd._download_gallery_images(
        _FakeCtx([]), page, tmp_path,
        "https://www.douyin.com/note/7123456789012345678",
        on_step=warnings.append)
    assert result == []
    assert warnings == []   # closed 是正常路径，不告警


def test_download_gallery_images_warns_on_unexpected_js_error(tmp_path):
    """非"页面关闭"的 JS 异常应该告警，不能被静默吞掉。"""
    page = _FakePage(RuntimeError("Unexpected navigation"))
    warnings = []
    bd._download_gallery_images(
        _FakeCtx([]), page, tmp_path,
        "https://www.douyin.com/note/7123456789012345678",
        on_step=warnings.append)
    assert any("JS 执行失败" in w for w in warnings)
