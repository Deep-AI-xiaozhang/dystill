import json
import subprocess
from pathlib import Path

import pytest

from distill_workshop.config import AppConfig, DownloadConfig
from distill_workshop.core import download as dlmod
from distill_workshop.core.download import (
    DownloaderNotConfigured,
    DownloadError,
    GalleryDetected,
    RiskControlDetected,
    _latest_by_aweme,
    manifest_gallery_files,
    reuse_downloaded,
    step_download,
)


def make_app(tmp_path: Path) -> AppConfig:
    return AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                     data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")


def make_douyin_app(tmp_path: Path, *, config_yml: str = "") -> AppConfig:
    """带一个假下载器目录（只需存在 run.py）的配置。"""
    dd = tmp_path / "downloader"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "run.py").write_text("# fake", encoding="utf-8")
    app = make_app(tmp_path)
    app.download = DownloadConfig(downloader_dir=str(dd), config_yml=config_yml)
    return app


def write_manifest(dl: Path, records: list[dict]):
    dl.mkdir(parents=True, exist_ok=True)
    (dl / "download_manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")


def test_latest_by_aweme_keeps_last_record(tmp_path: Path):
    dl = tmp_path / "dl"
    write_manifest(dl, [
        {"aweme_id": "111", "file_paths": ["old.mp4"]},
        {"aweme_id": "111", "file_paths": ["new.mp4"]},
        {"aweme_id": "222", "file_paths": ["other.mp4"]},
    ])
    rec = _latest_by_aweme(dl, "111")
    assert rec["file_paths"] == ["new.mp4"]


def test_reuse_downloaded_identity_checked(tmp_path: Path):
    dl = tmp_path / "dl"
    f = dl / "aweme_111_title.mp4"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x" * 100)
    write_manifest(dl, [{"aweme_id": "111", "file_paths": [f.name]}])
    # 在 before 集合中 + 文件名含 aweme_id → 命中
    assert reuse_downloaded("https://v.douyin.com/x/", dl, {f}, aweme_id="111") == f
    # 不在 before 集合 → 拒绝复用（防引入别的视频产物）
    assert reuse_downloaded("https://v.douyin.com/x/", dl, set(), aweme_id="111") is None


def test_reuse_downloaded_without_identity_gives_up(tmp_path: Path):
    dl = tmp_path / "dl"
    write_manifest(dl, [{"aweme_id": "111", "file_paths": ["aweme_111_a.mp4"]}])
    # 无法取得 aweme_id（模拟短链展开失败）→ 安全放弃，绝不扫全目录
    assert reuse_downloaded("https://v.douyin.com/x/", dl, set(), aweme_id=None) is None


def test_manifest_gallery_files_all_or_nothing(tmp_path: Path):
    dl = tmp_path / "dl"
    f1 = dl / "aweme_111_p1.jpg"
    f2 = dl / "aweme_111_p2.jpg"
    dl.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x")
    write_manifest(dl, [{"aweme_id": "111", "media_type": "gallery",
                         "file_paths": [f1.name, f2.name]}])
    # 只有一张在盘 → 不完整，返回 []（宁可不命中，不可错命中）
    assert manifest_gallery_files("https://www.douyin.com/note/111", dl, aweme_id="111") == []
    f2.write_bytes(b"x")
    assert len(manifest_gallery_files("https://www.douyin.com/note/111", dl, aweme_id="111")) == 2


def test_step_download_local_file_passthrough(tmp_path: Path):
    app = make_app(tmp_path)
    media = tmp_path / "local.mp4"
    media.write_bytes(b"x" * 10)
    assert step_download(str(media), app) == media


def test_step_download_douyin_requires_downloader(tmp_path: Path):
    app = make_app(tmp_path)  # download.downloader_dir 为空
    try:
        step_download("https://v.douyin.com/x/", app)
        assert False, "应当抛 DownloaderNotConfigured"
    except DownloaderNotConfigured:
        pass


# ── cookie 体检（下载前提示，不做门禁）─────────────────────────
def test_cookie_candidates_prefers_downloader_dir(tmp_path: Path):
    dcfg = DownloadConfig(downloader_dir=str(tmp_path / "dd"))
    cands = dlmod._cookie_candidates(dcfg, tmp_path / "cache")
    assert cands[0] == tmp_path / "dd" / "config" / "cookies.json"
    # 四种位置都在：下载器目录、配置目录、配置目录上一级、cache 目录
    assert tmp_path / "dd" / ".cookies.json" in cands
    assert tmp_path / "cache" / ".cookies.json" in cands
    assert tmp_path / ".cookies.json" in cands


def test_cookie_health_checks_three_required_keys(tmp_path: Path, monkeypatch):
    cf = tmp_path / "dd" / ".cookies.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dlmod, "_cookie_candidates", lambda d, w: [cf])
    dcfg = DownloadConfig(downloader_dir=str(tmp_path / "dd"))

    ok, why = dlmod.cookie_health(dcfg, tmp_path / "cache")
    assert not ok and "未找到" in why

    cf.write_text(json.dumps({"ttwid": "t", "msToken": "m"}), encoding="utf-8")
    ok, why = dlmod.cookie_health(dcfg, tmp_path / "cache")
    assert not ok and "sessionid" in why  # 缺登录态 → 报出缺哪一项

    cf.write_text(json.dumps({"sessionid": "s", "ttwid": "t", "msToken": "m"}), encoding="utf-8")
    ok, why = dlmod.cookie_health(dcfg, tmp_path / "cache")
    assert ok and "登录态完整" in why

    cf.write_text("{ not json", encoding="utf-8")
    ok, why = dlmod.cookie_health(dcfg, tmp_path / "cache")
    assert not ok and "读取失败" in why


# ── 风控分类：瞬时失败 vs 确定性失败 ─────────────────────────
def _proc(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_api_attempt_classifies_risk_control(tmp_path: Path, monkeypatch):
    app = make_douyin_app(tmp_path)
    monkeypatch.setattr(dlmod, "run", lambda *a, **k: _proc(
        1, stderr="403 Forbidden duration_ms=31 risk_control=True"))
    with pytest.raises(RiskControlDetected):
        dlmod._download_douyin_api("https://v.douyin.com/x/",
                                   app.cache_dir / "dl", app.download, app.cache_dir)


def test_api_attempt_keeps_deterministic_failure_separate(tmp_path: Path, monkeypatch):
    app = make_douyin_app(tmp_path)
    monkeypatch.setattr(dlmod, "run", lambda *a, **k: _proc(1, stderr="bad config: 缺少 link"))
    with pytest.raises(DownloadError) as e:
        dlmod._download_douyin_api("https://v.douyin.com/x/",
                                   app.cache_dir / "dl", app.download, app.cache_dir)
    assert not isinstance(e.value, RiskControlDetected)  # 确定性失败不许触发重试/兜底


def test_api_attempt_no_product_is_risk_control(tmp_path: Path, monkeypatch):
    """下载器静默失败（returncode=0 但零产物）也要归为风控型，否则不会走兜底。

    用带 aweme_id 的完整链接，免得测试期去展开短链（那会发真实网络请求）。
    """
    app = make_douyin_app(tmp_path)
    monkeypatch.setattr(dlmod, "run", lambda *a, **k: _proc(0))
    with pytest.raises(RiskControlDetected):
        dlmod._download_douyin_api("https://www.douyin.com/video/7381234567890123456",
                                   app.cache_dir / "dl", app.download, app.cache_dir)


# ─ 编排顺序：风控先回浏览器，再退避重试 API ──────────────────
def test_browser_fallback_tried_first_without_sleep(tmp_path: Path, monkeypatch):
    app = make_douyin_app(tmp_path)
    media = tmp_path / "dl" / "2026-09-17_标题_7381234567890123456.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"x" * 200_000)
    calls = {"api": 0}
    slept: list[float] = []

    def fake_api(*a, **k):
        calls["api"] += 1
        raise RiskControlDetected("毫秒级 403")

    monkeypatch.setattr(dlmod, "cookie_health", lambda d, w: (True, "测试用登录态"))
    monkeypatch.setattr(dlmod, "_download_douyin_api", fake_api)
    monkeypatch.setattr(dlmod, "_try_browser_fallback", lambda url, d, on_step=None: media)
    monkeypatch.setattr(dlmod.time, "sleep", lambda s: slept.append(s))

    assert dlmod._download_douyin("https://v.douyin.com/x/",
                                  app.cache_dir / "dl", app.download, app.cache_dir) == media
    # 指纹型 403 退避无效 → 第一次风控就直接兜底，不该先睡 30s
    assert calls["api"] == 1 and slept == []


def test_backoff_retries_only_after_browser_fallback_fails(tmp_path: Path, monkeypatch):
    app = make_douyin_app(tmp_path)
    media = tmp_path / "dl" / "ok.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"x")
    seq: list[str] = []
    slept: list[float] = []

    def fake_api(*a, **k):
        seq.append("api")
        if len([s for s in seq if s == "api"]) < 3:
            raise RiskControlDetected("403")
        return media

    monkeypatch.setattr(dlmod, "cookie_health", lambda d, w: (False, "缺 sessionid"))
    monkeypatch.setattr(dlmod, "_download_douyin_api", fake_api)
    monkeypatch.setattr(dlmod, "random",
                        type("R", (), {"randint": staticmethod(lambda a, b: 0)}))
    monkeypatch.setattr(dlmod.time, "sleep", lambda s: slept.append(s))

    def fake_browser(url, d, on_step=None):
        seq.append("browser")
        # 兜底不可用

    monkeypatch.setattr(dlmod, "_try_browser_fallback", fake_browser)
    assert dlmod._download_douyin("https://v.douyin.com/x/",
                                  app.cache_dir / "dl", app.download, app.cache_dir) == media
    assert seq == ["api", "browser", "api", "api"]      # 只兜底一次
    assert slept == [30, 90]                              # 两次退避，不叠抖动


def test_all_failed_gives_actionable_hint(tmp_path: Path, monkeypatch):
    app = make_douyin_app(tmp_path)
    monkeypatch.setattr(dlmod, "cookie_health", lambda d, w: (True, "测试用登录态"))
    monkeypatch.setattr(dlmod, "_download_douyin_api",
                        lambda *a, **k: (_ for _ in ()).throw(RiskControlDetected("403")))
    monkeypatch.setattr(dlmod, "_try_browser_fallback", lambda url, d, on_step=None: None)
    monkeypatch.setattr(dlmod.time, "sleep", lambda s: None)
    with pytest.raises(DownloadError) as e:
        dlmod._download_douyin("https://v.douyin.com/x/",
                               app.cache_dir / "dl", app.download, app.cache_dir)
    msg = str(e.value)
    assert "douyin-browser" in msg and "playwright install chromium" in msg
    assert "不是 cookie 失效" in msg


def test_gallery_and_deterministic_errors_not_retried(tmp_path: Path, monkeypatch):
    """图文帖/配置错这类确定性结果必须原样上抛，不许被兜底与重试包装。"""
    app = make_douyin_app(tmp_path)
    monkeypatch.setattr(dlmod, "cookie_health", lambda d, w: (True, "测试用登录态"))
    monkeypatch.setattr(dlmod.time, "sleep",
                        lambda s: (_ for _ in ()).throw(AssertionError("不该退避")))
    monkeypatch.setattr(dlmod, "_try_browser_fallback",
                        lambda url, d, on_step=None: (_ for _ in ()).throw(AssertionError("不该兜底")))
    for exc in (GalleryDetected("图文帖"), DownloaderNotConfigured("未配置")):
        monkeypatch.setattr(dlmod, "_download_douyin_api",
                            lambda *a, _e=exc, **k: (_ for _ in ()).throw(_e))
        with pytest.raises(type(exc)):
            dlmod._download_douyin("https://v.douyin.com/x/",
                                   app.cache_dir / "dl", app.download, app.cache_dir)


def test_bilibili_without_yt_dlp_gives_install_hint(tmp_path: Path, monkeypatch):
    """yt-dlp 是可选组件（bilibili extra）：没装必须报可行动补救，不裸抛子进程报错。"""
    monkeypatch.setattr(dlmod, "_yt_dlp_available", lambda: False)
    monkeypatch.setattr(dlmod, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("缺依赖就不该起子进程")))
    with pytest.raises(DownloadError) as e:
        dlmod._download_bilibili("https://www.bilibili.com/video/BV1xx411c7sd/", tmp_path / "dl")
    msg = str(e.value)
    assert "dystill[bilibili]" in msg
    assert "本地文件" in msg or "dystill run" in msg      # 不装也有路可走


