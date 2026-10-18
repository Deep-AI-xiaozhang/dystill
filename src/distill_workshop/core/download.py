"""下载层（移植自早期实现的下载层实测修复）。

四条铁律，全部来自真实事故：
1. 时间窗口判别（v82 事故）：本次下载结果只认窗口内新出现的 mp4 / 新图片 / 被改写文件，
   绝不回退扫全目录——批量场景会误取上一视频的遗留文件，产出张冠李戴的笔记。
2. manifest 身份复用（2026-09-01 风控兜底）：下载失败时按 aweme_id 查历史记录复用，
   身份校验三重（manifest 记录 + 文件在 before 集合 + 文件名含 aweme_id），展开失败安全放弃。
3. B站必须带 Referer + UA（HTTP 412 实测），且与抖音分流（douyin-downloader 会抓错视频）。
4. 抖音风控先回浏览器（2026-09-17 实测）：毫秒级 403/空 200 是「纯 HTTP 客户端指纹被识破」，
   退避重试救不了，先切真浏览器兜底（core/browser_download.py）再退避重试 API。

架构改进：图文帖用类型化异常 GalleryDetected（替代字符串匹配）；
短链解析器可注入（网络逻辑可测试）；风控型瞬时失败用 RiskControlDetected 与确定性失败分开。
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ..config import AppConfig, DownloadConfig
from ..identity import extract_aweme_id

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
       "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")


class DownloadError(RuntimeError):
    """下载失败。"""


class GalleryDetected(DownloadError):
    """抖音图文帖：无视频轨，应切换图文蒸馏流程。

    images = 本次识别到的图片路径，直接交给图文流程用（省一次目录扫描）。
    """

    def __init__(self, msg: str = "", images: list[Path] | None = None):
        super().__init__(msg)
        self.images = list(images or [])


class DownloaderNotConfigured(DownloadError):
    """douyin-downloader 未配置（需要用户 clone 一份并在 config 指路径）。"""


class RiskControlDetected(DownloadError):
    """命中抖音边缘 WAF（瞬时）：毫秒级 403/429，或 HTTP 200 空 body。

    与「链接坏 / 图文帖」这类确定性失败分开：这类可以重试，而且**先回浏览器才是正解**
    （指纹型 403 退避重试无效，实测三次同速被拒）。
    """


# 抖音从 2026-09-14 起对「单个视频」接口（aweme/detail）上了 Argus 风控，放行所需的
# x-secsdk-web-signature 只能由抖音网页内的 SDK 生成，CLI 直连拿不到。这不是我们的 bug、
# 换 cookie 也没用——实测：完整登录态 cookie 与新鲜游客 cookie 都照样被拒（2026-09-17）。
# 所以这里不写"cookie 可能失效"误导人，给出真正能走通的路。
DOUYIN_BLOCKED_HINT = (
    "抖音对「单个视频」接口（aweme/detail）有 Argus 风控：非浏览器请求一律被拒，"
    "带不带 cookie、重试几次都一样（**不是 cookie 失效**，别在这上面耗）。"
    "本项目已自动尝试真浏览器兜底，走到这里说明兜底也没成。能走通的路：\n"
    "  ① 装上浏览器兜底：pip install -e '.[douyin-browser]' 然后 playwright install chromium\n"
    "  ② 本地文件：自己把视频存下来，再 `dystill run <文件路径>` 直接蒸馏（永远可用）\n"
    "  ③ 换源：同一内容若在 B 站等平台也有，给那个链接（B 站走 yt-dlp，可选组件：pip install 'dystill[bilibili]'）\n"
    "  实测记录见 docs/DECISIONS.md D11 / D13"
)

# 下载器输出里出现这些词即判为风控型失败（对齐 core/api_client.py 的 _RISK_CONTROL_HTTP_STATUSES）
_RISK_HINT_RE = re.compile(
    r"403|429|安全验证|验证码|anti-?bot|risk|Empty 200|风控|请先登录|未登录", re.IGNORECASE)
_DL_BACKOFF_S = (30, 90)   # 第 2、3 次尝试前分别等待（再叠 0~10s 抖动，避免同刻撞限频）

# 下载器真正认的登录态三项：sessionid=登录态，ttwid+msToken=接口必需（对齐 auth/cookie_manager.py）
_COOKIE_KEYS = ("sessionid", "ttwid", "msToken")


def _tell(on_step, msg: str) -> None:
    if on_step:
        on_step(msg)


def run(cmd: list[str], timeout: int = 1200) -> subprocess.CompletedProcess:
    # 下载器内部用 rich 渲染输出；Windows 控制台默认 GBK，rich 打印 ℹ 这类符号时
    # 会直接 UnicodeEncodeError 崩掉（实测 2026-10-13，抖音链接全军覆没）。
    # 给子进程一个 UTF-8 环境即可绕开，与父进程的 _force_utf8_stdio 是两回事。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env, check=False,
                          creationflags=_NO_WINDOW)


# ── 短链 / 身份 ──────────────────────────────────────────────
def expand_short_link(url: str, timeout: int = 15) -> str | None:
    """展开 v.douyin.com 短链，返回最终 URL；失败返回 None（安全放弃，绝不猜）。"""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        return urllib.request.urlopen(req, timeout=timeout).geturl()
    except (OSError, ValueError):
        return None


def aweme_id_from_url(url: str, expand: bool = True) -> str | None:
    """优先直接提取；短链失败时展开后再提取。展开失败返回 None。"""
    aweme = extract_aweme_id(url)
    if aweme or not expand:
        return aweme
    final = expand_short_link(url)
    return extract_aweme_id(final) if final else None


# ── manifest（下载器 append 的 JSONL）──────────────────────
def _manifest_records(dl: Path) -> list[dict]:
    manifest = Path(dl) / "download_manifest.jsonl"
    if not manifest.is_file():
        return []
    out = []
    for ln in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(ln))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def _latest_by_aweme(dl: Path, aweme_id: str) -> dict | None:
    rec = None
    for r in _manifest_records(dl):
        if str(r.get("aweme_id", "")) == aweme_id:
            rec = r  # 保留最后一条（最新）
    return rec


def reuse_downloaded(url: str, dl: Path, before: set[Path],
                     aweme_id: str | None = None) -> Path | None:
    """下载失败时按 manifest 身份校验复用历史 mp4（v82 防换源）。任一环节不满足即放弃。"""
    aweme_id = aweme_id or aweme_id_from_url(url)
    if not aweme_id:
        return None
    rec = _latest_by_aweme(dl, aweme_id)
    if not rec:
        return None
    for fp in rec.get("file_paths", []):
        cand = Path(dl) / fp
        if cand.exists() and cand in before and cand.stat().st_size > 0 and aweme_id in cand.name:
            return cand
    return None


def manifest_gallery_files(url: str, dl: Path, aweme_id: str | None = None) -> list[Path]:
    """按 manifest 身份校验找该 URL 图文帖的历史图片（全在盘才算命中）。"""
    aweme_id = aweme_id or aweme_id_from_url(url)
    if not aweme_id:
        return []
    rec = _latest_by_aweme(dl, aweme_id)
    if not rec or rec.get("media_type") != "gallery":
        return []
    fps = rec.get("file_paths") or []
    paths = []
    for fp in fps:
        cand = Path(dl) / fp
        if cand.exists() and cand.stat().st_size > 0 and aweme_id in cand.name:
            paths.append(cand)
    return paths if paths and len(paths) == len(fps) else []


# ── douyin-downloader ────────────────────────────────────────
def parse_cookie_string(raw: str) -> dict[str, str]:
    """`k=v; k=v; ...` 形态的 cookie 串 → dict（浏览器"复制为 cURL/请求头"拿到的就是这种）。

    为什么需要：抖音的风控要靠**完整登录态**（sessionid/sid_guard/odin_tt 等），
    只给一个游客 ttwid 会被判"未返回作品详情"。而把登录态写进 config.yml 又不合适——
    所以支持 `cookies_file` 指向一个 cookie 文件，凭据只留在那一处。
    """
    out: dict[str, str] = {}
    for part in (raw or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip()
    return out


def _write_distill_config(url: str, dl: Path, dcfg: DownloadConfig, work_dir: Path) -> Path:
    """注入蒸馏专用配置：720p + 只下视频 + 继承 cookies/proxy（凭据不落仓库）。"""
    cfg = {
        "link": [url],
        "path": str(dl),
        "video": True, "music": False, "cover": False, "avatar": False,
        "json": False, "folderstyle": False,
        "filename_template": "{date}_{title}_{id}",
        "author_dir": "sec_uid", "thread": 5, "retry_times": 3,
        "video_quality": "720p", "database": False,
        "transcript": {"enabled": False}, "comments": {"enabled": False},
        "notifications": {"enabled": False}, "browser_fallback": {"enabled": False},
        "live": {"max_duration_seconds": 0},
    }
    if dcfg.config_yml and Path(dcfg.config_yml).is_file():
        try:
            base = yaml.safe_load(Path(dcfg.config_yml).read_text(encoding="utf-8")) or {}
            for k in ("proxy", "cookies"):
                if base.get(k):
                    cfg[k] = base[k]
            # cookies_file：从文件读完整登录态（优先级低于内联 cookies）
            cf = base.get("cookies_file")
            if cf and not cfg.get("cookies") and Path(cf).is_file():
                cfg["cookies"] = parse_cookie_string(Path(cf).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass  # 主配置读不到就无登录态运行，交给上层报错
    out = work_dir / "dl_config.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


# ── cookie 体检（下载器真正认的那份登录态）─────────────────
def _cookie_candidates(dcfg: DownloadConfig, work_dir: Path) -> list[Path]:
    """候选登录态文件，顺序对齐下载器 config_loader._candidate_auto_cookie_paths 的搜索根：
    配置目录（config 目录）、它的上一级、当前 cwd；另加上下载器自己目录下常见位置。

    ⚠️ 下载器默认读的是 `.cookies.json`，**不是** config.yml 里的 `cookies:` 块（2026-09-17
    实测差点因改错文件把可用的登录态换掉）——所以体检必须把两者都看在眼里。
    """
    roots = [Path(work_dir), Path(work_dir).parent, Path.cwd()]
    if dcfg.downloader_dir:
        roots.insert(0, Path(dcfg.downloader_dir))
    out: list[Path] = []
    for root in roots:
        out.extend([root / "config" / "cookies.json", root / ".cookies.json"])
    return out


def cookie_health(dcfg: DownloadConfig, work_dir: Path) -> tuple[bool, str]:
    """下载前 cookie 体检：sessionid=登录态，ttwid+msToken=接口必需，缺任一项风控必命中。

    返回 (是否健康, 说明)。这是**提示性**的（只写进日志/回调），不做门禁：
    实测 cookie 完整也会被风控挡（那是指纹问题，不是登录态问题）。
    """
    first_bad = None
    for cf in _cookie_candidates(dcfg, work_dir):
        if not cf.is_file():
            continue
        try:
            c = json.loads(cf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as e:
            first_bad = first_bad or f"{cf} 读取失败（{e}）"
            continue
        miss = [k for k in _COOKIE_KEYS if not str(c.get(k) or "").strip()]
        if miss:
            first_bad = first_bad or f"{cf} 缺少 {'/'.join(miss)}（共 {len(c)} 条）"
            continue
        return True, f"{cf} 登录态完整（共 {len(c)} 条）"
    if first_bad:
        return False, first_bad
    return False, "未找到 cookie 文件（下载器读 .cookies.json / config/cookies.json）"


# ── 抖音（单次 API 尝试 + 风控自愈编排）─────────────────────
def _download_douyin_api(url: str, dl: Path, dcfg: DownloadConfig, work_dir: Path) -> Path:
    """跑一次 douyin-downloader。风控型失败抛 RiskControlDetected（可重试/可换浏览器），
    其余失败（配置错、链接坏）抛 DownloadError——确定性失败重试没意义。"""
    runner = Path(dcfg.downloader_dir) / "run.py" if dcfg.downloader_dir else None
    if not runner or not runner.is_file():
        raise DownloaderNotConfigured(
            "douyin-downloader 未配置：config.yaml 的 download.downloader_dir 应指向含 run.py 的目录")
    cfg_path = _write_distill_config(url, dl, dcfg, work_dir)
    t0 = time.time()
    before = set(dl.rglob("*.mp4")) | set(dl.rglob("*.jpg")) | set(dl.rglob("*.png"))
    try:
        r = run([sys.executable, str(runner), "-c", str(cfg_path), "-u", url, "-p", str(dl)])
    finally:
        # 生成的配置里可能内联了完整登录态 cookie（凭据只该留在 cookies_file 一处）：
        # 无论成败用后即删，别让凭据在 cache 里躺到下次清理
        cfg_path.unlink(missing_ok=True)
    if r.returncode != 0:
        tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
        if _RISK_HINT_RE.search(tail):
            raise RiskControlDetected("下载器输出命中风控特征: " + tail)
        raise DownloadError("下载失败: " + tail)

    # 铁律 1：只认时间窗口内的产物，禁止扫全目录
    new_mp4 = [p for p in dl.rglob("*.mp4") if p.stat().st_mtime >= t0 - 5]
    if new_mp4:
        return max(new_mp4, key=lambda p: p.stat().st_size)

    new_imgs = [p for p in list(dl.rglob("*.jpg")) + list(dl.rglob("*.png"))
                if p.stat().st_mtime >= t0 - 5 and p not in before]
    if new_imgs:
        raise GalleryDetected(f"图文帖（窗口内新图片 {len(new_imgs)} 张，无视频）",
                              images=sorted(new_imgs))

    aweme = aweme_id_from_url(url)
    if aweme:
        gallery = manifest_gallery_files(url, dl, aweme_id=aweme)
        if gallery:
            raise GalleryDetected("图文帖（manifest 身份校验命中历史图片）", images=gallery)

    rewritten = [p for p in dl.rglob("*.mp4")
                 if p.stat().st_mtime >= t0 - 5 and p in before]  # 重复下载被改写的 mp4
    if rewritten:
        return max(rewritten, key=lambda p: p.stat().st_size)

    # 铁律 2：风控兜底——manifest 身份校验复用历史下载
    reused = reuse_downloaded(url, dl, before, aweme_id=aweme)
    if reused:
        return reused
    # 没产物、也没历史可复用：最常见就是风控（也可能链接失效/需登录），交给上层先试浏览器
    raise RiskControlDetected("未产出新 mp4，且无历史可复用（疑似风控或链接失效）")


def _download_douyin(url: str, dl: Path, dcfg: DownloadConfig, work_dir: Path,
                     on_step=None) -> Path:
    """抖音下载编排：cookie 体检 → API 一次 → **风控先切真浏览器**（指纹型 403 退避无效）
    → 浏览器不行再退避重试 API → 全失败才给可行动提示。
    """
    ok, why = cookie_health(dcfg, work_dir)
    _tell(on_step, f"   cookie 体检：{why}" if ok else f"⚠️ cookie 体检不通过：{why}")
    if not ok:
        _tell(on_step, "   cookie 缺失先补登录态（下载器目录下重新登录）；若 cookie 完整仍被挡，"
                       "就不要再折腾 cookie 了——那是指纹问题（docs/DECISIONS.md D11/D13）")

    last: Exception | None = None
    risk_hits = 0
    for i, wait in enumerate((0,) + _DL_BACKOFF_S, 1):
        if wait:
            pause = wait + random.randint(0, 10)
            _tell(on_step, f"   退避 {pause}s 后重试 API（第 {i}/{1 + len(_DL_BACKOFF_S)} 次尝试）...")
            time.sleep(pause)
        try:
            return _download_douyin_api(url, dl, dcfg, work_dir)
        except RiskControlDetected as e:
            last = e
            risk_hits += 1
            _tell(on_step, f"   风控命中：{e}")
            if risk_hits == 1:
                browser_media = _try_browser_fallback(url, dl, on_step)
                if browser_media:
                    return browser_media
    raise DownloadError(f"抖音下载失败（{last}）。\n" + DOUYIN_BLOCKED_HINT)


def _try_browser_fallback(url: str, dl: Path, on_step=None) -> Path | None:
    """风控兜底：真实Chromium 取播放地址。失败返回 None（不中断编排，让它继续退避重试 API）。"""
    from .browser_download import (
        BrowserFallbackFailed,
        BrowserFallbackUnavailable,
        download_via_browser,
        playwright_available,
    )
    ok, why = playwright_available()
    if not ok:
        _tell(on_step, f"   浏览器兜底不可用：{why}")
        return None
    _tell(on_step, "   ↻ 切真浏览器兜底（绕开纯 HTTP 指纹识别，首次启动约 5-15s）...")
    try:
        media = download_via_browser(url, dl)
    except BrowserFallbackUnavailable as e:
        _tell(on_step, f"   浏览器兜底不可用：{e}")
        return None
    except BrowserFallbackFailed as e:
        _tell(on_step, f"   浏览器兜底失败：{e}")
        return None
    _tell(on_step, f"   ✅ 浏览器兜底成功：{media.name}（{media.stat().st_size // 1024} KB）")
    return media


def _yt_dlp_available() -> bool:
    """yt-dlp 是否可用（独立成函数：可选依赖的可测试接缝）。"""
    return importlib.util.find_spec("yt_dlp") is not None


def _download_bilibili(url: str, dl: Path) -> Path:
    """yt-dlp 下载 B站（≤720p）。铁律 3：必须带 Referer + UA，否则 412。

    yt-dlp 是可选组件（`dystill[bilibili]`）：产品定位只做抖音，不该让所有人背着 B 站下载器。
    没装时不裸抛子进程报错，给可行动的补救路径。
    """
    if not _yt_dlp_available():
        raise DownloadError(
            "B站下载依赖 yt-dlp，它是可选组件、当前没装。能走通的路：\n"
            "  ① 装上再跑：pip install 'dystill[bilibili]'（源码安装：pip install -e '.[bilibili]'）\n"
            "  ② 不装：同一内容若在抖音也有，给抖音链接；或先把视频存到本地，"
            "`dystill run <文件路径>` 直接蒸馏（永远可用）"
        )
    before = {p.resolve() for p in dl.rglob("*.mp4")}
    r = run([sys.executable, "-m", "yt_dlp", "-f", "bv*+ba/b", "-S", "res:720",
             "--merge-output-format", "mp4", "-o", str(dl / "%(id)s_%(title).80s.%(ext)s"),
             "--no-playlist", "--no-progress", "--retries", "3",
             "--referer", "https://www.bilibili.com/", "--user-agent", _UA, url])
    if r.returncode != 0:
        raise DownloadError("B站下载失败: " + (r.stderr or "")[-400:])
    vids = [p for p in dl.rglob("*.mp4") if p.resolve() not in before]
    if not vids:  # yt-dlp 对已存在文件跳过且返回 0：按 BV 号回认，保证重跑幂等
        m = re.search(r"(BV[0-9A-Za-z]{8,12})", url)
        if m:
            vids = [p for p in dl.rglob("*.mp4") if m.group(1).lower() in p.name.lower()]
    if not vids:
        raise DownloadError("B站下载完成但未找到 mp4")
    return max(vids, key=lambda p: p.stat().st_size)


# ── 统一入口 ─────────────────────────────────────────────────
def step_download(url: str, app: AppConfig, on_step=None) -> Path:
    """按平台分流下载，返回媒体文件路径。
    本地文件直接返回；B站走 yt-dlp；抖音走 douyin-downloader（图文帖抛 GalleryDetected，
    撞风控时自动切真浏览器兜底再退避重试）。on_step 是进度回调（默认静默）。"""
    p = Path(url)
    if p.is_file():
        return p
    dl = app.cache_dir / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    from .media import detect_source
    typ = detect_source(url)
    if typ == "bilibili":
        return _download_bilibili(url, dl)
    if typ == "douyin":
        return _download_douyin(url, dl, app.download, app.cache_dir, on_step=on_step)
    raise DownloadError(f"不支持的平台或无法识别的输入: {url}")
