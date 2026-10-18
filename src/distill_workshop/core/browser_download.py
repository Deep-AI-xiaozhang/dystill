"""抖音风控兜底下载：用真浏览器取播放地址，绕开「纯 HTTP 客户端指纹被识破」。

为什么需要这个兜底（实测，2026-09-17）：
抖音边缘 WAF 对 `/aweme/v1/web/aweme/detail/` 的**非浏览器请求**毫秒级挡回
（403 + duration_ms=十几~几十 + risk_control=True），放行需要的网页签名只存在于
真实 JS 环境里。同一 IP、同一时刻对照：下载器 3 次全 403，headless Chromium 打开
视频页能拦到 HTTP 200 / status_code=0 / play_addr×3，15s 下完 8508 KB。

所以这**不是 cookie 失效、也不是 IP 被封**——退避重试救不了指纹型 403，回浏览器才是正解。
本模块把这条已实测走通的路固化下来：headless Chromium（不必有头）拦截 detail 响应 →
取 play_addr → 用页面上下文自己的 request 下 CDN（与页面同 cookie/UA/指纹链）。

playwright 是**可选依赖**：没装不报错崩溃，抛 BrowserFallbackUnavailable 让上层退回 API 重试，
本地文件与 B 站路径完全不受影响。
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

from .download import DownloadError

# playwright 是可选依赖，装法写进错误信息（用户看到提示就能自己补上）
_INSTALL_HINT = "pip install -e '.[douyin-browser]' 然后 playwright install chromium"
_DETAIL_API = "/aweme/v1/web/aweme/detail/"
_MIN_BYTES = 100 * 1024            # 小于这个体积基本是错误页而非视频
_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


class BrowserFallbackUnavailable(DownloadError):
    """兜底能力不可用（playwright / chromium 没装好）：属环境问题，退回 API 重试。"""


class BrowserFallbackFailed(DownloadError):
    """浏览器跑起来了，但没拿到播放地址或 CDN 下载失败。"""


def playwright_available() -> tuple[bool, str]:
    """探测兜底能力是否就绪。只探 python 包，不启动浏览器（启动开销留给真正要用的时候）。"""
    try:
        import playwright  # noqa: F401
    except ImportError as e:
        return False, f"未安装 playwright（{e}）；安装：{_INSTALL_HINT}"
    return True, "playwright 已安装"


def play_addr_urls(detail: dict) -> list[str]:
    """从 detail 响应里取播放地址候选（抖音按 CDN 线路给多条，逐条试）。"""
    video = (detail or {}).get("video") or {}
    urls = (video.get("play_addr") or {}).get("url_list") or []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def media_filename(detail: dict, final_url: str) -> str:
    """落盘命名与下载器的 filename_template `{date}_{title}_{id}` 保持一致。

    下游 notes.clean_title / identity.extract_aweme_id 都依赖这个形状：
    日期前缀会被 clean_title 剥掉当标题，文件名尾部保留 aweme_id 供身份校验。
    """
    aweme_id = str((detail or {}).get("aweme_id") or "").strip()
    if not aweme_id:
        m = re.search(r"/(?:video|note)/(\d{6,})", final_url or "")
        aweme_id = m.group(1) if m else ""
    if not aweme_id:
        raise BrowserFallbackFailed(f"取不到 aweme_id（页面：{final_url}）")
    title = _BAD_FILENAME_CHARS.sub(" ", str((detail or {}).get("desc") or "")).strip()
    title = re.sub(r"\s+", " ", title)[:80] or "未命名"
    created = (detail or {}).get("create_time")
    try:
        date = time.strftime("%Y-%m-%d", time.localtime(int(created))) if created else ""
    except (TypeError, ValueError):
        date = ""
    date = date or time.strftime("%Y-%m-%d")
    return f"{date}_{title}_{aweme_id}.mp4"


def looks_like_mp4(body: bytes) -> bool:
    """ISO BMFF 的 `ftyp` box 在偏移 4，用来挡住「200 但内容是 HTML/JSON 错误页」。

    只判体积是不够的：CDN 错误页也能超过 100KB，存下来会在下游切帧/转写阶段才炸。
    """
    return len(body) >= 12 and body[4:8] == b"ftyp"


def _stream_via_requests(ctx, page, url: str, out: Path) -> tuple[bool, str]:
    """流式下载 CDN（长视频不进内存）。借用浏览器上下文的 cookie 与页面 UA，
    指纹链与页面尽量一致；任何失败返回 (False, 原因)，由调用方回退 ctx.request。

    为什么不直接用 ctx.request：playwright 的 APIResponse 只能 body() 整体读入，
    长视频几百 MB 全进内存会 OOM——流式是刚需，cookie/UA 靠搬过去补。
    """
    try:
        headers = {"Referer": "https://www.douyin.com/"}
        try:
            headers["User-Agent"] = page.evaluate("navigator.userAgent")
        except (AttributeError, RuntimeError, TypeError):
            pass  # 拿不到 UA 就裸请求（CDN 对 UA 通常不敏感，detail 接口才查指纹）
        cookies = "; ".join(f"{c['name']}={c['value']}" for c in ctx.cookies() if c.get("name"))
        if cookies:
            headers["Cookie"] = cookies
        with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            it = r.iter_content(chunk_size=1 << 20)
            head = next(it, b"")
            if len(head) < 12 or head[4:8] != b"ftyp":
                return False, f"不是 mp4 容器（前 16 字节 {head[:16]!r}）"
            with open(out, "wb") as f:
                f.write(head)
                f.writelines(it)
        size = out.stat().st_size
        if size < _MIN_BYTES:
            out.unlink(missing_ok=True)
            return False, f"体积异常（{size} 字节）"
        return True, ""
    except (OSError, requests.RequestException, RuntimeError, TypeError, ValueError) as e:
        out.unlink(missing_ok=True)
        return False, f"{type(e).__name__}: {e}"


def download_via_browser(url: str, dl: Path, *, timeout_ms: int = 90_000) -> Path:
    """打开视频页 → 拦 detail 响应 → 取 play_addr → 下 CDN，返回落盘路径。

    失败一律抛 BrowserFallbackUnavailable（环境问题）或 BrowserFallbackFailed（页面/网络问题），
    让调用方决定是退避重试 API 还是给用户可行动提示。
    注意：产物不写下载器的 download_manifest.jsonl，所以重跑时不会被 manifest 复用命中。
    """
    ok, why = playwright_available()
    if not ok:
        raise BrowserFallbackUnavailable(why)

    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    dl = Path(dl)
    dl.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, args=["--disable-blink-features=AutomationControlled"])
            try:
                ctx = browser.new_context(locale="zh-CN",
                                          viewport={"width": 1440, "height": 900})
                page = ctx.new_page()
                seen: list[dict] = []

                def _on_response(resp):
                    if _DETAIL_API in resp.url:
                        try:
                            seen.append(resp.json())
                        except (ValueError, TypeError):
                            pass  # 非 JSON 响应（如预检/错误页）直接忽略

                page.on("response", _on_response)
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                final_url = page.url
                # detail 接口是页面异步拉的，给它最多 10s；拦到的第一个带 aweme_detail 的响应为准
                for _ in range(20):
                    if any((d or {}).get("aweme_detail") for d in seen):
                        break
                    page.wait_for_timeout(500)
                detail = next(((d or {}).get("aweme_detail") for d in seen
                               if (d or {}).get("aweme_detail")), None)
                if detail is None:
                    raise BrowserFallbackFailed(
                        f"浏览器未能拿到视频详情（页面：{final_url}）；"
                        f"可能是链接失效、需要登录或该作品仅 App 可见")

                urls = play_addr_urls(detail)
                if not urls:
                    raise BrowserFallbackFailed("详情里没有 play_addr（疑似图文帖或受限作品）")
                out = dl / media_filename(detail, final_url)

                last_err = None
                for u in urls:
                    # 先流式（长视频不进内存）；不行再退回 ctx.request 整体读入
                    # （与页面同 cookie/UA/指纹链，实测走通过的那条路，留着当兜底）
                    ok, err = _stream_via_requests(ctx, page, u, out)
                    if ok:
                        return out
                    last_err = err
                    try:
                        resp = ctx.request.get(u, timeout=180_000)
                        if not resp.ok:
                            last_err = f"HTTP {resp.status}"
                            continue
                        body = resp.body()
                        if len(body) < _MIN_BYTES:
                            last_err = f"体积异常（{len(body)} 字节）"
                            continue
                        if not looks_like_mp4(body):
                            last_err = f"不是 mp4 容器（前 16 字节 {body[:16]!r}）"
                            continue
                        out.write_bytes(body)
                        return out
                    except (OSError, PlaywrightError, requests.RequestException, RuntimeError,
                            TypeError, ValueError) as e:
                        last_err = f"{type(e).__name__}: {e}"
                raise BrowserFallbackFailed(f"CDN 下载失败（{last_err}）")
            finally:
                browser.close()
    except (BrowserFallbackUnavailable, BrowserFallbackFailed):
        raise
    except (OSError, PlaywrightError, RuntimeError, ValueError) as e:
        raise BrowserFallbackUnavailable(
            f"浏览器启动失败（{type(e).__name__}: {e}）；若提示缺浏览器：playwright install chromium"
        ) from e
