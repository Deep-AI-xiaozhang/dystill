"""媒体类型识别（移植自早期实现的类型识别）。

关键教训（2026-09-06 实测）：抖音把纯音频也存成 .mp4——只看后缀会误走视频分支，
切 0 帧、VLM 白跑。纯音频判定必须在下载后用 ffprobe 复核容器内部流，后缀不可信。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_DOUYIN_RE = re.compile(r"(?i)(douyin\.com|v\.douyin\.com|iesdouyin)")
_BILIBILI_RE = re.compile(r"(?i)(bilibili\.com|b23\.tv)")


def has_no_video_stream(path: Path) -> bool:
    """ffprobe 探测容器内部流：**探到「有流但无 video 流」才算纯音频**。

    探测失败（文件损坏 / 容器不支持 / ffprobe 缺席）必须返回 False，保持视频分支兜底——
    否则任何 ffprobe 解析不了的文件都会被静默当成纯音频，跳过切帧与 VLM，
    产出一份"没有画面理解"的残笔记，且因为 vlm.txt 写了合法标记还会被断点续跑永久复用。
    （2026-09-17 实测回归：test_video_pipeline_end_to_end 断言 VLM 被调用 1 次，实际 0 次）
    """
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=_NO_WINDOW, check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    if p.returncode != 0:
        return False           # 探测失败 ≠ 无视频流
    if not (p.stdout or "").strip():
        # 无 video 流：还要确认容器里确实有别的流（audio），避免"空文件也算纯音频"
        return _has_any_stream(path)
    return False


def _has_any_stream(path: Path) -> bool:
    """容器内是否存在任意流。空/损坏文件返回 False（不当作纯音频）。"""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=_NO_WINDOW, check=False,
        )
        return bool((p.stdout or "").strip())
    except (OSError, subprocess.SubprocessError):
        return False


def is_audio_file(path: Path) -> bool:
    path = Path(path)
    if path.suffix.lower() in AUDIO_EXTS:
        return True
    return has_no_video_stream(path)


def media_duration(path: Path) -> float:
    """ffprobe 取时长（秒）；失败返回 0。"""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=_NO_WINDOW, check=False,
        )
        return float(p.stdout.strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def detect_source(url_or_path: str) -> str:
    """输入分流：audio / local_video / douyin / bilibili / unknown。
    抖音具体是视频还是图文帖，由下载产物判断（短链无法预判）。"""
    u = url_or_path.strip()
    p = Path(u)
    if p.is_file():
        return "audio" if is_audio_file(p) else "local_video"
    # 路径暂不存在（如 dry-run/预检）：按后缀意图分流；真实文件优先走上方 ffprobe 实测
    if p.suffix.lower() in AUDIO_EXTS:
        return "audio"
    if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".ts"}:
        return "local_video"
    if _BILIBILI_RE.search(u):
        return "bilibili"
    if _DOUYIN_RE.search(u):
        return "douyin"
    return "unknown"
