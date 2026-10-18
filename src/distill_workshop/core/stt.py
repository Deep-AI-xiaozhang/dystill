"""STT 适配层：后端可插拔（默认 faster-whisper 本地，无需 key）。

移植自早期实现的三块实测修复：
1. 分段转写：>15min 音频 funasr/whisper 一次性大分配 OOM → ffmpeg 切 10min 段逐段转写拼接
2. VAD 重试：faster-whisper 默认 vad_filter 会把纯音乐/音效整段判静音 → 正文过短时关 VAD 重试一次
3. 缓存闸门：无效转写绝不写缓存（见 artifacts.py），坏缓存永久复用是 v226 事故根因
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from .artifacts import read_if_valid, strip_timestamps, write_if_valid
from .media import media_duration

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SEGMENT_SECONDS = 600          # 10min 一段
_LONG_THRESHOLD = 900           # >15min 才分段（实测 15min 内整段更稳）


class BaseSTT(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe_raw(self, wav: Path, vad_filter: bool = True) -> list[tuple[float, str]]:
        """返回 [(起始秒, 文本)]；由基类负责拼装与重试策略。"""


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, OSError, RuntimeError):
        return False


def resolve_device_compute(device: str = "auto", compute_type: str = "",
                           cuda_available: bool | None = None) -> tuple[str, str]:
    """把 device/compute_type 解析成 ctranslate2 认的显式值。

    **为什么不交给 ctranslate2 的 "auto"**：它在 CUDA 上会选 `int8_float16`，
    实测该量化会静默吃掉整段内容（646s 音频丢 29 秒，转写文件不报错、字数看着正常）。
    本工具的核对引擎以转写为证据源，丢字会直接制造假 ❌ —— 必须显式指定。

    规则：CUDA → float16（准确优先，只比 int8 慢约 8%）；CPU → int8。
    显式传了 compute_type 就原样用（留给显存不足时手动降级）。
    """
    if cuda_available is None:
        cuda_available = _cuda_available()
    dev = (device or "auto").strip().lower()
    if dev == "auto":
        dev = "cuda" if cuda_available else "cpu"
    comp = (compute_type or "").strip()
    if not comp:
        comp = "float16" if dev == "cuda" else "int8"
    return dev, comp


class FasterWhisperSTT(BaseSTT):
    name = "faster-whisper"

    def __init__(self, model_size: str = "small",
                 device: str = "auto", compute_type: str = ""):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # 延迟导入：未安装时给出明确报错
            device, compute = resolve_device_compute(self.device, self.compute_type)
            self._model = WhisperModel(self.model_size, device=device, compute_type=compute)
        return self._model

    def transcribe_raw(self, wav: Path, vad_filter: bool = True) -> list[tuple[float, str]]:
        model = self._ensure_model()
        segments, _info = model.transcribe(
            str(wav), vad_filter=vad_filter, language="zh",
            # 关掉上文条件：开着它模型会顺着上文"编"，是复读/幻觉的主要来源（同 D38）
            condition_on_previous_text=False,
        )
        return [(seg.start, seg.text.strip()) for seg in segments]


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0, int(seconds)), 60)
    return f"[{m:02d}:{s:02d}]"


def _join(parts: list[tuple[float, str]]) -> str:
    return "\n".join(f"{_fmt_ts(t)}{txt}" for t, txt in parts if txt)


def _extract_wav(src: Path, dst: Path) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
            capture_output=True, timeout=1200, creationflags=_NO_WINDOW, check=False,
        )
        return dst.exists() and dst.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        return False


def _segment_wav(src: Path, seg_dir: Path) -> list[Path]:
    seg_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-f", "segment", "-segment_time", str(_SEGMENT_SECONDS),
         "-ac", "1", "-ar", "16000", str(seg_dir / "seg_%03d.wav")],
        capture_output=True, timeout=1200, creationflags=_NO_WINDOW, check=False,
    )
    return sorted(seg_dir.glob("seg_*.wav"))


def _transcribe_one(stt: BaseSTT, wav: Path) -> list[tuple[float, str]]:
    """单文件转写 + VAD 重试：默认开 VAD；正文过短时关 VAD 重试一次，只在新结果更长时采纳。"""
    parts = stt.transcribe_raw(wav, vad_filter=True)
    if len(strip_timestamps(_join(parts)).strip()) < 40:
        retry = stt.transcribe_raw(wav, vad_filter=False)
        if len(_join(retry)) > len(_join(parts)):
            parts = retry
    return parts


def transcribe_media(media: Path, stt: BaseSTT, work_dir: Path) -> str:
    """媒体 → 带时间戳转写文本。>15min 自动分段；段级时间戳偏移拼接。"""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "audio.wav"
    if not wav.exists() and not _extract_wav(media, wav):
        wav = media  # 提取失败直接对原文件转写

    if media_duration(wav) <= _LONG_THRESHOLD:
        parts = _transcribe_one(stt, wav)
    else:
        segs = _segment_wav(wav, work_dir / "segments")
        parts = []
        for i, seg in enumerate(segs):
            offset = i * _SEGMENT_SECONDS
            parts.extend((t + offset, txt) for t, txt in _transcribe_one(stt, seg))
    return _join(parts)


def step_transcribe(media: Path, stt: BaseSTT, cache_dir: Path) -> str:
    """转写步骤（产物驱动断点续跑）：缓存先过有效性闸；无效结果绝不写缓存。"""
    cache_dir = Path(cache_dir)
    cache = cache_dir / "transcript.txt"
    cached = read_if_valid(cache, "transcript")
    if cached is not None:
        return cached
    text = transcribe_media(media, stt, cache_dir)
    if not write_if_valid(cache, text, "transcript"):
        pass  # 过短/失败：不写缓存，下次自动重转（v226 纪律）
    return text
