"""STT 设备/精度解析（D38）：CUDA 上不许落回 int8_float16。"""
from pathlib import Path

import pytest

from distill_workshop.core.stt import FasterWhisperSTT, resolve_device_compute


def test_auto_with_cuda_picks_float16():
    """有 CUDA 时必须显式 float16——ctranslate2 的 auto 会选 int8_float16，
    该量化实测会静默丢整段内容（D38）。"""
    dev, comp = resolve_device_compute("auto", "", cuda_available=True)
    assert dev == "cuda"
    assert comp == "float16"


def test_auto_without_cuda_picks_cpu_int8():
    dev, comp = resolve_device_compute("auto", "", cuda_available=False)
    assert dev == "cpu"
    assert comp == "int8"


def test_explicit_device_respected():
    dev, comp = resolve_device_compute("cpu", "", cuda_available=True)
    assert dev == "cpu"
    assert comp == "int8"          # CPU 上 float16 不支持


def test_explicit_compute_type_wins():
    """显存不够时手动降级必须生效（留给用户的逃生口）。"""
    dev, comp = resolve_device_compute("cuda", "int8_float16", cuda_available=True)
    assert dev == "cuda"
    assert comp == "int8_float16"


def test_blank_and_case_insensitive():
    assert resolve_device_compute("", "", cuda_available=True) == ("cuda", "float16")
    assert resolve_device_compute(" CUDA ", "", cuda_available=True) == ("cuda", "float16")


class _Segment:
    def __init__(self, text):
        self.start = 0.0
        self.text = text


class _FlakyModel:
    def __init__(self, fail):
        self.fail = fail

    def transcribe(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return iter([_Segment("CPU 转写成功")]), None


def test_auto_cuda_runtime_error_falls_back_to_cpu(monkeypatch):
    """CUDA 驱动在但 cuBLAS 缺失时，默认 auto 必须自动退回 CPU，而不是让首跑失败。"""
    monkeypatch.setattr("distill_workshop.core.stt.resolve_device_compute",
                        lambda *a, **k: ("cuda", "float16"))
    stt = FasterWhisperSTT(device="auto")
    made = []

    def make_model(device, compute):
        made.append((device, compute))
        return _FlakyModel(fail=device == "cuda")

    monkeypatch.setattr(stt, "_make_model", make_model)
    got = stt.transcribe_raw(Path("audio.wav"))

    assert got == [(0.0, "CPU 转写成功")]
    assert made == [("cuda", "float16"), ("cpu", "int8")]
    assert "已自动改用 CPU" in stt.last_fallback


def test_explicit_cuda_does_not_silently_fallback(monkeypatch):
    """用户明确指定 cuda 时不能悄悄换 CPU；错误必须暴露，避免误以为在用显卡。"""
    stt = FasterWhisperSTT(device="cuda")
    monkeypatch.setattr(stt, "_make_model",
                        lambda device, compute: _FlakyModel(fail=True))
    with pytest.raises(RuntimeError, match="cublas"):
        stt.transcribe_raw(Path("audio.wav"))
