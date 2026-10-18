"""STT 设备/精度解析（D38）：CUDA 上不许落回 int8_float16。"""
from distill_workshop.core.stt import resolve_device_compute


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
