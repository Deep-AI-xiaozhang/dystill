from distill_workshop.core.media import detect_source


def test_detect_douyin():
    assert detect_source("https://v.douyin.com/abcXYZ/") == "douyin"
    assert detect_source("https://www.douyin.com/video/741") == "douyin"


def test_bilibili_is_no_longer_a_platform():
    """平台范围只有抖音（D57）：B 站链接归到 unknown，由下载层给可行动提示。"""
    assert detect_source("https://www.bilibili.com/video/BV1xx") == "unknown"
    assert detect_source("https://b23.tv/abc") == "unknown"


def test_local_audio_by_ext():
    assert detect_source("C:/x/song.mp3") == "audio"


def test_unknown():
    assert detect_source("https://example.com/foo") == "unknown"
