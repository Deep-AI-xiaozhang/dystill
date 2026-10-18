from distill_workshop.core.media import detect_source


def test_detect_douyin():
    assert detect_source("https://v.douyin.com/abcXYZ/") == "douyin"
    assert detect_source("https://www.douyin.com/video/741") == "douyin"


def test_detect_bilibili():
    assert detect_source("https://www.bilibili.com/video/BV1xx") == "bilibili"
    assert detect_source("https://b23.tv/abc") == "bilibili"


def test_local_audio_by_ext():
    assert detect_source("C:/x/song.mp3") == "audio"


def test_unknown():
    assert detect_source("https://example.com/foo") == "unknown"
