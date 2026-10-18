from distill_workshop.identity import extract_aweme_id, record_key


def test_extract_aweme_id_from_video_url():
    assert extract_aweme_id("https://www.douyin.com/video/7412345678901234567") == "7412345678901234567"


def test_extract_aweme_id_from_note_url():
    assert extract_aweme_id("https://www.douyin.com/note/7412345678901234567") == "7412345678901234567"


def test_extract_aweme_id_from_query():
    assert extract_aweme_id("https://www.douyin.com/?aweme_id=7412345678901234567") == "7412345678901234567"


def test_short_link_returns_none_until_expanded():
    # 短链必须先展开再提取——这是下载层职责，身份层不做网络请求
    assert extract_aweme_id("https://v.douyin.com/abcXYZ/") is None


def test_record_key_prefers_aweme_id():
    assert record_key(aweme_id="123", fingerprint="abc").startswith("aweme:")


def test_record_key_falls_back_to_fingerprint():
    assert record_key(fingerprint="abc") == "fp:abc"
