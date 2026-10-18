import os
import time
from pathlib import Path

from distill_workshop.core.artifacts import (
    cache_media_files,
    cleanup_cache,
    cleanup_transcribe_scratch,
    is_stale_dir,
    read_if_valid,
    record_dir,
    strip_timestamps,
    transcribe_scratch,
    transcript_valid,
    write_if_valid,
)


def test_transcript_valid_rejects_junk():
    assert not transcript_valid("")
    assert not transcript_valid("[00:00] 字幕君")          # v226 实测坏缓存
    assert not transcript_valid("[00:04] 字幕by索兰娅")      # v204 字幕组署名


def test_transcript_valid_accepts_real_content():
    assert transcript_valid("[00:01] 今天讲一下 Agent 开发的四大硬核能力，第一个是业务拆解，把一个模糊的产品想法拆成可以独立验证的小任务。")
    assert transcript_valid("没有时间戳但足够长的正文也应该通过闸门，这一句是为了凑满四十个字的长度要求而写的补充说明文字。")


def test_strip_timestamps():
    assert strip_timestamps("[00:15]你好[01:02:30]世界") == "你好世界"


def test_write_if_valid_blocks_bad_artifact(tmp_path: Path):
    p = tmp_path / "transcript.txt"
    assert write_if_valid(p, "[00:00] 短", "transcript") is False
    assert not p.exists()                                  # 无效结果绝不落盘
    assert write_if_valid(p, "足够长的有效正文" * 10, "transcript") is True
    assert read_if_valid(p, "transcript") is not None


def test_cache_gate_rejects_persisted_junk(tmp_path: Path):
    p = tmp_path / "transcript.txt"
    p.write_text("[00:00] 字幕君", encoding="utf-8")         # 模拟历史坏缓存
    assert read_if_valid(p, "transcript") is None           # 闸门拦截，调用方将重转


def test_record_dir_naming(tmp_path: Path):
    d1 = record_dir(tmp_path, "aweme:7412345678901234567")
    assert d1.name == "aweme_7412345678901234567"
    d2 = record_dir(tmp_path, "fp:" + "ab" * 32)
    assert d2.name.startswith("fp_") and len(d2.name) == 15  # fp_ + 12 位


def test_stale_dir_detection(tmp_path: Path):
    d = tmp_path / "aweme_1"
    d.mkdir()
    (d / "meta.json").write_text("{}", encoding="utf-8")
    assert is_stale_dir(d)                                  # 仅 meta = 失败残留
    (d / "transcript.txt").write_text("x" * 100, encoding="utf-8")
    assert not is_stale_dir(d)


def _age(p: Path, days: float) -> None:
    ts = time.time() - days * 86400
    os.utime(p, (ts, ts))


def test_cache_cleanup_keeps_recent_and_spares_credentials(tmp_path: Path):
    dl = tmp_path / "dl"
    dl.mkdir()
    fresh = dl / "fresh.mp4"
    old = dl / "old.mp4"
    for p in (fresh, old):
        p.write_bytes(b"x" * 1024)
    _age(fresh, 1)
    _age(old, 30)
    # 凭据在 config/ 下，绝不能被当成缓存删掉
    cred = tmp_path / "config" / "cookies.json"
    cred.parent.mkdir()
    cred.write_text("{}", encoding="utf-8")
    _age(cred, 30)

    doomed = cache_media_files(tmp_path, days=7)
    assert doomed == [old]                       # 只挑出过期媒体
    assert cred not in doomed                    # 凭据不在名单里

    cleanup_cache(tmp_path, days=7, apply=False)  # 默认只报告
    assert old.exists()

    cleanup_cache(tmp_path, days=7, apply=True)
    assert not old.exists()
    assert fresh.exists()                        # 7 天内的保留
    assert cred.exists()                         # 凭据毫发无损


def test_cache_cleanup_disabled_when_days_zero(tmp_path: Path):
    dl = tmp_path / "dl"
    dl.mkdir()
    old = dl / "old.mp4"
    old.write_bytes(b"x")
    _age(old, 999)
    assert cache_media_files(tmp_path, days=0) == []   # 0 = 关闭
    cleanup_cache(tmp_path, days=0, apply=True)
    assert old.exists()


def test_cache_cleanup_removes_empty_subdirs(tmp_path: Path):
    sub = tmp_path / "dl" / "author_123"
    sub.mkdir(parents=True)
    f = sub / "v.mp4"
    f.write_bytes(b"x")
    _age(f, 30)
    cleanup_cache(tmp_path, days=7, apply=True)
    assert not sub.exists()                      # 删空后顺手收掉空目录


def test_transcribe_scratch_only_when_transcript_valid(tmp_path: Path):
    """转写音轨只有在成果有效时才算可删——无效时它是重转的依据，动不得。"""
    data = tmp_path / "data"
    rd = data / "aweme_1"
    rd.mkdir(parents=True)
    (rd / "audio.wav").write_bytes(b"x" * 100)
    seg = rd / "segments"
    seg.mkdir()
    (seg / "seg_000.wav").write_bytes(b"y" * 100)

    assert transcribe_scratch(rd) == []          # 转写成果还没出来 → 不许动音轨

    (rd / "transcript.txt").write_text(
        "足够长的有效转写正文，用来通过闸门的四十个字下限要求，包含具体步骤与数字。" * 2,
        encoding="utf-8")
    assert len(transcribe_scratch(rd)) == 2      # 音轨 + 分段各一个

    cleanup_transcribe_scratch(data, apply=True)
    assert not (rd / "audio.wav").exists()
    assert not seg.exists()                      # 空目录一并收掉
    assert (rd / "transcript.txt").exists()      # 成果不动
