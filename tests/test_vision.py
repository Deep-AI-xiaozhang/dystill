import json
from pathlib import Path

from PIL import Image

from distill_workshop.core.vision import (
    ahash,
    hamming,
    keep_note_frames,
    parse_scene_csv,
    resample_indices,
    select_kept,
    step_scenes,
)


def test_select_kept_aba_pattern():
    # A-B-A：镜头切走再切回，第三帧与窗口内第一帧相似 → 只喂两次
    h1, h2 = 0b11110000, 0b00001111
    assert select_kept([h1, h2, h1]) == [0, 1]


def test_select_kept_all_distinct():
    # 间距足够的哈希（两两 hamming ≥8）全部保留
    assert select_kept([0, 0xFF, 0xFFFF, 0xFFFFFF]) == [0, 1, 2, 3]


def test_resample_indices_keeps_ends():
    assert resample_indices(5, 5) == [0, 1, 2, 3, 4]
    idx = resample_indices(122, 48)
    assert len(idx) == 48 and idx[0] == 0 and idx[-1] == 121 and idx == sorted(set(idx))
    assert resample_indices(10, 1) == [0]


def test_cached_frames_respect_reduced_max_frames(tmp_path: Path):
    """切帧缓存命中也要服从当前上限；改小配置不能继续送全部旧帧。"""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(1, 11):
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(frames_dir / f"{i:03d}.jpg")

    picked = step_scenes(tmp_path / "not-used.mp4", frames_dir, max_frames=3)

    assert [p.name for p in picked] == ["001.jpg", "005.jpg", "010.jpg"]
    assert len(list(frames_dir.glob("*.jpg"))) == 10   # 只选子集，不破坏缓存


def test_parse_scene_csv(tmp_path: Path):
    csv = tmp_path / "scenes.csv"
    lines = ["Scene Number,Start Frame,Start Timecode,Start Time (seconds),End Frame,End",
             ",,,,",
             "1,0,00:00:00.000,0.0,100,4.0",
             "2,100,00:00:04.000,4.0,220,9.0"]
    csv.write_text("\n".join(lines), encoding="utf-8")
    assert parse_scene_csv(csv) == {1: 0.0, 2: 4.0}


def test_ahash_hamming(tmp_path: Path):
    # 左白右黑 vs 左黑右白：均值相同、位图完全相反 → hamming 64
    a = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            a.putpixel((x, y), (255, 255, 255) if x < 32 else (0, 0, 0))
    b = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            b.putpixel((x, y), (0, 0, 0) if x < 32 else (255, 255, 255))
    ha, hb = ahash(a), ahash(b)
    assert hamming(ha, ha) == 0
    assert hamming(ha, hb) == 64


def test_keep_note_frames_remaps_timecodes(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    n = 10
    for i in range(1, n + 1):
        Image.new("RGB", (32, 32), (i * 20, 0, 0)).save(frames_dir / f"{i:03d}.jpg")
    (frames_dir / "timecodes.json").write_text(
        json.dumps({f"{i:03d}.jpg": float(i) for i in range(1, n + 1)}), encoding="utf-8")
    out = tmp_path / "note_media"
    picked = keep_note_frames(sorted(frames_dir.glob("*.jpg")), out, max_n=3)
    assert len(picked) == 3 and all(p.exists() for p in picked)
    tc = json.loads((out / "timecodes.json").read_text(encoding="utf-8"))
    assert set(tc) == {p.name for p in picked}          # 只含被选中帧
    first = picked[0].name[:3]                           # "001.jpg" → "001"
    assert tc[picked[0].name] == float(int(first))       # 时间码与源一致


def test_step_scenes_narrows_timecodes_on_cache_resample(tmp_path: Path):
    """缓存帧数超过上限时，时间轴必须同步收窄到真正留下的那几帧。

    回归（2026-11-04）：原来只返回子集、timecodes.json 还是全量，下游按「第 N 张 →
    排序后第 N 个键」映射，把 001/004/007/010 四帧报成 10/20/30/40 秒（真实 10/40/70/100）。
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(1, 11):
        (frames_dir / f"{i:03d}.jpg").write_bytes(b"x")
    (frames_dir / "timecodes.json").write_text(
        json.dumps({f"{i:03d}.jpg": float(i * 10) for i in range(1, 11)}), encoding="utf-8")

    kept = step_scenes(tmp_path / "dummy.mp4", frames_dir, max_frames=4)

    assert [p.name for p in kept] == ["001.jpg", "004.jpg", "007.jpg", "010.jpg"]
    tc = json.loads((frames_dir / "timecodes.json").read_text(encoding="utf-8"))
    assert set(tc) == {p.name for p in kept}            # 时间轴只剩留下的帧
    assert tc["004.jpg"] == 40.0                        # 且值仍是这一帧自己的秒数
