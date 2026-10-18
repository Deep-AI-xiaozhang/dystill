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
