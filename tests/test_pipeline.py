import json
from pathlib import Path

from distill_workshop.config import AppConfig
from distill_workshop.core.ledger import read_events
from distill_workshop.core.pipeline import (
    AUDIO_MARKER,
    SUCCESS_STATUSES,
    process_batch,
    process_record,
)
from distill_workshop.core.stt import BaseSTT

TRANSCRIPT = "这是一段足够长的转写正文，用来通过有效性闸门的四十个字下限要求，包含具体的方法步骤和关键数字。"
VLM_TEXT = "画面中出现代码编辑器，标题栏显示构建成功，屏幕字幕提到部署流程与回滚方案，结尾展示性能对比图表。"
NOTE_TEXT = "这是一篇足够长的测试笔记正文，用来通过笔记有效性闸门的一百字下限要求。" * 3


class FakeSTT(BaseSTT):
    name = "fake"

    def __init__(self):
        self.calls = 0

    def transcribe_raw(self, wav, vad_filter=True):
        self.calls += 1
        return [(0.0, TRANSCRIPT)]


class EmptySTT(BaseSTT):
    name = "empty"

    def transcribe_raw(self, wav, vad_filter=True):
        return []


def make_app(tmp_path: Path) -> AppConfig:
    return AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                     data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")


def test_video_pipeline_end_to_end(tmp_path: Path):
    app = make_app(tmp_path)
    media = tmp_path / "v.mp4"
    media.write_bytes(b"fake video bytes")
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"fake jpg")
    stt = FakeSTT()
    vlm_calls = []

    def note_fn(record_dir, artifacts, app, on_step):
        (Path(record_dir) / "note.md").write_text(NOTE_TEXT, encoding="utf-8")

    r = process_record(str(media), app, stt=stt,
                       scenes_fn=lambda m, d, n: [jpg],
                       vlm_fn=lambda frames, app, on_step: (vlm_calls.append(1), VLM_TEXT)[1],
                       note_fn=note_fn, on_step=lambda m: None)

    assert r["status"] == "ok"
    rd = Path(r["record_dir"])
    assert rd.name.startswith("fp_")            # 本地文件身份 = 内容指纹
    assert (rd / "transcript.txt").exists()
    assert (rd / "vlm.txt").exists()
    assert (rd / "note.md").exists()
    assert (rd / "meta.json").exists()
    assert len(read_events(app)) == 1            # 台账记录
    assert len(vlm_calls) == 1 and stt.calls == 1

    # 第二次运行：笔记已存在 → 整条跳过，VLM/STT 都不再调用（断点续跑）
    r2 = process_record(str(media), app, stt=stt,
                        scenes_fn=lambda m, d, n: [jpg],
                        vlm_fn=lambda frames, app, on_step: (vlm_calls.append(1), VLM_TEXT)[1],
                        note_fn=note_fn, on_step=lambda m: None)
    assert r2["status"] == "skipped"
    assert len(vlm_calls) == 1 and stt.calls == 1


def test_skipped_is_a_successful_batch_result(tmp_path: Path):
    app = make_app(tmp_path)
    media = tmp_path / "existing.mp3"
    media.write_bytes(b"audio")
    # 先跑一遍生成有效笔记，再验证第二遍的 skipped 状态属于成功。
    process_record(str(media), app, stt=FakeSTT(), note_fn=lambda d, a, ap, log:
                   (Path(d).mkdir(parents=True, exist_ok=True),
                    (Path(d) / "note.md").write_text(NOTE_TEXT, encoding="utf-8")),
                   on_step=lambda m: None)
    results = process_batch([str(media)], app, stt=FakeSTT(), note_fn=lambda *args: None,
                            on_step=lambda m: None)
    assert results[0]["status"] == "skipped"
    assert results[0]["status"] in SUCCESS_STATUSES


def test_transcript_cache_reuse_across_runs(tmp_path: Path):
    app = make_app(tmp_path)
    media = tmp_path / "v2.mp4"
    media.write_bytes(b"abc")
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"j")
    stt = FakeSTT()
    kw = {"stt": stt, "scenes_fn": lambda m, d, n: [jpg],
              "vlm_fn": lambda f, a, o: VLM_TEXT, "on_step": lambda m: None}
    process_record(str(media), app, **kw)
    process_record(str(media), app, **kw)        # 无 note_fn：转写缓存命中，不重转
    assert stt.calls == 1
    events = read_events(app)
    assert len(events) == 2 and all(e["ok"] for e in events)


def test_audio_branch_skips_frames_and_vlm(tmp_path: Path):
    app = make_app(tmp_path)
    media = tmp_path / "song.mp3"
    media.write_bytes(b"fake audio")

    def boom(*a, **k):
        raise AssertionError("纯音频分支不应切帧")

    r = process_record(str(media), app, stt=FakeSTT(), scenes_fn=boom,
                       vlm_fn=boom, on_step=lambda m: None)
    rd = Path(r["record_dir"])
    assert (rd / "vlm.txt").read_text(encoding="utf-8") == AUDIO_MARKER
    assert not any(rd.glob("frames"))


def test_unsupported_platform_rejected(tmp_path: Path):
    try:
        process_record("https://example.com/foo", make_app(tmp_path),
                       stt=FakeSTT(), on_step=lambda m: None)
        assert False, "应当拒绝不支持的平台"
    except ValueError:
        pass


def test_gallery_pipeline_skips_transcribe_and_frames(tmp_path: Path, monkeypatch):
    """图文帖：不转写、不切帧，只做图片理解 + 笔记（图片路径由下载层带出来）。"""
    from distill_workshop.core import pipeline
    from distill_workshop.core.download import GalleryDetected

    app = make_app(tmp_path)
    img1, img2 = tmp_path / "1.jpg", tmp_path / "2.jpg"
    img1.write_bytes(b"a")
    img2.write_bytes(b"b")

    def fake_download(url, app_, on_step=None):
        raise GalleryDetected("图文帖（窗口内新图片 2 张，无视频）", images=[img1, img2])

    monkeypatch.setattr(pipeline, "step_download", fake_download)

    stt = FakeSTT()
    vlm_calls = []

    def boom(*a, **k):
        raise AssertionError("图文帖不应转写或切帧")

    def note_fn(record_dir, artifacts, app_, on_step):
        (Path(record_dir) / "note.md").write_text(NOTE_TEXT, encoding="utf-8")

    r = pipeline.process_record(
        "https://www.douyin.com/video/7412345678901234567", app,
        stt=stt, scenes_fn=boom,
        vlm_fn=lambda imgs, a, o: (vlm_calls.append(imgs), VLM_TEXT)[1],
        note_fn=note_fn, on_step=lambda m: None)

    assert r["status"] == "ok" and r["media_type"] == "gallery"
    rd = Path(r["record_dir"])
    assert rd.name == "aweme_7412345678901234567"
    assert (rd / "vlm.txt").exists() and (rd / "note.md").exists()
    assert not (rd / "transcript.txt").exists()          # 图文帖没有转写
    assert stt.calls == 0                                 # STT 完全没碰
    assert len(vlm_calls) == 1 and len(vlm_calls[0]) == 2  # 两张图一起交给 VLM
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    assert meta["media_type"] == "gallery" and len(meta["images"]) == 2
    assert read_events(app)[0]["media_type"] == "gallery"


def test_gallery_without_images_reports_clearly(tmp_path: Path, monkeypatch):
    """识别成图文帖却一张图都没有：要报明确错误，而不是产出一份空笔记。"""
    from distill_workshop.core import pipeline
    from distill_workshop.core.download import GalleryDetected

    monkeypatch.setattr(pipeline, "step_download",
                        lambda url, app_, on_step=None: (_ for _ in ()).throw(
                            GalleryDetected("图文帖", images=[])))
    try:
        pipeline.process_record("https://www.douyin.com/video/7412345678901234567",
                                make_app(tmp_path), stt=FakeSTT(), on_step=lambda m: None)
        assert False, "没有图片时应当报错"
    except RuntimeError as e:
        assert "图片" in str(e)


def test_invalid_transcript_does_not_report_success(tmp_path: Path):
    media = tmp_path / "bad.mp3"
    media.write_bytes(b"fake")
    try:
        process_record(str(media), make_app(tmp_path), stt=EmptySTT(), on_step=lambda m: None)
        assert False, "无效转写不应被当成成功"
    except RuntimeError as e:
        assert "转写" in str(e)


def test_invalid_vlm_does_not_report_success(tmp_path: Path):
    media = tmp_path / "bad-vlm.mp4"
    media.write_bytes(b"fake")
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"fake")
    try:
        process_record(str(media), make_app(tmp_path), stt=FakeSTT(),
                       scenes_fn=lambda m, d, n: [jpg],
                       vlm_fn=lambda f, a, o: "",
                       on_step=lambda m: None)
        assert False, "无效画面理解不应被当成成功"
    except RuntimeError as e:
        assert "画面理解" in str(e)


def test_note_fn_that_writes_no_note_does_not_report_success(tmp_path: Path):
    media = tmp_path / "bad-note.mp4"
    media.write_bytes(b"fake")
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"fake")
    try:
        process_record(str(media), make_app(tmp_path), stt=FakeSTT(),
                       scenes_fn=lambda m, d, n: [jpg],
                       vlm_fn=lambda f, a, o: VLM_TEXT,
                       note_fn=lambda *args: None,
                       on_step=lambda m: None)
        assert False, "没有写出笔记不应被当成成功"
    except RuntimeError as e:
        assert "笔记" in str(e)
