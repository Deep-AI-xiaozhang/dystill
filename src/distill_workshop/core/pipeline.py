"""三模态流水线骨架（F1#4）：下载 → 转写 → 切帧 → 画面理解 →（笔记，F1#5）。

设计约束：
- 身份驱动：产物目录 = data/aweme_<id> | data/fp_<hash12>，无编号参与
- 断点续跑：每步产物先过有效性闸门（artifacts.py），命中即复用
- 纯音频分支：伪装 .mp4 用 ffprobe 识别，跳过切帧与 VLM（实测教训）
- 图文帖：GalleryDetected 在 process_record 内被捕获，转图片流程（跳过转写与切帧）
- 进度回调：启动即报总账、每步完成即报（"进度汇报义务"纪律的代码化）
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from ..config import AppConfig
from ..identity import content_fingerprint, record_key
from . import ledger
from .artifacts import read_if_valid, record_dir, write_if_valid
from .download import GalleryDetected, aweme_id_from_url, step_download
from .media import detect_source, is_audio_file
from .stt import BaseSTT, step_transcribe
from .vision import step_scenes

AUDIO_MARKER = "（纯音频源，无画面）"
# 断点续跑命中已有笔记是成功结果，不应被 CLI 当成失败。
SUCCESS_STATUSES = frozenset({"ok", "skipped"})


def _log(on_step: Callable, msg: str) -> None:
    if on_step:
        on_step(msg)


def resolve_key(url_or_path: str) -> tuple[str, str]:
    """返回 (身份键, 来源类型)。本地文件用内容指纹；抖音链接必须解析出 aweme_id。"""
    p = Path(url_or_path)
    if p.is_file():
        return record_key(fingerprint=content_fingerprint(p)), "local"
    src = detect_source(url_or_path)
    if src == "douyin":
        aweme = aweme_id_from_url(url_or_path)
        if not aweme:
            raise ValueError(f"无法从链接解析 aweme_id（短链展开失败或非视频页）: {url_or_path}")
        return record_key(aweme_id=aweme), src
    if src == "bilibili":
        # B站无 aweme_id，用URL哈希做临时键，下载后再用文件指纹
        import hashlib
        fp_hash = hashlib.sha256(url_or_path.encode()).hexdigest()
        return record_key(fingerprint=fp_hash), src
    raise ValueError(f"不支持的平台: {url_or_path}（source={src}）")


def _default_vlm(frames: list[Path], app: AppConfig, on_step: Callable) -> str:
    from ..providers.vision import PROMPT_MULTI, PROMPT_SINGLE, describe_images
    # 只用标记了能读图的供应商：退回"全部供应商"等于把图片发给纯文本模型，
    # 要么白等一轮降级、要么拿到一段不知所云的回答——宁可当场报错让人改配置。
    # （vision 默认就是 true，所以老配置不受影响；只有显式标了 false 才会走到这里）
    providers = app.vision_providers
    if not providers:
        raise RuntimeError("画面理解需要能读图的模型供应商：运行 dystill doctor 按提示补配置"
                           "（纯文本模型请在 providers 里标 vision: false）")
    batch = int(app.pipeline.get("vlm_batch_frames", 1) or 1)
    workers = int(app.pipeline.get("workers", 2) or 1)
    retry_pause = float(app.pipeline.get("vlm_retry_pause", 30) or 0)
    _log(on_step, f"④ 画面理解（VLM：{len(frames)} 帧 / 每请求 {batch} 帧 / {workers} 路并发）...")
    r = describe_images(frames, PROMPT_SINGLE if batch <= 1 else PROMPT_MULTI, providers,
                        batch_size=batch, workers=workers, on_step=on_step,
                        retry_pause=retry_pause)
    return r.content


def _gallery_vlm(images: list[Path], app: AppConfig, on_step: Callable) -> str:
    """图文帖的图片理解：复用逐帧 VLM 管线，只把提示词换成图文帖专用那份。"""
    from ..providers.vision import PROMPT_GALLERY, describe_images
    providers = app.vision_providers
    if not providers:
        raise RuntimeError("图文帖需要能读图的模型供应商：运行 dystill doctor 按提示补配置"
                           "（纯文本模型请在 providers 里标 vision: false）")
    batch = int(app.pipeline.get("vlm_batch_frames", 1) or 1)
    workers = int(app.pipeline.get("workers", 2) or 1)
    retry_pause = float(app.pipeline.get("vlm_retry_pause", 30) or 0)
    _log(on_step, f"④ 图片理解（VLM：{len(images)} 张 / 每请求 {batch} 张 / {workers} 路并发）...")
    r = describe_images(images, PROMPT_GALLERY, providers,
                        batch_size=batch, workers=workers, on_step=on_step,
                        retry_pause=retry_pause)
    return r.content


def _process_gallery(url_or_path: str, app: AppConfig, out: Path, artifacts: dict,
                     images: list[Path], *, vlm_fn=None, note_fn=None,
                     on_step: Callable | None = print) -> dict:
    """图文帖流程：图片理解 → 笔记。跳过下载产物之外的转写与切帧。

    与视频流程共用同一套产物闸门与笔记层，所以断点续跑、核对、台账都不用另写一套。
    """
    on_step = on_step or (lambda m: None)
    t0 = time.time()
    images = [Path(p) for p in images]
    if not images:
        raise RuntimeError("识别为图文帖，但没拿到任何图片（下载器可能没开图片下载）")
    _log(on_step, f"🖼️ 图文帖：{len(images)} 张图，跳过转写与切帧")
    artifacts.update(media_type="gallery", images=[str(p) for p in images])

    vlm_text = (vlm_fn or _gallery_vlm)(images, app, on_step)
    if not write_if_valid(out / "vlm.txt", vlm_text, "vlm"):
        raise RuntimeError("图片理解结果无效，拒绝生成成功记录")
    artifacts["vlm"] = str(out / "vlm.txt")

    if note_fn:
        _log(on_step, "⑤ 生成笔记...")
        note_fn(out, artifacts, app, on_step)
        if read_if_valid(out / "note.md", "note") is None:
            raise RuntimeError("笔记生成失败或结果无效，拒绝生成成功记录")
        artifacts["note"] = str(out / "note.md")

    meta = {"key": artifacts["key"], "source": artifacts["source"], "url": url_or_path,
            "media": str(images[0]), "media_type": "gallery",
            "images": [str(p) for p in images],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    seconds = round(time.time() - t0, 1)
    ledger.append_event(app, {"key": artifacts["key"], "event": "distill", "ok": True,
                              "seconds": seconds, "media_type": "gallery",
                              "steps": {"transcript": False, "frames": False, "note": bool(note_fn)}})
    _log(on_step, f"✅ 完成（{seconds}s）：{out.name}")
    artifacts.update(status="ok", seconds=seconds)
    return artifacts


def process_record(url_or_path: str, app: AppConfig, *,
                   stt: BaseSTT | None = None,
                   scenes_fn=None, vlm_fn=None, note_fn=None,
                   on_step: Callable | None = print) -> dict:
    """蒸馏单条记录（视频/图文/纯音频），返回产物路径字典。

    可注入依赖（测试/替换后端用）：scenes_fn(video, frames_dir, max_frames)、
    vlm_fn(frames, app, on_step)、note_fn(record_dir, artifacts, app, on_step)、stt。
    """
    on_step = on_step or (lambda m: None)
    t0 = time.time()
    key, source = resolve_key(url_or_path)
    out = record_dir(app.data_dir, key)
    artifacts = {"record_dir": str(out), "key": key, "source": source}
    _log(on_step, f"身份键: {key}（{source}）→ {out.name}")

    # 断点续跑：笔记已存在且有效 → 整条跳过
    if note_fn and read_if_valid(out / "note.md", "note"):
        _log(on_step, "⏭️ 笔记已存在且有效，跳过（断点续跑）")
        ledger.append_event(app, {"key": key, "event": "skip", "ok": True, "seconds": 0})
        artifacts.update(status="skipped")
        return artifacts

    # ① 媒体获取（图文帖由下载层识别 → 转图片流程，不转写、不切帧）
    p = Path(url_or_path)
    if p.is_file():
        media = p
        _log(on_step, f"① 本地文件，跳过下载（{p.name}）")
    else:
        _log(on_step, "① 下载中（douyin-downloader / yt-dlp）...")
        try:
            media = step_download(url_or_path, app, on_step=on_step)
        except GalleryDetected as e:
            return _process_gallery(url_or_path, app, out, artifacts, e.images,
                                    vlm_fn=vlm_fn, note_fn=note_fn, on_step=on_step)
    artifacts["media"] = str(media)

    # ② 转写（缓存过闸才复用；无效结果不落盘）
    _log(on_step, "② 转写...")
    if stt is None:
        raise RuntimeError("未配置 STT 后端：安装 faster-whisper（pip install 'dystill[local-stt]'）")
    step_transcribe(media, stt, out)
    if read_if_valid(out / "transcript.txt", "transcript") is None:
        raise RuntimeError("转写结果无效，拒绝继续生成成功记录")
    artifacts["transcript"] = str(out / "transcript.txt")

    # ③/④ 画面分支：纯音频跳过切帧与 VLM（后缀不可信，ffprobe 已在 is_audio_file 复核）
    if is_audio_file(media):
        _log(on_step, "🎧 纯音频源：跳过切帧和画面理解")
        write_if_valid(out / "vlm.txt", AUDIO_MARKER, "vlm")
        frames: list[Path] = []
    else:
        _log(on_step, "③ 切帧...")
        max_frames = int(app.pipeline.get("vlm_max_frames", 48))
        frames = (scenes_fn or step_scenes)(media, out / "frames", max_frames)
        _log(on_step, f"   切出 {len(frames)} 帧")
        vlm_text = (vlm_fn or _default_vlm)(frames, app, on_step)
        if not write_if_valid(out / "vlm.txt", vlm_text, "vlm"):
            raise RuntimeError("画面理解结果无效，拒绝生成成功记录")
        artifacts["frames"] = [str(f) for f in frames]
    artifacts["vlm"] = str(out / "vlm.txt")

    # ⑤ 笔记（F1#5 接入默认实现）
    if note_fn:
        _log(on_step, "⑤ 生成笔记...")
        note_fn(out, artifacts, app, on_step)
        if read_if_valid(out / "note.md", "note") is None:
            raise RuntimeError("笔记生成失败或结果无效，拒绝生成成功记录")
        artifacts["note"] = str(out / "note.md")

    # meta.json + 台账
    meta = {"key": key, "source": source, "url": url_or_path if not p.is_file() else "",
            "media": artifacts["media"], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    seconds = round(time.time() - t0, 1)
    ledger.append_event(app, {"key": key, "event": "distill", "ok": True, "seconds": seconds,
                              "steps": {"transcript": True, "frames": bool(frames), "note": bool(note_fn)}})
    _log(on_step, f"✅ 完成（{seconds}s）：{out.name}")
    artifacts.update(status="ok", seconds=seconds)
    return artifacts


def process_batch(urls: list[str], app: AppConfig, **kwargs) -> list[dict]:
    """批量蒸馏：串行执行；单条失败不中断整批，末尾汇总（批量纪律）。"""
    on_step = kwargs.get("on_step") or (lambda m: None)
    results = []
    total = len(urls)
    for i, u in enumerate(urls, 1):
        try:
            results.append(process_record(u, app, **kwargs))
        except GalleryDetected as e:
            # 正常路径下图文帖已在 process_record 内消化；走到这里说明异常来自更外层
            _log(on_step, f"⚠️ [{i}/{total}] 图文帖未能处理：{e}")
            results.append({"url": u, "status": "gallery", "error": str(e)})
        except Exception as e:  # noqa: BLE001
            _log(on_step, f"❌ [{i}/{total}] 失败: {type(e).__name__}: {e}")
            results.append({"url": u, "status": "failed", "error": f"{type(e).__name__}: {e}"})
    success = sum(1 for r in results if r.get("status") in SUCCESS_STATUSES)
    _log(on_step, f"—— 批量完成：{success}/{total} 成功（含断点跳过） ——")
    return results
