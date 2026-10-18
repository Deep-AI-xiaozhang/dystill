"""视觉层（移植自早期实现的切帧优化）。

实测优化全保留：
1. 滑动窗口 aHash 去重（win=4，hamming≤3）：A-B-A 镜头切走再切回只喂一次，实测 122 帧→7 帧省 15 倍
2. 上限重采样 VLM_MAX_FRAMES（默认 48）：均匀取样保首尾
3. timecodes.json 时间轴：帧名 → 场景起始秒（核对报告时间戳回查的数据源）
4. glob generator 教训（2026-09-01 长视频崩溃）：凡 len() 必先 sorted()，本实现从源头避免

纯函数（ahash/去重/重采样/CSV 解析）与子进程调用分离，核心算法可独立测试。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# ── 纯函数：感知哈希与去重 ─────────────────────────────────
def ahash(img) -> int:
    """8x8 灰度均值位图。"""
    small = img.convert("L").resize((8, 8))
    px = list(small.tobytes())   # L 模式 tobytes 即逐像素灰度（getdata 已被 Pillow 弃用）
    avg = sum(px) / len(px)
    h = 0
    for i, v in enumerate(px):
        if v >= avg:
            h |= 1 << i
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def select_kept(hashes: list[int], window: int = 4, threshold: int = 3) -> list[int]:
    """滑动窗口去重：与最近保留的 window 帧比较，全相似才丢。返回保留的原始下标。
    A-B-A 序列：切走再切回的画面只喂 VLM 一次。"""
    kept: list[int] = []
    recent: list[int] = []
    for i, h in enumerate(hashes):
        if any(hamming(prev, h) <= threshold for prev in recent[-window:]):
            continue
        recent.append(h)
        kept.append(i)
    return kept


def resample_indices(n: int, max_n: int) -> list[int]:
    """超出上限时均匀取样，保持首尾。n<=max_n 原样返回。"""
    if n <= max_n:
        return list(range(n))
    if max_n <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (max_n - 1)) for i in range(max_n)})


def parse_scene_csv(csv_path: Path) -> dict[int, float]:
    """list-scenes CSV：Scene Number → 起始秒。解析失败返回空（时间轴尽力而为）。"""
    out: dict[int, float] = {}
    try:
        for ln in Path(csv_path).read_text(encoding="utf-8", errors="replace").splitlines()[2:]:
            cells = [c.strip() for c in ln.split(",")]
            if len(cells) >= 5 and cells[0].isdigit():
                try:
                    out[int(cells[0])] = float(cells[3])
                except ValueError:
                    pass
    except (OSError, ValueError):
        pass
    return out


# ── 子进程：切镜头 + 抽帧 ───────────────────────────────────
def _scene_no(name: str) -> int:
    m = re.search(r"Scene-(\d+)", name)
    return int(m.group(1)) if m else 0


def _ffprobe_duration(video: Path) -> float:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=_NO_WINDOW, check=False)
        return float(p.stdout.strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def _fallback_single_frame(video: Path, frames_dir: Path) -> list[Path]:
    """scenedetect 切出 0 帧（纯口播/静态画面）：取时长中点 1 帧，时间轴标 0。"""
    out = frames_dir / "001.jpg"
    ss = max(1.0, _ffprobe_duration(video) / 2)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(ss), "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)],
        capture_output=True, timeout=300, creationflags=_NO_WINDOW, check=False)
    (frames_dir / "timecodes.json").write_text(
        json.dumps({out.name: 0.0}, ensure_ascii=False), encoding="utf-8")
    return [out] if out.exists() else []


def step_scenes(video: Path, frames_dir: Path, max_frames: int = 48) -> list[Path]:
    """切镜头 → 每场景 1 帧（720p）→ 滑窗去重 → 上限重采样 → timecodes.json。"""
    video, frames_dir = Path(video), Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(frames_dir.glob("*.jpg"))
    if cached:  # 断点续跑：帧缓存命中直接复用
        return cached

    raw = frames_dir / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "scenedetect", "-i", str(video), "-o", str(raw),
         "detect-content", "--threshold", "27",
         "save-images", "--num-images", "1", "list-scenes", "--filename", "scenes"],
        capture_output=True, timeout=1200, creationflags=_NO_WINDOW, check=False)
    raws = sorted(raw.glob("*.jpg"))
    if not raws:
        result = _fallback_single_frame(video, frames_dir)
        if raw.is_dir() and not any(raw.iterdir()):
            raw.rmdir()
        return result

    scene_time = parse_scene_csv(raw / "scenes.csv")

    # 缩放 + 去重
    from PIL import Image
    hashes: list[int] = []
    paths: list[Path] = []
    kept_raws: list[Path] = []      # 与 paths 严格并行：某帧打不开时一起跳过，
                                    # 否则 paths[i] 与 raws[i] 错位，timecodes 会把
                                    # 帧号映射到别的场景（画面证据时间戳全错，静默）
    for i, rp in enumerate(raws):
        dst = frames_dir / f"{i + 1:03d}.jpg"
        try:
            img = Image.open(rp).convert("RGB")
            if img.width > 720:
                img = img.resize((720, max(1, int(img.height * 720 / img.width))), Image.LANCZOS)
            img.save(dst, "JPEG", quality=85)
        except (OSError, ValueError):
            continue
        hashes.append(ahash(img))
        paths.append(dst)
        kept_raws.append(rp)

    kept = select_kept(hashes)
    for idx, p in enumerate(paths):
        if idx not in kept:
            p.unlink(missing_ok=True)
    saved = [paths[i] for i in kept]

    # 上限重采样（保持首尾）
    if len(saved) > max_frames:
        keep = {saved[i] for i in resample_indices(len(saved), max_frames)}
        for p in saved:
            if p not in keep:
                p.unlink(missing_ok=True)
        saved = sorted(keep)

    # 时间轴：帧名 → 场景起始秒
    tc = {p.name: scene_time.get(_scene_no(rp.name), 0.0)
          for p, rp in zip([paths[i] for i in kept], [kept_raws[i] for i in kept])
          if p.exists()} if scene_time else {}
    if tc:
        (frames_dir / "timecodes.json").write_text(
            json.dumps(tc, ensure_ascii=False), encoding="utf-8")

    import shutil
    shutil.rmtree(raw, ignore_errors=True)
    return saved


def keep_note_frames(frames: list[Path], out_dir: Path, max_n: int = 5) -> list[Path]:
    """均匀采样 ≤max_n 张复制到笔记目录；timecodes 同步重映射（v1.9 修复）。"""
    frames = sorted(frames)
    if not frames:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = resample_indices(len(frames), max_n)
    picked = [frames[i] for i in idxs]
    result = []
    for src in picked:
        dst = out_dir / src.name
        dst.write_bytes(src.read_bytes())
        result.append(dst)
    src_tc = src_timecodes(frames[0].parent)
    if src_tc:
        remap = {p.name: src_tc[p.name] for p in result if p.name in src_tc}
        if remap:
            (out_dir / "timecodes.json").write_text(
                json.dumps(remap, ensure_ascii=False), encoding="utf-8")
    return result


def src_timecodes(frames_dir: Path) -> dict:
    p = Path(frames_dir) / "timecodes.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
