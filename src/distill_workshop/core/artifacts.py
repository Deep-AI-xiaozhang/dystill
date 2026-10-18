"""产物有效性闸门（移植自早期实现，2026-09-06/09-08 事故的通用化收敛）。

纪律：任何"命中就复用"的分支（转写/帧/VLM/笔记），都必须先过有效性校验——
只判 exists() 不判内容，坏产物会被断点续跑永久复用（v226/v227 空壳事故根因）。
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

MIN_TRANSCRIPT_CHARS = 40
_TS_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def strip_timestamps(text: str) -> str:
    return _TS_RE.sub("", text or "")


def transcript_valid(text: str) -> bool:
    """转写可用：去时间戳后正文 ≥ 40 字。0 字节或仅「字幕君」类垃圾必须判废。"""
    return len(strip_timestamps(text).strip()) >= MIN_TRANSCRIPT_CHARS


def _vlm_valid(text: str) -> bool:
    # 纯音频源没有画面，合法标记直接放行（同类约定）
    return (text or "").strip().startswith("（纯音频源") or len((text or "").strip()) >= 40


def _note_valid(text: str) -> bool:
    """笔记可用：≥100 字且不是 LLM 失败时写的草稿。

    降级产物（notes.py:65-69）会写「> LLM 汇总失败，以下为原始素材」标记，
    它能过 100 字闸但不该被断点续跑当成有效笔记（会导致重跑永远拿不到正式笔记）。
    """
    s = (text or "").strip()
    if len(s) < 100:
        return False
    return not ("LLM 汇总失败" in s or "以下为原始素材" in s)


VALIDATORS = {
    "transcript": transcript_valid,
    "vlm": _vlm_valid,
    "note": _note_valid,
}


def read_if_valid(path: Path, kind: str) -> str | None:
    """读产物，过闸才返回内容；无效/缺失返回 None（调用方走重做）。"""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    return text if VALIDATORS[kind](text) else None


def write_if_valid(path: Path, text: str, kind: str) -> bool:
    """过闸才落盘。无效结果绝不写盘——失败/过短的结果不能污染断点续跑。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if VALIDATORS[kind](text):
        p.write_text(text, encoding="utf-8")
        return True
    return False


def is_stale_dir(d: Path) -> bool:
    """失败残留：目录里只有 meta.json 没有任何实质产物 → 提示删除而非补跑（防越补越多）。"""
    d = Path(d)
    if not d.is_dir():
        return False
    return not any(
        f.name != "meta.json" and f.stat().st_size > 0 and not f.is_dir()
        for f in d.iterdir()
    )


def record_dir(data_dir: Path, key: str) -> Path:
    """素材目录以身份键命名：data/aweme_<id> 或 data/fp_<hash12>。
    编号体系在开源版不参与任何逻辑——目录名即身份，漂移事故结构性消失。"""
    if key.startswith("aweme:"):
        name = "aweme_" + key.split(":", 1)[1]
    elif key.startswith("fp:"):
        name = "fp_" + key.split(":", 1)[1][:12]
    else:
        raise ValueError(f"非法身份键: {key}")
    d = Path(data_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_stale(data_dir: Path, *, apply: bool = False) -> list[Path]:
    """扫描失败残留目录；apply=False 只报告（默认），True 才删除。"""
    stale = [d for d in Path(data_dir).glob("*") if d.is_dir() and is_stale_dir(d)]
    if apply:
        for d in stale:
            shutil.rmtree(d, ignore_errors=True)
    return stale


# 下载缓存里算"媒体"的扩展名：视频 + 图文帖的图片。
# 白名单而非黑名单——cache/ 下还住着 config/cookies.json（凭据），只按名单删才碰不到它。
CACHE_MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".avi", ".ts",
                    ".jpg", ".jpeg", ".png", ".webp"}


def cache_media_files(cache_dir: Path, *, days: int = 7) -> list[Path]:
    """下载缓存里**超过 days 天**的媒体文件（按修改时间）。

    只扫 `<cache>/dl/`——凭据在 `<cache>/config/cookies.json`，其余文件（台账等）也不动。
    days<=0 视为"不清理"，返回空列表。
    """
    dl = Path(cache_dir) / "dl"
    if days <= 0 or not dl.is_dir():
        return []
    cutoff = time.time() - days * 86400
    return sorted(p for p in dl.rglob("*")
                  if p.is_file() and p.suffix.lower() in CACHE_MEDIA_EXTS
                  and p.stat().st_mtime < cutoff)


def cleanup_cache(cache_dir: Path, *, days: int = 7, apply: bool = False) -> list[Path]:
    """清理过期的下载缓存；apply=False 只报告（默认）。删完顺手收掉空目录。

    这些文件是可再生的（重跑同一条会重新下载），删掉只影响"重跑时不用再下"这一点便利。
    """
    stale = cache_media_files(cache_dir, days=days)
    if apply and stale:
        for p in stale:
            p.unlink(missing_ok=True)
        dl = Path(cache_dir) / "dl"
        for d in sorted((x for x in dl.rglob("*") if x.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    return stale


# 转写中间物：提取出的音轨与长音频切片。转写只认 transcript.txt 做缓存，
# 这些音轨转写完就再无用途，却和 cache/dl 一样"只进不出"（实测 19 条记录攒了 471 MB，
# 而真正的成果 note/transcript/vlm/check 加起来才 6 MB）。
_TRANSCRIBE_FILES = ("audio.wav",)
_TRANSCRIBE_DIRS = ("segments",)


def transcribe_scratch(record_dir: Path) -> list[Path]:
    """一条记录目录里可以清掉的转写中间物。

    **只在转写成果有效时才算可删**：transcript.txt 无效时音轨是重转的依据，动不得。
    """
    rd = Path(record_dir)
    if read_if_valid(rd / "transcript.txt", "transcript") is None:
        return []
    out = [rd / name for name in _TRANSCRIBE_FILES if (rd / name).is_file()]
    for name in _TRANSCRIBE_DIRS:
        d = rd / name
        if d.is_dir():
            out += [f for f in d.rglob("*") if f.is_file()]
    return out


def cleanup_transcribe_scratch(data_dir: Path, *, apply: bool = False) -> list[Path]:
    """清理各记录目录里的转写中间物。apply=False 只报告（默认）。删完收掉空的 segments/。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    stale: list[Path] = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            stale += transcribe_scratch(d)
    if apply and stale:
        for p in stale:
            p.unlink(missing_ok=True)
        for d in sorted((x for x in data_dir.rglob("segments") if x.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    return stale


# 中间产物：转写文本、画面理解文本、核对报告、身份信息。
# 与「转写中间物（音轨）」不同——这些是核对引擎的证据来源，所以不立即删，
# 而是给一个保留期（沿用 pipeline.cache_keep_days，默认 7 天），过期才清。
#
# **最终只留一样**：note.md（成品笔记）。这个工具的产品定义就是「一份笔记」。
# 逐项说明为什么都可以清：
#   - check.md / check.json：核对结论已写进笔记顶部徽章（✅ 已抽核 N/M · ⚠️ K 条存疑）
#   - transcript.txt / vlm.txt / frames/：笔记的依据，可再生（重跑会重新下载 + 重新转写）
#   - meta.json：身份信息。**断点续跑并不依赖它**（pipeline 只看 note.md 是否有效，
#     身份则由目录名 data/aweme_<id> 承载）——笔记一旦生成，它就没用了。
#     实测确认：process_record 的续跑判断是 read_if_valid(out/"note.md")，
#     meta.json 只在「直接调用 note_fn（测试路径）」时作标题回退用。
RAW_NAMES = ("transcript.txt", "vlm.txt", "check.md", "check.json", "meta.json")
RAW_DIRS = ("frames",)


def raw_material(record_dir: Path, *, days: int = 7) -> list[Path]:
    """一条记录目录里**超过 days 天**的原始素材（转写 / 画面理解 / 切帧）。

    days<=0 = 不清理。只按修改时间判断，不看内容——保留期是用户的策略，不是代码的判断。
    """
    if days <= 0:
        return []
    rd = Path(record_dir)
    if not rd.is_dir():
        return []
    cutoff = time.time() - days * 86400
    out: list[Path] = []
    for name in RAW_NAMES:
        p = rd / name
        if p.is_file() and p.stat().st_mtime < cutoff:
            out.append(p)
    for name in RAW_DIRS:
        d = rd / name
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        if files and max(f.stat().st_mtime for f in files) < cutoff:
            out.append(d)
    return out


def cleanup_raw_material(data_dir: Path, *, days: int = 7, apply: bool = False) -> list[Path]:
    """清理各记录目录里过期的中间产物。apply=False 只报告（默认）。

    **只保留 note.md**：这个工具的产品定义是「把视频蒸馏成一份笔记」，其余都是中间产物。
    过期后每个记录目录里就只剩 note.md（+ 空目录）。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir() or days <= 0:
        return []
    stale: list[Path] = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            stale += raw_material(d, days=days)
    if apply and stale:
        for p in stale:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    return stale
