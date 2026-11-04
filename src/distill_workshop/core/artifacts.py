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
QUALITY_WARNING_MARKER = "初稿质量自检未通过"
_TS_RE = re.compile(
    r"\[\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[-–—~至]\s*\d{1,2}:\d{2}(?::\d{2})?)?\]"
)


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


def read_note_if_complete(path: Path) -> str | None:
    """读可复用笔记；自检失败稿允许留盘供人工看，但不能让断点续跑当成成功。"""
    text = read_if_valid(path, "note")
    if text is None or QUALITY_WARNING_MARKER in text:
        return None
    return text


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


# 大小两类产物分开处理：
# - 大文件（frames/、cache/dl/ 媒体、转写音轨）可清：重跑可再生，且占空间最多。
# - 小证据（转写/画面/核对报告/身份）默认长期保留：它们是「可回查」的实际依据，
#   全部加起来通常只有几十 KB 到几 MB，不该为了省一点空间把产品核心价值删掉。
#
# 只有用户显式执行 `dystill clean --purge-evidence` 时，才允许按保留期清掉小证据。
# 一旦清掉，旧笔记仍能读，但无法再基于原始证据重新核对。
DURABLE_EVIDENCE_NAMES = (
    "transcript.txt",
    "vlm.txt",
    "check.md",
    "check.json",
    "check.html",
    "meta.json",
)
RAW_DIRS = ("frames",)


def raw_material(record_dir: Path, *, days: int = 7) -> list[Path]:
    """一条记录目录里**超过 days 天**的大体积产物（目前只有切帧目录）。

    转写、画面理解、核对报告属于可回查证据，不在这里自动清理。
    days<=0 = 不清理。只按修改时间判断，不看内容——保留期是用户的策略，不是代码的判断。
    """
    if days <= 0:
        return []
    rd = Path(record_dir)
    if not rd.is_dir():
        return []
    cutoff = time.time() - days * 86400
    out: list[Path] = []
    for name in RAW_DIRS:
        d = rd / name
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        if files and max(f.stat().st_mtime for f in files) < cutoff:
            out.append(d)
    return out


def cleanup_raw_material(data_dir: Path, *, days: int = 7, apply: bool = False) -> list[Path]:
    """清理各记录目录里过期的大体积产物。apply=False 只报告（默认）。

    默认永远保留可回查的小证据；需要彻底腾空间时，由调用方显式调用
    cleanup_evidence()，不能混在常规清理里静默执行。
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


def old_evidence(record_dir: Path, *, days: int = 7) -> list[Path]:
    """一条记录里**超过 days 天**的可回查证据。

    这是危险清单，只供用户显式执行“彻底清理证据”时使用。常规 run/clean 不会调用。
    """
    if days <= 0:
        return []
    rd = Path(record_dir)
    if not rd.is_dir():
        return []
    cutoff = time.time() - days * 86400
    return sorted(
        p for name in DURABLE_EVIDENCE_NAMES
        if (p := rd / name).is_file() and p.stat().st_mtime < cutoff
    )


def cleanup_evidence(data_dir: Path, *, days: int = 7, apply: bool = False) -> list[Path]:
    """显式清理过期证据；apply=False 只报告。

    删除后笔记仍保留，但该记录无法再跑 dystill check，也无法查看当时的转写、
    画面理解与逐条核对明细。调用方必须在执行前向用户说明这一后果。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir() or days <= 0:
        return []
    stale: list[Path] = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            stale += old_evidence(d, days=days)
    if apply:
        for p in stale:
            p.unlink(missing_ok=True)
    return stale


_ORPHAN_MATERIAL_NAMES = ("transcript.txt", "vlm.txt", "check.json", "check.md", "check.html")


def orphan_material_dirs(data_dir: Path, *, days: int = 7) -> list[Path]:
    """有素材、没成品笔记、且已经过期的记录目录。

    这是“清完笔记后留下的素材孤儿”，只在用户显式执行
    `dystill clean --purge-orphans --apply` 时删除；普通清理不动它，
    避免把尚未生成笔记的任务误删。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir() or days <= 0:
        return []
    cutoff = time.time() - days * 86400
    out: list[Path] = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or (d / "note.md").is_file() or is_stale_dir(d):
            continue
        evidence = [d / name for name in _ORPHAN_MATERIAL_NAMES if (d / name).is_file()]
        if not evidence:
            continue
        files = [p for p in d.rglob("*") if p.is_file()]
        if not files or max(p.stat().st_mtime for p in files) >= cutoff:
            continue
        out.append(d)
    return out


def cleanup_orphan_materials(data_dir: Path, *, days: int = 7, apply: bool = False) -> list[Path]:
    """删除过期且没有 note.md 的素材目录；默认只报告，不删除。"""
    stale = orphan_material_dirs(data_dir, days=days)
    if apply:
        for d in stale:
            shutil.rmtree(d, ignore_errors=True)
    return stale
