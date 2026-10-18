#!/usr/bin/env python
"""笔记生成提示词 A/B 定点回放：量「禁补视频外背景常识」这一句的效果（D23 执行项 1）。

为什么要它：D23 把 FP 主因归到「笔记补视频外常识」，治理动作是笔记提示词加了一句禁令
（`notes.py` build_prompt）。但那句是 2026-10-04 才进代码的，而三轮验收跑在 10-03 23:06–23:21
⇒ **治理效果从未被测量**。本脚本补这一次测量。

为什么不能用存档笔记当基线：D16/D20 已证，拿隔天存档做对照会把模型漂移记成改动效果
（实测同配置重跑 acc -4.3pp、339 条里 68 条翻转）。所以**同日跑两组**，唯一差异是那一句：
    A = 旧提示词（把禁令句从 build_prompt 输出里摘掉，等价于 991cf5b 之前）
    B = 新提示词（现码原样）

判据（两层，都不花人标）：
  1. **年份注入计数（通用、可自动）**：笔记里出现的 3~4 位数字年份，逐个回**原素材全部模态**
     （转写 + 画面）找——都找不到的才算「视频里查无出处」。A/B 对比看它有没有降。
     ⚠️ 只查转写会误报：画面板书/表格里的年份（实测 916/1038/1115/1271）会被错判成注入。
  2. **靶向命中表**：把已确认的 FP 源声明（三轮验收里 engine=ok 而人标非 ok 的 4 条）
     拆成特化片段，看 A/B 笔记里还有没有。片段选得特化，避免误报。

用法：
    python scripts/replay_note_prompt.py                       # 默认跑验收那 3 个视频
    python scripts/replay_note_prompt.py data/aweme_xxx ...     # 指定记录目录
结果落 `bench/notes_ab/`（该目录被 .gitignore 覆盖，不入库）。

⛔ D26 硬约束（2026-10-04，用户明示）：本脚本的输出**不得**用来支持「FP 会降 / 效果已验证」
这类结论。它的默认 3 个视频就是定义那批 FP 的同一批视频，4 个探针也全部取自那 4 条已知项，
且它根本不聚合 FP。真要验证改动，必须先用**未参与过任何提示词迭代的新视频**重新蒸馏并做
新鲜人工标注（留出集），验证集与定义过 FP 的视频不得重叠。详见 docs/DECISIONS.md D26。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import distill_workshop.core.notes as notes_mod
from distill_workshop.config import load_config
from distill_workshop.core.artifacts import read_if_valid
from distill_workshop.core.notes import clean_title, safe_title

# 验收口径那 3 个视频（三轮 204 条 = 120 + 56 + 28，与 4 条 FP 的来源视频完全一致）
DEFAULT_KEYS = ["aweme_7687942146737802981", "aweme_7689851246873759030",
                "aweme_7690808441719311625"]

# 禁令句：从「**也不要补充视频未提及的背景知识**」到本句句末。句中前段全是顿号/冒号，
# 第一个「。」就是句末，所以 [^。]*。 正好整段命中。
BAN_RE = re.compile(r"\*\*也不要补充视频未提及的背景知识\*\*[^。]*。")

# 靶向探针：三轮验收里 engine=ok 而人标非 ok 的那几条源声明。标签按 2026-10-04 逐条定性的
# 归类写——**不要再用「外部常识注入」这类整类归因**（实测那个归因不成立）：
#   · 960 时间轴：引擎有逐字证据（转写 00:19 + 画面板书），人标偏严待复核
#   · 高频考点：视频外评价，当前引擎已自己拦成 doubt
#   · 不服输：观点/推断升格为断言
#   · 快餐客群：把专名（沙县小吃）概括掉 + 「主要客群为学生」属推断
FP_PROBES = [
    ("aweme_7689851246873759030", "宋朝960年时间轴（引擎有逐字证据，人标待复核）", ["960"]),
    ("aweme_7689851246873759030", "应试评价（视频外评价，现引擎已拦 doubt）", ["高频考点", "填空点", "应试"]),
    ("aweme_7687942146737802981", "观点/推断升格为断言", ["不服输"]),
    ("aweme_7690808441719311625", "概括丢专名 + 推断升格", ["连锁快餐", "客群"]),
]

# 年份类数字（含 916/1038 这类 3~4 位历史年份）。带边界，避免命中长数字的一部分。
NUM_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")


def strip_ban(prompt: str) -> str:
    """摘掉禁令句 = 还原 991cf5b 之前的旧版提示词。改版没匹配上就报错停，
    绝不让对照退化成「两组同提示词」这种假实验。"""
    out, n = BAN_RE.subn("", prompt)
    if n != 1:
        raise RuntimeError(f"禁令句匹配 {n} 次（应为 1）：提示词可能已改版，对照组失真")
    return out


def title_of(rd: Path) -> str:
    try:
        meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
        media = meta.get("media") or ""
    except Exception:  # noqa: BLE001
        media = ""
    return safe_title(clean_title(Path(media).stem)) if media else rd.name


def injected_years(note: str, source: str) -> list[str]:
    """笔记里出现、但**原素材全部模态**里都找不到的 3~4 位数字。

    必须传「转写 + 画面」合并文本：只查转写会把画面表格里的年份（916/1038/1115/1271）
    误报成「视频外注入」。这是本项目踩过的坑（skill 实战经验速查 九-8）。
    """
    return sorted({n for n in NUM_RE.findall(note) if n not in source})


def main() -> int:
    ap = argparse.ArgumentParser(description="笔记生成提示词 A/B 定点回放（D23 执行项 1）")
    ap.add_argument("record_dirs", nargs="*", help="记录目录（默认验收那 3 个视频）")
    ap.add_argument("--out-dir", default="bench/notes_ab")
    args = ap.parse_args()

    dirs = [Path(p) for p in args.record_dirs] or [
        Path("data") / k for k in DEFAULT_KEYS]
    app = load_config()
    if not app.ready_providers:
        print("[!] 没有就绪的模型供应商，先跑 dystill doctor")
        return 2
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    orig_build = notes_mod.build_prompt
    groups = {"A_old": True, "B_new": False}
    notes: dict[str, dict[str, str]] = {}
    meta: dict[str, dict] = {}
    for label, is_old in groups.items():
        if is_old:
            notes_mod.build_prompt = lambda *a, **kw: strip_ban(orig_build(*a, **kw))
        else:
            notes_mod.build_prompt = orig_build
        try:
            for rd in dirs:
                transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
                vlm = read_if_valid(rd / "vlm.txt", "vlm") or ""
                if not transcript:
                    print(f"[!] {rd} 的转写无效/缺失，跳过（闸门拦住了，别硬跑）")
                    continue
                t0 = time.time()
                print(f"[{label}] {rd.name} 生成中……", flush=True)
                text = notes_mod.generate_note(transcript, vlm, title_of(rd), app,
                                               on_step=lambda s: print(f"    {s}", flush=True))
                dt = time.time() - t0
                (outdir / f"{rd.name}.{label}.md").write_text(text, encoding="utf-8")
                notes.setdefault(rd.name, {})[label] = text
                meta.setdefault(rd.name, {})[label] = {"chars": len(text), "sec": round(dt, 1)}
                print(f"[{label}] {rd.name} 完成：{len(text)} 字，{dt:.0f}s", flush=True)
        finally:
            notes_mod.build_prompt = orig_build

    # ── 判据 1：年份注入计数 ──────────────────────────────────────────────
    print("\n## 判据 1 · 年份注入（笔记里的 3~4 位数字，原转写里查不到）\n")
    print("| 记录 | A 旧提示词 | B 新提示词 | 变化 |")
    print("|---|---|---|---|")
    for rd in dirs:
        if rd.name not in notes:
            continue
        transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
        # 年份判据的比对源必须是全部模态：画面表格里的年份只查转写会误报成注入
        source = transcript + "\n" + (read_if_valid(rd / "vlm.txt", "vlm") or "")
        a = injected_years(notes[rd.name].get("A_old", ""), source)
        b = injected_years(notes[rd.name].get("B_new", ""), source)
        meta[rd.name]["injected_A"] = a
        meta[rd.name]["injected_B"] = b
        delta = f"{len(a)} → {len(b)}"
        print(f"| {rd.name} | {len(a)} 个：{'、'.join(a) if a else '—'} | "
              f"{len(b)} 个：{'、'.join(b) if b else '—'} | {delta} |")

    # ── 判据 2：靶向命中表 ──────────────────────────────────────────────
    print("\n## 判据 2 · FP 源声明是否还在（特化片段命中）\n")
    print("| FP 源 | A 旧提示词 | B 新提示词 |")
    print("|---|---|---|")
    hits = {}
    for key, label, frags in FP_PROBES:
        row = [label]
        for g in ("A_old", "B_new"):
            text = notes.get(key, {}).get(g, "")
            got = [f for f in frags if f in text] if text else []
            hits.setdefault(label, {})[g] = got
            row.append("命中 " + "、".join(got) if got else ("—" if text else "无笔记"))
        print("| " + " | ".join(row) + " |")

    (outdir / "summary.json").write_text(json.dumps(
        {"meta": meta, "probe_hits": hits, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
         "providers": [getattr(p, "name", str(p)) for p in app.ready_providers][:3]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] 笔记与摘要已写 {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())