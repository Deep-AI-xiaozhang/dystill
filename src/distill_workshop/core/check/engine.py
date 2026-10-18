"""核对引擎编排（L1 → L2 → L3 → L4 一条链）。

入口只有一个 `run_check`，输入是**已经蒸馏过的一条记录目录**（data/aweme_<id> 或 data/fp_<hash>），
输出是落盘的 check.json / check.md + 笔记顶部徽章。这样设计的理由：核对不重新下载、不重新转写，
只吃已有产物——零额外媒体成本，且天然可复跑（改判定标准后重跑同一批笔记即可回归对比）。

失败策略（与流水线同一条纪律）：
- 笔记无效 → 直接报错，不产半成品报告；
- 模型不可用 → 全部声明记 ⚠️ 存疑并写明原因，报告照出（**绝不空手**，也绝不假装核过）。
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ...config import AppConfig, ProviderConfig
from ...identity import extract_aweme_id
from .. import ledger
from ..artifacts import read_if_valid
from .claim_extractor import extract_claims
from .evidence_retriever import SourceIndex, retrieve_all
from .models import VERDICT_DOUBT, CheckResult
from .modes import CheckMode, get_mode
from .report import apply_fixes, badge, inject_badge, write_reports
from .verifier import apply_gates, verify_claims

NOTE_CANDIDATES = ("note.md",)
# 核对自己产出的文件：绝不能被当成"笔记"再核一遍
RESERVED_MD = {"check.md"}


def _log(on_step: Callable | None, msg: str) -> None:
    if on_step:
        on_step(msg)


def _apply_mode_overrides(m: CheckMode, app: AppConfig) -> CheckMode:
    """允许 config.yaml 覆盖模式参数（目前只有 max_claims：长笔记想要核全就得调大）。

    用 dataclasses.replace 派生副本而不是改全局 MODES——模式表是模块级共享状态，
    改它会让"配了一次上限"污染同一进程里后续所有核对。
    """
    try:
        cap = int(app.pipeline.get("check_max_claims", 0) or 0)
    except (TypeError, ValueError):
        cap = 0
    return replace(m, max_claims=cap) if cap > 0 else m


def resolve_record_dir(target: str, app: AppConfig) -> Path:
    """把用户给的目标解析成记录目录：目录 / note 文件 / 抖音链接 / 身份键 / 关键词。"""
    t = (target or "").strip().strip('"')
    p = Path(t)
    if p.is_dir():
        return p
    if p.is_file():
        return p.parent
    aweme = extract_aweme_id(t)
    if aweme:
        d = Path(app.data_dir) / f"aweme_{aweme}"
        if d.is_dir():
            return d
        raise FileNotFoundError(f"链接对应的记录还没蒸馏过：{d}")
    # 关键词：按身份键前缀或目录名片段找唯一匹配（唯一才认，多个就报错让用户说清楚）
    hits = [d for d in Path(app.data_dir).glob("*") if d.is_dir() and t and t in d.name]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"找不到记录：{target}（data_dir={app.data_dir}）")
    raise ValueError(f"匹配到多条记录，请给出更精确的目标：{[d.name for d in hits]}")


def find_note(record_dir: Path) -> Path | None:
    """找要核对的笔记：note.md 优先 → 唯一一篇 *.md → note 开头的 *.md。

    排除自己产出的 check.md（2026-09-17 实测踩到）：核对过一次后目录里就多了 check.md，
    "目录里唯一一篇 md"这个判断随之失效 → 第二次核对直接报"没有笔记可核对"——
    等于**工具跑一次就把自己弄坏了**。更糟的可能：若把 check.md 当笔记，它能过笔记长度闸门，
    于是把核对报告本身拿去核对。
    """
    rd = Path(record_dir)
    for name in NOTE_CANDIDATES:
        p = rd / name
        if p.is_file():
            return p
    mds = [p for p in sorted(rd.glob("*.md")) if p.name not in RESERVED_MD]
    if len(mds) == 1:
        return mds[0]
    notes = [p for p in mds if p.stem.lower().startswith("note")]
    return notes[0] if len(notes) == 1 else None


def load_previous(record_dir: Path) -> dict[str, dict]:
    """读上一轮的 check.json，建索引供断点续跑复用。

    **按笔记原句（original）与判定句（text）双键索引**：original 由笔记原文决定、与模型无关，
    是最稳的匹配键（复用判定时 L1 还会重跑消歧，若只按 text 匹配，模型换一种说法就匹配不上，
    续跑会退化成全量重判）。旧报告没有 original 字段 → 退回 text 键。
    """
    p = Path(record_dir) / "check.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 旧报告坏了不该挡住重新核对
        return {}
    out: dict[str, dict] = {}
    for c in data.get("claims") or []:
        if not c.get("verdict"):
            continue
        for k in (str(c.get("original") or "").strip(), str(c.get("text") or "").strip()):
            if k:
                out.setdefault(k, c)
    return out


def prev_failed(prev: dict) -> bool:
    """上一轮这条是不是"没判成"（限流/网络），值得重判。

    兼容旧报告：早期版本没有 failed 字段，只能按理由文字认。
    """
    if prev.get("failed"):
        return True
    return "调用失败" in str(prev.get("reason") or "")


def analyze(note_text: str, transcript: str, vlm: str, *, mode: str = "standard",
            chat_fn=None, app: AppConfig | None = None, frames_dir: Path | None = None,
            key: str = "", on_step: Callable | None = None,
            batch_delay: float | None = None,
            previous: dict[str, dict] | None = None,
            resume: bool = False, only_failed: bool = False) -> CheckResult:
    """四层跑完拿结果，**不碰磁盘**（基准集/回归对比用，避免为了算个准确率就改用户笔记）。

    previous/resume/only_failed = 核对层的"断点续跑"（与流水线同一条纪律）：
    - resume：上一轮已判定的条目直接复用，不重复花 token（改判定标准后想全量重判就别开）
    - only_failed：只重判上一轮"调用失败"的条目——低配额通道被限流打断后最省的恢复方式
    """
    m = get_mode(mode)
    if app is not None:
        m = _apply_mode_overrides(m, app)
    result = CheckResult(key=key, mode=m.name, created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not transcript and not vlm:
        raise ValueError("没有转写/画面素材，无法核对")

    claims, skipped = extract_claims(note_text, m, chat_fn=_l1_chat_fn(app, chat_fn),
                                     on_step=on_step)
    result.claims, result.skipped = claims, skipped
    if not claims:
        _log(on_step, "没有可核对的原子声明（笔记可能全是概括/观点）")
        return result

    retrieve_all(claims, SourceIndex.build(transcript, vlm, frames_dir), on_step=on_step)

    # 断点续跑：先决定哪些条目还需要问模型
    pending = claims
    if previous and (resume or only_failed):
        pending, reused = [], 0
        for c in claims:
            prev = previous.get(c.original.strip()) or previous.get(c.text.strip())
            if prev and not (only_failed and prev_failed(prev)):
                c.verdict = prev.get("verdict") or c.verdict
                c.reason, c.fix = prev.get("reason", ""), prev.get("fix", "")
                c.failed = bool(prev.get("failed"))
                reused += 1
            else:
                pending.append(c)
        _log(on_step, f"♻️ 复用上一轮判定 {reused} 条，需重判 {len(pending)} 条")
        if not pending:
            _log(on_step, "没有需要重判的条目（要全量重判请去掉 --resume / --retry-failed）")

    provider_hint = ""
    if not pending:
        chat_fn = None
    elif chat_fn is None and app is not None:
        chat_fn, provider_hint = _default_chat(app)
    if not pending:
        pass
    elif chat_fn is None:
        for c in pending:
            c.verdict = VERDICT_DOUBT
            c.reason = "未配置模型供应商，无法判定（运行 dystill doctor 补配置）"
            c.fix = "建议回看原片确认"
        _log(on_step, "⚠️ 未配置模型供应商 → 全部声明记存疑（报告仍照出）")
    else:
        if batch_delay is None:
            try:
                batch_delay = float(app.pipeline.get("check_batch_delay", 0.0) if app else 0.0)
            except (TypeError, ValueError):
                batch_delay = 0.0
        _log(on_step, f"L3 逐条判定（{len(pending)} 条 / 每批 {m.batch_size} 条"
                      f"{f' / 批间隔 {batch_delay:g}s' if batch_delay else ''}）...")
        result.provider = verify_claims(pending, m, chat_fn, on_step,
                                        batch_delay=batch_delay) or provider_hint

    # 复用来的判定也要重新过闸：判定标准升级（如新增"数字查不到"这道闸）时，
    # 旧的 ✅ 必须被同一套规则复查，否则报告会混着新旧两套标准
    for c in claims:
        apply_gates(c, m)
    return result


def run_check(record_dir: Path, app: AppConfig, *, mode: str = "standard",
              note_path: str | Path | None = None, chat_fn=None,
              rewrite: bool = False, on_step: Callable | None = print,
              batch_delay: float | None = None,
              resume: bool = False, only_failed: bool = False) -> dict:
    """核对一条已蒸馏记录并落盘：check.json / check.md + 笔记徽章。"""
    rd = Path(record_dir)
    note_file = Path(note_path) if note_path else find_note(rd)
    if not note_file or not note_file.is_file():
        raise FileNotFoundError(f"记录目录里没有笔记可核对：{rd}")

    note_text = read_if_valid(note_file, "note")
    if note_text is None:
        raise ValueError(f"笔记未过有效性闸门（可能过短/空壳），拒绝核对：{note_file}")

    transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
    vlm = read_if_valid(rd / "vlm.txt", "vlm") or ""
    key = rd.name
    try:
        key = json.loads((rd / "meta.json").read_text(encoding="utf-8")).get("key") or rd.name
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        pass

    _log(on_step, f"核对 [{get_mode(mode).label}]：{note_file.name}"
                  f"（素材 {len(transcript)} 字转写 / {len(vlm)} 字画面）")
    result = analyze(note_text, transcript, vlm, mode=mode, chat_fn=chat_fn, app=app,
                     frames_dir=rd / "frames", key=key, on_step=on_step,
                     batch_delay=batch_delay,
                     previous=load_previous(rd) if (resume or only_failed) else None,
                     resume=resume, only_failed=only_failed)

    # L4
    _log(on_step, f"L4 报告：{badge(result)}")
    paths = write_reports(rd, result)

    new_note = inject_badge(note_text, result)
    applied: list[str] = []
    if rewrite:
        new_note, applied = apply_fixes(new_note, result, on_step=on_step)
    if new_note != note_text:
        note_file.write_text(new_note, encoding="utf-8")

    still_failed = sum(1 for c in result.claims if c.failed)
    ledger.append_event(app, {"key": key, "event": "check", "ok": True, "mode": result.mode,
                              "claims": result.total, "badge": badge(result),
                              "failed": still_failed, "rewrite": bool(applied)})
    return {"key": key, "mode": result.mode, "badge": badge(result), "status": result.status,
            "counts": {"total": result.total, "ok": result.count("ok"),
                       "mismatch": result.count("mismatch"),
                       "doubt": result.count("doubt")},
            "failed": still_failed,
            "record_dir": str(rd), "note": str(note_file), "rewrites": applied, **paths}


def _chat_fn(providers: list[ProviderConfig], timeout: int = 120):
    """把 provider 链包成核对层用的调用入口（temperature=0：判定要稳不要灵活）。

    timeout 默认 120s；stealth 类慢思考模型用 pipeline.judge_timeout 放宽
    （实测 120s 下 ReadTimeout/空 content，2026-10-04 space-bunny-alpha）。"""
    from ...providers.llm import chat
    return lambda messages: chat(messages, providers, max_tokens=4000, temperature=0.0,
                                 timeout=timeout)


def _l1_chat_fn(app: AppConfig | None, given):
    """L1 消歧用的模型入口。

    默认返回 None = 只走规则（与 2026-10-04 之前的生产行为一致）。要开消歧必须显式设
    `pipeline.check_l1_disambiguate: true`：它会改写/拆分声明，属于改变基准集可比性的
    行为变更，按 D16/D20 纪律得先用同日对照组量过（详见 D24）。
    **开关是唯一的决定者**：关着的时候即使调用方传了 chat_fn 也不开——否则
    `run_check(chat_fn=...)` 这类只想换 L3 通道的调用方（测试/回放）会被动改掉 L1，
    基准数字就说不清是哪一版了。
    开了之后调用方显式给的 chat_fn 优先（方便对 L1 单独做定点验证），否则用**通用链**
    （与笔记生成同一条），不碰 judge_provider——那是 L3 判定专用通道。

    为什么会有这个函数：原实现把通道解析放在 extract_claims **之后**，而 CLI/bench_check
    都不传 chat_fn，导致生产路径里消歧从未执行过（docstring/ROADMAP 却记作已落地）。
    """
    if app is None:
        return None
    if str(app.pipeline.get("check_l1_disambiguate", "")).strip().lower() \
            not in ("1", "true", "yes", "on", "是"):
        return None
    if given is not None:
        return given
    providers = app.ready_providers
    return _chat_fn(providers) if providers else None


def _default_chat(app: AppConfig):
    """取默认判定通道：providers 降级链（与笔记生成共用一条链）。

    判定通道可与笔记通道分离（config.example 的混搭设计）：pipeline.judge_provider
    指定 provider name（**逗号分隔多个 = 判定专用降级链**，按序降级）时，核对判定
    只用这些家。实测依据（2026-09-30 D21）：同一份存档证据，agnes-2.5-flash 判定 70.1%、
    DeepSeek-V4-Flash 判定 83.8%——判定质量是主要瓶颈，值得为它单指定模型。
    通道稳定性同样重要（D21 追记）：硅基流动免费通道有日级"判定变保守"降级窗口，
    单通道判定撞上即整批报废——所以判定链应该至少两家不同供应商互为备份。
    """
    providers = app.ready_providers
    raw = str(app.pipeline.get("judge_provider") or "").strip()
    names = [s.strip() for s in raw.split(",") if s.strip()]
    # 名字校验要排在"有没有就绪 provider"**之前**：否则"名字写错 + 没配 provider"时报的是
    # "未配置模型供应商"，把人引去查 key（2026-10-04 查 D21 通道分离时发现的盲区）
    if not providers:
        if names:
            raise ValueError(f"pipeline.judge_provider 指定了 {names}，但没有就绪的 provider")
        return None, ""
    if names:
        by_name = {p.name: p for p in providers}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"pipeline.judge_provider 指定的 {missing} 不在就绪 provider 里"
                             f"（现有：{[p.name for p in providers]}）")
        # 同名去重后按指定顺序组成判定专用链（不同供应商互为备份，见上）
        seen: dict[str, ProviderConfig] = {}
        for n in names:
            seen.setdefault(n, by_name[n])
        providers = list(seen.values())
    timeout = int(app.pipeline.get("judge_timeout") or 120)
    return _chat_fn(providers, timeout), providers[0].name
