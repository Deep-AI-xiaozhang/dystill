"""dystill 命令行入口：init / doctor（流水线子命令随 M1 逐步接入）。"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path

from . import __version__
from .config import CONFIG_NAME
from .doctor import print_report, run_doctor


def cmd_init(args) -> int:
    target = Path(args.path or Path.cwd()) / CONFIG_NAME
    if target.exists() and not args.force:
        print(f"已存在 {target}（--force 覆盖）")
        return 1
    try:
        example_text = files('distill_workshop').joinpath('config.example.yaml').read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError, OSError, UnicodeError):
        print("找不到 config.example.yaml 模板")
        return 1
    target.write_text(example_text, encoding='utf-8')
    print(f"已生成 {target}\n下一步：填 providers（或设置环境变量 DISTILL_API_BASE / DISTILL_API_KEY / DISTILL_MODEL），"
          f"然后运行 dystill doctor")
    return 0


def cmd_doctor(args) -> int:
    return print_report(run_doctor(do_probe_calls=args.probe), as_json=args.json)


def cmd_clean(args) -> int:
    from .config import load_config
    from .core.artifacts import (
        cache_media_files,
        cleanup_cache,
        cleanup_raw_material,
        cleanup_stale,
        cleanup_transcribe_scratch,
        raw_material,
        transcribe_scratch,
    )
    app = load_config()
    days = args.cache_days

    stale = cleanup_stale(app.data_dir, apply=args.apply)
    # 先列举算体积，再执行删除——删完文件就没了，体积就算不出来（第一版踩过）
    data_dir = Path(app.data_dir)
    scratch = ([p for d in sorted(data_dir.iterdir()) if d.is_dir()
                for p in transcribe_scratch(d)] if data_dir.is_dir() else [])
    scratch_mb = sum(p.stat().st_size for p in scratch if p.exists()) / 1024 / 1024
    cleanup_transcribe_scratch(app.data_dir, apply=args.apply)
    raw = ([p for d in sorted(data_dir.iterdir()) if d.is_dir()
            for p in raw_material(d, days=days)] if data_dir.is_dir() else [])
    raw_mb = sum((sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                  if p.is_dir() else p.stat().st_size) for p in raw if p.exists()) / 1024 / 1024
    cleanup_raw_material(app.data_dir, days=days, apply=args.apply)
    doomed = cache_media_files(app.cache_dir, days=days)
    freed = sum(p.stat().st_size for p in doomed if p.exists())
    cleanup_cache(app.cache_dir, days=days, apply=args.apply)

    if not stale and not scratch and not raw and not doomed:
        print(f"没有需要清理的东西 ✅（下载缓存保留最近 {days} 天）")
        return 0
    verb = "已删除" if args.apply else "发现"
    if stale:
        print(f"{verb} {len(stale)} 个失败残留目录：")
        for d in stale:
            print(f"  - {d}")
    if scratch:
        print(f"{verb} {len(scratch)} 个转写中间物（音轨/分段，约 {scratch_mb:.0f} MB）"
              f"——转写成果有效，这些音轨已无用途")
    if raw:
        print(f"{verb} {len(raw)} 项过期中间产物（转写/画面/切帧/核对报告，超过 {days} 天，"
              f"约 {raw_mb:.0f} MB）——成品笔记保留不动")
    if doomed:
        print(f"{verb} {len(doomed)} 个过期下载缓存（超过 {days} 天，约 {freed / 1024 / 1024:.0f} MB）：")
        for p in doomed[:10]:
            print(f"  - {p.name}")
        if len(doomed) > 10:
            print(f"  ... 另有 {len(doomed) - 10} 个")
    if not args.apply:
        print("\n以上只是报告，加 --apply 才真删。")
        print("可再生的：下载缓存、转写音轨、中间产物（重跑会重新下载 + 重新转写）。")
        print("永久保留：只有 note.md（成品笔记）——本工具的产物就是一份笔记。")
    return 0


def cmd_run(args) -> int:
    from .config import load_config
    from .core.notes import default_note_fn
    from .core.pipeline import SUCCESS_STATUSES, process_batch
    app = load_config()
    # 下载缓存保留策略：超过 N 天的媒体文件自动清掉（0=关闭，默认 7 天）。
    # 缓存是可再生的，留着只为"重跑时省一次下载"，不该无限涨。
    try:
        keep_days = int(app.pipeline.get("cache_keep_days", 7) or 0)
    except (TypeError, ValueError):
        keep_days = 7
    if keep_days > 0:
        from .core.artifacts import (
            cache_media_files,
            cleanup_cache,
            cleanup_raw_material,
            raw_material,
        )
        doomed = cache_media_files(app.cache_dir, days=keep_days)
        freed = sum(p.stat().st_size for p in doomed if p.exists())
        cleanup_cache(app.cache_dir, days=keep_days, apply=True)
        # 原始素材（转写/画面/切帧）同样按保留期自动清，不必等用户手动跑 clean
        data_dir = Path(app.data_dir)
        raw = ([p for d in sorted(data_dir.iterdir()) if d.is_dir()
                for p in raw_material(d, days=keep_days)] if data_dir.is_dir() else [])
        raw_mb = sum((sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                      if p.is_dir() else p.stat().st_size) for p in raw if p.exists())
        cleanup_raw_material(app.data_dir, days=keep_days, apply=True)
        if not getattr(args, "json", False):
            if doomed:
                print(f"🧹 清理过期下载缓存 {len(doomed)} 个（超过 {keep_days} 天），"
                      f"释放 {freed / 1024 / 1024:.0f} MB")
            if raw:
                print(f"🧹 清理过期原始素材 {len(raw)} 项（超过 {keep_days} 天），"
                      f"释放 {raw_mb / 1024 / 1024:.0f} MB（只保留 note.md）")
    stt = None
    if app.stt.backend == "faster-whisper":
        try:
            from .core.stt import FasterWhisperSTT
            stt = FasterWhisperSTT(app.stt.model, device=app.stt.device,
                                   compute_type=app.stt.compute_type)
        except Exception as e:  # noqa: BLE001 —— 未装 faster-whisper 时给出明确指引
            print(f"[!] 本地转写后端不可用（{e}）；运行 dystill doctor 查看怎么补")
            return 2
    # on_step 必须接上：process_batch 是靠它把单条失败原因报出来的。
    # 不接的话 `_log` 走默认的 lambda 把消息丢掉，用户只会看到 exit=1 而没有任何提示（实测踩到）
    results = process_batch(args.urls, app, stt=stt, note_fn=default_note_fn(app),
                            on_step=None if getattr(args, "json", False) else print)
    failed = [r for r in results if r.get("status") not in SUCCESS_STATUSES]
    if getattr(args, "json", False):
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif failed:
        print(f"\n[!] {len(failed)}/{len(results)} 条未成功：")
        for r in failed:
            print(f"    - {r.get('url', '')[:70]}  [{r.get('status')}] {r.get('error', '')[:160]}")
    check_failed = 0
    if args.check:
        from .core.check import resolve_record_dir, run_check
        for r in results:
            if r.get("status") not in SUCCESS_STATUSES:
                continue
            try:
                run_check(resolve_record_dir(r.get("url") or r["record_dir"], app),
                          app, mode=args.check,
                          on_step=None if getattr(args, "json", False) else print)
            except Exception as e:  # noqa: BLE001 —— 核对失败不该把已成功的蒸馏判失败
                check_failed += 1
                print(f"[!] 核对失败（{type(e).__name__}: {e}）")
    # 蒸馏全成但核对有失败也要返回非零：agent 编排靠退出码判断这批产物是否完整
    return 1 if (any(r.get("status") not in SUCCESS_STATUSES for r in results) or check_failed) else 0


def cmd_check(args) -> int:
    import json as _json

    from .config import load_config
    from .core.check import resolve_record_dir, run_check
    app = load_config()
    out, failed = [], 0
    for t in args.targets:
        try:
            rd = resolve_record_dir(t, app)
            r = run_check(rd, app, mode=args.mode, note_path=args.note,
                          rewrite=args.rewrite, batch_delay=args.delay,
                          resume=args.resume, only_failed=args.retry_failed,
                          on_step=None if args.json else print)
            out.append(r)
            if not args.json:
                print()
        except Exception as e:  # noqa: BLE001 —— 单个目标失败不打断整批（批量纪律）
            failed += 1
            if not args.json:
                print(f"❌ {t}: {type(e).__name__}: {e}")
            out.append({"target": t, "status": "failed", "error": f"{type(e).__name__}: {e}"})
    if args.json:
        print(_json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def cmd_kb(args) -> int:
    """知识库：汇总所有笔记成索引 + 完整性自检。只归拢，不做价值判断。"""
    import datetime
    import json as _json

    from .config import load_config
    from .core.kb import (
        INDEX_NAME,
        build_index,
        check_kb,
        find_notes,
        upsert_frontmatter,
    )

    app = load_config()
    metas = find_notes(app.data_dir, app.notes_dir)
    if args.out:
        from pathlib import Path as _P
        index_path = _P(args.out)
    else:
        index_path = app.notes_dir / INDEX_NAME

    if not metas:
        if args.json:
            print(_json.dumps({"notes": 0, "index": None, "issues": []}, ensure_ascii=False))
        else:
            print("没找到任何笔记。先跑 dystill run <链接> 蒸馏几条。")
        return 0

    today = datetime.date.today().isoformat()
    touched = 0
    if args.frontmatter:
        for m in metas:
            try:
                touched += 1 if upsert_frontmatter(m.path, m) else 0
            except OSError as e:
                print(f"[!] 写 frontmatter 失败：{m.path}（{e}）")

    if args.check:
        issues = check_kb(metas, app.data_dir, index_path)
        errs = [i for i in issues if i.level == "ERROR"]
        warns = [i for i in issues if i.level == "WARN"]
        if args.json:
            print(_json.dumps({
                "notes": len(metas), "index": str(index_path),
                "errors": [i.__dict__ for i in errs], "warnings": [i.__dict__ for i in warns],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"知识库自检：{len(metas)} 篇笔记 ｜ ERROR {len(errs)} ｜ WARN {len(warns)}")
            for i in errs:
                print(f"  ❌ [{i.kind}] {i.detail}")
            for i in warns:
                print(f"  ⚠️ [{i.kind}] {i.detail}")
            if not issues:
                print("  全部正常。")
        return 1 if errs else 0

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(build_index(metas, index_path, today), encoding="utf-8")
    # 自检放在写索引之后：否则首次生成必然报「索引未生成」（自己刚写的东西）
    issues = check_kb(metas, app.data_dir, index_path)

    if args.json:
        print(_json.dumps({
            "notes": len(metas), "index": str(index_path),
            "frontmatter_updated": touched,
            "warnings": [i.__dict__ for i in issues],
        }, ensure_ascii=False, indent=2))
        return 0

    print(f"知识库已生成：{index_path}")
    print(f"  收录 {len(metas)} 篇" + (f"，更新 frontmatter {touched} 篇" if args.frontmatter else ""))
    if issues:
        print(f"  自检发现 {len(issues)} 条待办（跑 dystill kb --check 看详情）：")
        for i in issues[:5]:
            print(f"    {'❌' if i.level == 'ERROR' else '⚠️'} [{i.kind}] {i.detail}")
        if len(issues) > 5:
            print(f"    …另 {len(issues) - 5} 条")
    return 0


def _force_utf8_stdio() -> None:
    """Windows 默认控制台是 GBK：报告里的 ✅/❌/⚠️ 会直接抛 UnicodeEncodeError。

    必须在 parse_args 之前做 —— argparse 的 --help/usage 也会打印这些字符。
    只动编码与错误策略，不改缓冲行为；stdout 被重定向走管道时 reconfigure 仍安全。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            # line_buffering：输出被重定向到管道/文件时（跑长任务、被 agent 调用），
            # 默认块缓冲会把进度攒在内存里，半小时看不到一行（实测踩到）。
            # 每行 flush 后进度实时可见；输出量不大，性能代价可忽略。
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):  # 已被替换成非 TextIOWrapper 的流
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(prog="dystill", description="蒸馏工坊：只做抖音、带笔记级事实核对的本地蒸馏工具")
    parser.add_argument("--version", action="version", version=f"dystill {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="生成 config.yaml（从模板）")
    p_init.add_argument("--path", help="目标目录（默认当前目录）")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_doc = sub.add_parser("doctor", help="体检：缺什么、怎么补（agent 可读 --json）")
    p_doc.add_argument("--json", action="store_true", help="结构化输出（供 agent 解析 NEED_SETUP）")
    p_doc.add_argument("--probe", action="store_true", help="真实调用一次供应商验证连通（消耗少量 token）")
    p_doc.set_defaults(func=cmd_doctor)

    p_clean = sub.add_parser("clean", help="清理失败残留目录（只有 meta.json、没产物的空壳）")
    p_clean.add_argument("--apply", action="store_true",
                         help="真的删除（默认只报告）。别在蒸馏/核对运行时执行")
    p_clean.add_argument("--cache-days", type=int, default=7,
                         help="下载缓存保留天数，超过的算过期（默认 7；0=不清理缓存）")
    p_clean.set_defaults(func=cmd_clean)

    p_run = sub.add_parser("run", help="蒸馏：抖音链接 / 本地媒体文件 → 笔记（可批量）")
    p_run.add_argument("urls", nargs="+")
    p_run.add_argument("--check", nargs="?", const="standard", default=None,
                       metavar="MODE", help="蒸馏完顺手核对（fast/standard/deep，默认 standard）")
    p_run.add_argument("--json", action="store_true", help="结构化输出（供 agent 解析）")
    p_run.set_defaults(func=cmd_run)

    p_chk = sub.add_parser("check", help="核对：笔记 vs 原视频（✅一致 / ❌不一致 / ⚠️存疑）")
    p_chk.add_argument("targets", nargs="+",
                       help="记录目录 / 笔记文件 / 抖音链接 / 目录名关键词（可批量）")
    p_chk.add_argument("--mode", default="standard",
                       help="核对模式：fast（只核数字专名引用）/ standard（默认）/ deep（双信号）")
    p_chk.add_argument("--note", help="指定要核对的笔记文件（默认 note.md）")
    p_chk.add_argument("--rewrite", action="store_true",
                       help="把 ❌ 不一致的句子按证据改写回笔记（默认只报告不改写）")
    p_chk.add_argument("--delay", type=float, default=None,
                       help="判定批次之间的间隔秒数（默认读 config.yaml 的 pipeline.check_batch_delay；"
                            "免费/低配额通道遇到 429 时调大）")
    p_chk.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                       help="断点续跑：复用上一轮判定，只重判「调用失败（限流/网络）」的条目（最省 token）")
    p_chk.add_argument("--resume", action="store_true",
                       help="断点续跑：复用上一轮全部判定，完全不重新调用模型（只重建报告/徽章）")
    p_chk.add_argument("--json", action="store_true", help="结构化输出（供 agent 解析）")
    p_chk.set_defaults(func=cmd_check)

    p_kb = sub.add_parser("kb", help="知识库：汇总所有笔记成索引 + 完整性自检")
    p_kb.add_argument("--check", action="store_true",
                      help="只做自检，不生成索引（缺身份键/重复/空壳/有素材无笔记/索引过期）")
    p_kb.add_argument("--frontmatter", action="store_true",
                      help="同时给每篇笔记补 type/topic/tags 三个 YAML 字段（幂等，供 Obsidian 检索）")
    p_kb.add_argument("-o", "--out", help="索引输出路径（默认 <notes_dir>/知识库.md）")
    p_kb.add_argument("--json", action="store_true", help="结构化输出（供 agent 解析）")
    p_kb.set_defaults(func=cmd_kb)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
