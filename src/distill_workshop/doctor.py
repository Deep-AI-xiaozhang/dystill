"""体检与安装向导：agent 友好。

设计目标：用户装完第一件事跑 `dystill doctor`，缺什么直接给 NEED_SETUP 结构化输出，
agent 可解析并向用户要"一个 key"，而不是让用户读安装文档。
"""
from __future__ import annotations

import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

from .config import load_config


def _check(name: str, ok: bool, detail: str = "", fix: str = "", need_setup: bool = False) -> dict:
    return {"name": name, "status": "ok" if ok else ("NEED_SETUP" if need_setup else "missing"),
            "detail": detail, "fix": fix}


def run_doctor(do_probe_calls: bool = False) -> dict:
    checks: list[dict] = []

    # 1. 配置文件
    cfg = load_config()
    providers = cfg.ready_providers
    checks.append(_check(
        "providers", bool(providers),
        f"{len(providers)} 个可用（api_base+key+model 齐全）" if providers else "尚未配置模型供应商",
        fix="运行 `dystill init` 生成 config.yaml，再填一个带视觉能力的对话模型"
            "（key/api_base/model）；或设置环境变量 DISTILL_API_BASE / DISTILL_API_KEY / "
            "DISTILL_MODEL",
        need_setup=not providers,
    ))

    # 2. 带视觉的供应商（画面理解要靠它；只有文本模型时流水线会在第④步断）
    vis = cfg.vision_providers
    checks.append(_check(
        "provider(vision)", bool(vis),
        f"{len(vis)} 个可读图（{[p.name for p in vis]}）" if vis else
        "没有标记 vision: true 的可用供应商，画面理解会失败",
        fix="在 providers 里给带视觉的模型保留 vision: true（默认就是 true）；"
            "文本模型请显式写 vision: false，免得画面理解先打到它上面白等一轮",
        need_setup=bool(cfg.ready_providers) and not vis,
    ))

    # 2b. key 文件读不到（配了 api_key_file 但路径错/文件没了）——不算致命，但要说清为什么这家被跳过
    bad_keys = [p for p in cfg.providers if p._key_file_error]
    if bad_keys:
        checks.append(_check(
            "provider(keyfile)", False,
            "；".join(f"{p.name}: {p._key_file_error}" for p in bad_keys),
            fix="检查 api_key_file 路径是否正确、文件是否还在（各家免费通道的 *_key.txt）",
            need_setup=True,
        ))

    # 2c. config.yaml 含未识别的键——通常是拼写错误，静默丢弃容易让人白查半天
    if cfg.unknown_config_keys:
        checks.append(_check(
            "config(unknown_keys)", False,
            "发现未识别的配置键（可能是拼写错误，这些键目前被忽略）：" +
            "、".join(cfg.unknown_config_keys),
            fix="对照 config.example.yaml 检查拼写；pipeline.* 字段不在此范围",
            need_setup=False,
        ))

    # 3. STT 后端
    # 目前只有 faster-whisper 一种实现。配了别的名字必须报错而不是报绿——
    # 原实现写「云端，随 providers 走」+ status=ok，但 stt.py 里根本没有云端后端：
    # doctor 说可以开跑，真跑起来 pipeline 会抛「未配置 STT 后端」（2026-11-04 复查发现）。
    if cfg.stt.backend == "faster-whisper":
        has_stt = find_spec("faster_whisper") is not None
        checks.append(_check(
            "stt(faster-whisper)", has_stt,
            "已安装" if has_stt else "未安装（首次转录将自动下载 small 模型 ~500MB）",
            fix="pip install faster-whisper（源码安装也可用 pip install -e '.[local-stt]'）",
            need_setup=not has_stt,
        ))
    else:
        checks.append(_check(
            "stt", False,
            f"backend={cfg.stt.backend}：本工具目前只实现了 faster-whisper，没有云端转写后端",
            fix="把 config.yaml 的 stt.backend 改回 faster-whisper（本地转写，不需要 key）",
            need_setup=True,
        ))

    # 3. ffmpeg（下载/切帧/分段转写依赖）
    has_ffmpeg = shutil.which("ffmpeg") is not None
    checks.append(_check(
        "ffmpeg", has_ffmpeg,
        shutil.which("ffmpeg") or "未找到",
        fix="安装 ffmpeg 并加入 PATH（winget install Gyan.FFmpeg / apt install ffmpeg）",
        need_setup=not has_ffmpeg,
    ))

    # 4. 切帧依赖（scene 切镜头 + Pillow 缩放/感知哈希）——缺了不是"少个功能"，是"产物质量塌"
    has_pil = find_spec("PIL") is not None
    checks.append(_check(
        "pillow", has_pil,
        "已安装" if has_pil else "未安装（切出帧后缩放/去重会直接报错）",
        fix="pip install Pillow（通常重装本体也会补上）",
        need_setup=not has_pil,
    ))
    has_scene = find_spec("scenedetect") is not None
    checks.append(_check(
        "scenedetect", has_scene,
        "已安装（按镜头切帧）" if has_scene else "未安装（会退回整片抽 1 帧，画面理解质量大幅下降）",
        fix="pip install scenedetect（通常重装本体也会补上）",
        need_setup=not has_scene,
    ))

    # 4b. 抖音下载器自己的依赖（它不是本仓库的依赖，只 clone 不等于是可用）
    if cfg.download.downloader_dir:
        dd = Path(cfg.download.downloader_dir).expanduser()
        mods = {
            "aiohttp": "aiohttp", "aiofiles": "aiofiles", "aiosqlite": "aiosqlite",
            "rich": "rich", "yaml": "pyyaml", "dateutil": "python-dateutil",
            "gmssl": "gmssl", "imageio_ffmpeg": "imageio-ffmpeg",
        }
        missing = [pkg for mod, pkg in mods.items() if find_spec(mod) is None]
        req = dd / "requirements.txt"
        checks.append(_check(
            "douyin-downloader", not missing and dd.is_dir(),
            ("依赖齐全" if not missing else f"缺依赖：{', '.join(missing)}")
            if dd.is_dir() else f"配置目录不存在：{dd}",
            fix=(f"python scripts/setup_downloader.py（会自动安装 {req.name}）"
                 if req.is_file() else "检查 download.downloader_dir 是否指向含 run.py 的目录"),
            need_setup=True,
        ))

    # 5. ollama（可选：完全离线模式的进阶依赖）
    # 进程内直接探测：起 `python -c` 子进程在只有 py 启动器的机器上会静默误报"未运行"
    ollama_up = False
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        ollama_up = True
    except OSError:
        pass
    checks.append(_check("ollama(可选)", ollama_up,
                         "运行中" if ollama_up else "未运行（仅完全离线模式需要，默认不需要）"))

    # 5. 可选：真实调用一次供应商验证 key（默认关，避免费 token）
    if do_probe_calls and providers:
        from .providers.llm import chat
        try:
            r = chat([{"role": "user", "content": "回复:ok"}], providers, timeout=30)
            checks.append(_check("provider_probe", True, f"{r.provider}/{r.model} 实测连通"))
        except Exception as e:  # noqa: BLE001
            checks.append(_check("provider_probe", False, f"实测失败：{e}",
                                 fix="检查 key 是否有效、api_base 是否正确"))

    # 必需项全部 OK 才算真的能开跑。只看模型会造成假绿：
    # 模型配好了但 STT/ffmpeg 缺失时，第一条视频仍会失败（首次安装实测踩到）。
    # config(unknown_keys) 只警告不阻断——拼写错误不该让工具没法启动。
    ok = all(c["status"] == "ok" for c in checks
             if not c["name"].endswith("(可选)")
             and c["name"] != "config(unknown_keys)")
    return {"ok": ok, "need_setup": [c for c in checks if c["status"] == "NEED_SETUP"], "checks": checks}


def print_report(report: dict, as_json: bool = False) -> int:
    import json
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for c in report["checks"]:
            mark = {"ok": "✅", "NEED_SETUP": "🔑", "missing": "❌"}.get(c["status"], "·")
            print(f"{mark} {c['name']}: {c['detail']}")
            if c["status"] != "ok" and c.get("fix"):
                print(f"   ↳ 怎么补：{c['fix']}")
        print("\n结论：" + ("可以开跑 ✅" if report["ok"] else
              "还差必需配置/依赖 🔑 —— 按上面的提示补齐后重跑 dystill doctor"))
    return 0 if report["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(print_report(run_doctor()))
