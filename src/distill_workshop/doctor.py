"""体检与安装向导：agent 友好。

设计目标：用户装完第一件事跑 `dystill doctor`，缺什么直接给 NEED_SETUP 结构化输出，
agent 可解析并向用户要"一个 key"，而不是让用户读安装文档。
"""
from __future__ import annotations

import shutil
import sys
from importlib.util import find_spec

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
        fix="复制 config.example.yaml 为 config.yaml，填一个带视觉能力的对话模型（key/api_base/model）；"
            "或设置环境变量 DISTILL_API_BASE / DISTILL_API_KEY / DISTILL_MODEL 后运行 `dystill init`",
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

    # 3. STT 后端
    if cfg.stt.backend == "faster-whisper":
        has_stt = find_spec("faster_whisper") is not None
        checks.append(_check(
            "stt(faster-whisper)", has_stt,
            "已安装" if has_stt else "未安装（首次转录将自动下载 small 模型 ~500MB）",
            fix="pip install 'dystill[local-stt]' 或 pip install faster-whisper",
        ))
    else:
        checks.append(_check("stt", True, f"backend={cfg.stt.backend}（云端，随 providers 走）"))

    # 3. ffmpeg（下载/切帧/分段转写依赖）
    has_ffmpeg = shutil.which("ffmpeg") is not None
    checks.append(_check(
        "ffmpeg", has_ffmpeg,
        shutil.which("ffmpeg") or "未找到",
        fix="安装 ffmpeg 并加入 PATH（winget install Gyan.FFmpeg / apt install ffmpeg）",
    ))

    # 4. 切帧依赖（scene 切镜头 + Pillow 缩放/感知哈希）——缺了不是"少个功能"，是"产物质量塌"
    has_pil = find_spec("PIL") is not None
    checks.append(_check(
        "pillow", has_pil,
        "已安装" if has_pil else "未安装（切出帧后缩放/去重会直接报错）",
        fix="pip install 'dystill' 重装（Pillow 已是本体依赖）；或 pip install Pillow",
    ))
    has_scene = find_spec("scenedetect") is not None
    checks.append(_check(
        "scenedetect", has_scene,
        "已安装（按镜头切帧）" if has_scene else "未安装（会退回整片抽 1 帧，画面理解质量大幅下降）",
        fix="pip install 'dystill' 重装（scenedetect 已是本体依赖）；或 pip install scenedetect",
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

    # 结论只看两项：providers 就绪 + 有带视觉的供应商（视频蒸馏主场景跑不跑得动）。
    # 只查 providers 会假绿：唯一 provider 标了 vision: false 时画面理解必断（实测踩到）。
    ok = all(c["status"] != "NEED_SETUP"
             for c in checks if c["name"] in ("providers", "provider(vision)"))
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
              "还差配置 🔑 —— 把上面的 NEED_SETUP 补齐后重跑 dystill doctor"))
    return 0 if report["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(print_report(run_doctor()))
