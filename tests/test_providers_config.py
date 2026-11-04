"""多供应商配置与画面理解的挑人逻辑。

为什么值得单独测：项目支持"快文本模型跑判定 + 带视觉的模型跑画面理解"的混搭，
但**画面理解只能打在能读图的接口上**。挑错的后果不是报错，而是先浪费一轮请求再降级，
或者在最坏情况下把图片发给纯文本模型、拿到一段不知道在说什么的回答。
"""
from pathlib import Path

import pytest

from distill_workshop.config import AppConfig, ProviderConfig
from distill_workshop.core import pipeline

CFG = """
workspace:
  data_dir: ""
providers:
  - name: fast
    api_base: https://api-inference.modelscope.cn/v1
    api_key: k1
    model: Qwen/Qwen3.8-27B
    vision: false
  - name: vision
    api_base: https://token.sensenova.cn/v1
    api_key: k2
    model: sensenova-6.8-flash-lite
stt:
  backend: faster-whisper
"""


def _load(tmp_path: Path, text: str) -> AppConfig:
    from distill_workshop.config import load_config
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
    return load_config(tmp_path)


def test_vision_flag_filters_and_defaults_true(tmp_path: Path):
    app = _load(tmp_path, CFG)
    assert [p.name for p in app.ready_providers] == ["fast", "vision"]
    assert [p.name for p in app.vision_providers] == ["vision"]     # fast 标了 false
    assert app.vision_providers[0].vision is True                   # 没写的默认能读图


def test_provider_not_ready_is_excluded_from_both(tmp_path: Path):
    app = _load(tmp_path, CFG.replace("api_key: k1", "api_key: \"\""))
    assert [p.name for p in app.ready_providers] == ["vision"]
    assert [p.name for p in app.vision_providers] == ["vision"]


def test_unknown_provider_keys_are_ignored_not_crash(tmp_path: Path):
    """多写 timeout/stream 这类常见键不该把工具打崩（B2）——丢未知键，其余照常。"""
    app = _load(tmp_path, """
providers:
  - name: fast
    api_base: https://api-inference.modelscope.cn/v1
    api_key: k1
    model: Qwen/Qwen3.8-27B
    vision: false
    timeout: 30
    stream: false
""")
    assert [p.name for p in app.ready_providers] == ["fast"]
    assert app.ready_providers[0].model == "Qwen/Qwen3.8-27B"
    # 非 dict 的 providers 条目（如误写成字符串）也应跳过而不是崩溃
    app2 = _load(tmp_path, "providers:\n  - nope\n")
    assert app2.providers == []


def test_default_vlm_uses_vision_providers(tmp_path: Path, monkeypatch):
    app = _load(tmp_path, CFG)
    seen = {}

    def fake_describe(frames, prompt, providers, timeout=180, **kw):
        seen["names"] = [p.name for p in providers]
        seen["kw"] = kw
        return type("R", (), {"content": "画面描述"})()

    monkeypatch.setattr("distill_workshop.providers.vision.describe_images", fake_describe)
    out = pipeline._default_vlm([tmp_path / "a.jpg"], app, lambda m: None)
    assert out == "画面描述"
    assert seen["names"] == ["vision"]     # 绝不能把图发给纯文本模型
    assert seen["kw"]["batch_size"] == 1   # 默认逐帧（DECISIONS D7）


def test_default_vlm_raises_when_no_vision_provider(tmp_path: Path):
    no_vision = """
providers:
  - name: fast
    api_base: https://api-inference.modelscope.cn/v1
    api_key: k1
    model: Qwen/Qwen3.8-27B
    vision: false
  - name: also-text
    api_base: https://token.sensenova.cn/v1
    api_key: k2
    model: sensenova-6.8-flash-lite
    vision: false
"""
    app = _load(tmp_path, no_vision)
    assert app.ready_providers and not app.vision_providers
    with pytest.raises(RuntimeError) as e:
        pipeline._default_vlm([tmp_path / "a.jpg"], app, lambda m: None)
    assert "vision" in str(e.value)


def test_doctor_flags_missing_vision_provider(tmp_path: Path, monkeypatch):
    from distill_workshop import doctor
    app = _load(tmp_path, """
providers:
  - name: only-text
    api_base: https://api-inference.modelscope.cn/v1
    api_key: k1
    model: Qwen/Qwen3.8-27B
    vision: false
""")
    monkeypatch.setattr(doctor, "load_config", lambda *a, **k: app)
    report = doctor.run_doctor()
    vis = next(c for c in report["checks"] if c["name"] == "provider(vision)")
    assert vis["status"] == "NEED_SETUP" and vis["fix"]


def test_doctor_does_not_claim_ready_without_required_stt(tmp_path: Path, monkeypatch):
    """模型配好不等于能跑：缺 faster-whisper 时体检必须判未就绪。"""
    from distill_workshop import doctor

    app = _load(tmp_path, """
providers:
  - name: vision
    api_base: https://x/v1
    api_key: k
    model: m
""")
    monkeypatch.setattr(doctor, "load_config", lambda *a, **k: app)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(doctor, "find_spec",
                        lambda name: None if name == "faster_whisper" else object())

    report = doctor.run_doctor()
    stt = next(c for c in report["checks"] if c["name"] == "stt(faster-whisper)")
    assert stt["status"] == "NEED_SETUP"
    assert "pip install faster-whisper" in stt["fix"]
    assert not report["ok"]


def test_provider_config_defaults(monkeypatch):
    p = ProviderConfig(name="x", api_base="b", api_key="k", model="m")
    assert p.vision is True and p.ready is True
    assert ProviderConfig(api_base="b", api_key="k").ready is False    # 缺 model 不算就绪


def test_api_key_file_is_read_at_load(tmp_path: Path):
    """key 文件在加载时解析：密钥不必复制进 config.yaml（少一处泄露面）。"""
    kf = tmp_path / "agnes_key.txt"
    kf.write_text("  sk-from-file\n", encoding="utf-8")
    app = _load(tmp_path, f"""
providers:
  - name: agnes
    api_base: https://api.agnes-ai.cn/v1
    api_key_file: "{kf.as_posix()}"
    model: agnes-3.0-flash
""")
    p = app.providers[0]
    assert p.api_key == "sk-from-file" and p.ready and not p._key_file_error


def test_api_key_takes_precedence_over_file(tmp_path: Path):
    kf = tmp_path / "k.txt"
    kf.write_text("from-file", encoding="utf-8")
    app = _load(tmp_path, f"""
providers:
  - name: a
    api_base: https://x/v1
    api_key: from-inline
    api_key_file: "{kf.as_posix()}"
    model: m
""")
    assert app.providers[0].api_key == "from-inline"


def test_missing_key_file_marks_not_ready_and_doctor_reports(tmp_path: Path, monkeypatch):
    """key 文件路径写错时不能静默：provider 判不就绪，doctor 明说原因。"""
    from distill_workshop import doctor
    app = _load(tmp_path, f"""
providers:
  - name: agnes
    api_base: https://api.agnes-ai.cn/v1
    api_key_file: "{(tmp_path / 'nope.txt').as_posix()}"
    model: agnes-3.0-flash
""")
    p = app.providers[0]
    assert not p.ready and "nope.txt" in p._key_file_error
    monkeypatch.setattr(doctor, "load_config", lambda *a, **k: app)
    report = doctor.run_doctor()
    ck = next(c for c in report["checks"] if c["name"] == "provider(keyfile)")
    assert ck["status"] == "NEED_SETUP" and "agnes" in ck["detail"]

def test_config_ignores_unknown_keys(tmp_path):
    """配置里多写没用的键不该让工具崩——stt/download 段曾直接 TypeError 起不来。"""
    from distill_workshop.config import load_config

    (tmp_path / "config.yaml").write_text(
        "stt:\n  backend: faster-whisper\n  model: small\n  language: zh\n"
        "download:\n  downloader_dir: x\n  timeout: 30\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    assert cfg.stt.model == "small"
    # 相对路径按配置文件所在目录解析成绝对路径（不再随当前工作目录漂移）
    assert Path(cfg.download.downloader_dir) == (tmp_path / "x").resolve()


def test_relative_workspace_paths_are_resolved_from_root(tmp_path):
    """配置里的相对目录应跟着 root，不应跟着命令执行时的当前目录漂移。"""
    app = _load(tmp_path, """
root: project
workspace:
  notes_dir: notes
  data_dir: data
  cache_dir: cache
""")
    assert app.root == (tmp_path / "project").resolve()
    assert app.notes_dir == app.root / "notes"
    assert app.data_dir == app.root / "data"
    assert app.cache_dir == app.root / "cache"


def test_relative_config_paths_follow_config_dir_not_cwd(tmp_path, monkeypatch):
    """配置里的相对路径必须相对配置文件解析，不能随「在哪个目录敲命令」漂移。

    回归（2026-11-04）：`downloader_dir: external/douyin-downloader` 是相对写法，
    从 docs/ 子目录跑 dystill doctor 时报「配置目录不存在」；api_key_file 同理。
    """
    from distill_workshop.config import load_config

    kf = tmp_path / "keys" / "k.txt"
    kf.parent.mkdir()
    kf.write_text("KEY123", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "download:\n  downloader_dir: external/douyin-downloader\n  config_yml: downloader.base.yml\n"
        "providers:\n  - name: p\n    api_base: http://x\n    api_key_file: keys/k.txt\n    model: m\n",
        encoding="utf-8")
    # 模拟在别的目录执行：CWD 与配置文件目录不同
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(tmp_path)
    assert Path(cfg.download.downloader_dir) == (tmp_path / "external/douyin-downloader").resolve()
    assert Path(cfg.download.config_yml) == (tmp_path / "downloader.base.yml").resolve()
    assert Path(cfg.providers[0].api_key_file) == kf.resolve()
    assert cfg.providers[0].api_key == "KEY123"


def test_unknown_stt_backend_is_reported_as_setup_needed(tmp_path, monkeypatch):
    """配了不存在的 STT 后端必须报 NEED_SETUP，不能报绿。

    回归（2026-11-04）：doctor 原来对非 faster-whisper 的 backend 写「云端，随 providers 走」
    并判 ok，但代码里根本没有云端实现——doctor 说能跑，实际 pipeline 抛「未配置 STT 后端」。
    """
    from distill_workshop import doctor

    (tmp_path / "config.yaml").write_text("stt:\n  backend: some-cloud-api\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "load_config", lambda *a, **k: __import__(
        "distill_workshop.config", fromlist=["load_config"]).load_config(tmp_path))
    report = doctor.run_doctor()
    ck = next(c for c in report["checks"] if c["name"] == "stt")
    assert ck["status"] == "NEED_SETUP"
    assert "只实现了 faster-whisper" in ck["detail"]
