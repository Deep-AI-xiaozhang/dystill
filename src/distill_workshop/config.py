"""配置层：所有路径 / 模型 / key 只允许从这里进。

纪律（来自实测教训）：
- 代码中禁止出现任何盘符硬编码；一切路径经 AppConfig 解析。
- key 一律环境变量或 config.yaml（后者必须被 .gitignore 覆盖）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_NAME = "config.yaml"
_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _expand(value):
    """展开 ${ENV}；未设置的环境变量展开为空串，交由上层校验报 NEED_SETUP。"""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def find_config_file(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        p = candidate / CONFIG_NAME
        if p.is_file():
            return p
    return None


@dataclass
class ProviderConfig:
    name: str = "main"
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    max_attempts: int = 3
    vision: bool = True     # 该模型能否读图。文本模型必须显式写 vision: false，
                            # 否则画面理解会先打到一个读不了图的接口上、白等一轮再降级
    max_tokens: int = 0     # 该模型的硬上限（0=不指定）。有些免费模型对 max_tokens 卡得很死
                            # （智谱 glm-4v-flash 超过 1024 直接报 1210 参数非法），
                            # 写在 provider 上就不必让调用方记住每家的怪癖
    api_key_file: str = ""  # 从文件读 key（各家免费通道都是 xxx_key.txt 这种）。
                            # 设了它就不必把密钥复制进 config.yaml——少一处泄露面
    extra_body: dict = field(default_factory=dict)
    # 供应商专属的请求参数，原样合并进请求体。用途：思考型模型的思考档位
    # （如 {"reasoning_effort": "none"} 关掉思考链）、各家特有的开关等。
    # 合并顺序在最后——配了的键会覆盖 model/messages/temperature 等默认字段。
    _key_file_error: str = ""   # 内部：key 文件读不到时记原因，交给 doctor 报出来

    @property
    def ready(self) -> bool:
        # key 文件读失败时即使路径写着也不算就绪——否则 doctor 会报"可用"而调用时才炸
        has_key = bool(self.api_key) or bool(self.api_key_file and not self._key_file_error)
        return bool(self.api_base and has_key and self.model)


@dataclass
class STTConfig:
    backend: str = "faster-whisper"
    model: str = "small"
    # device: auto / cuda / cpu。auto = 有 CUDA 就用 CUDA
    device: str = "auto"
    # compute_type: 空 = 自动解析（CUDA→float16，CPU→int8）。
    # ⚠️ 不要依赖 ctranslate2 自己的 "auto"：它在 CUDA 上会选 int8_float16，
    # 实测会静默丢整段内容（见 DECISIONS D38）。
    compute_type: str = ""


@dataclass
class DownloadConfig:
    """下载层配置：douyin-downloader 本地 checkout（MIT, github.com/jiji262/douyin-downloader）。"""
    downloader_dir: str = ""    # 含 run.py 的目录；空 = 未配置
    config_yml: str = ""        # 下载器 config.yml（继承 cookies/proxy，不落盘凭据）


@dataclass
class AppConfig:
    root: Path
    notes_dir: Path
    data_dir: Path
    cache_dir: Path
    providers: list[ProviderConfig] = field(default_factory=list)
    stt: STTConfig = field(default_factory=STTConfig)
    pipeline: dict = field(default_factory=dict)
    download: DownloadConfig = field(default_factory=DownloadConfig)

    @property
    def ready_providers(self) -> list[ProviderConfig]:
        return [p for p in self.providers if p.ready]

    @property
    def vision_providers(self) -> list[ProviderConfig]:
        """能读图的供应商（画面理解专用）。列表里混合文本模型时，靠这个挑出对的那些。"""
        return [p for p in self.providers if p.ready and p.vision]


def _known_keys(cls, raw: dict | None) -> dict:
    """只保留 dataclass 认得的键。

    配置里多写一个没用的键（拼错、或从别处抄来的 timeout/language）不该让整个工具崩：
    实测 `stt: {language: zh}` 会直接 TypeError 起不来。providers 段早有这层过滤，
    stt / download 段漏了——统一走这里。
    """
    return {k: v for k, v in (raw or {}).items() if k in cls.__dataclass_fields__}


def load_config(start: Path | None = None) -> AppConfig:
    cfg_path = find_config_file(start)
    raw: dict = {}
    if cfg_path:
        raw = _expand(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})

    base_dir = cfg_path.parent if cfg_path else Path.cwd()
    root_raw = Path(raw["root"]) if raw.get("root") else base_dir
    root = (root_raw if root_raw.is_absolute() else base_dir / root_raw).resolve()
    ws = raw.get("workspace") or {}

    def workspace_path(name: str, default: str) -> Path:
        raw_path = Path(ws[name]) if ws.get(name) else Path(default)
        return (raw_path if raw_path.is_absolute() else root / raw_path).resolve()

    notes = workspace_path("notes_dir", "notes")
    data = workspace_path("data_dir", "data")
    cache = workspace_path("cache_dir", "cache")

    providers = []
    for p in raw.get("providers") or []:
        if not isinstance(p, dict):
            continue
        # 只收 ProviderConfig 认得的字段：用户多写 timeout/stream 这类常见键时，
        # dataclass 构造会直接 TypeError 把整个工具打崩（实测踩到）。未知键丢弃而不是报错——
        # 配置的目的是\"能跑\"，多写一个没用的键不该让工具没法启动。
        known = {k: v for k, v in p.items() if k in ProviderConfig.__dataclass_fields__}
        providers.append(ProviderConfig(**{"name": "main", "max_attempts": 3, **known}))
    # key 文件在加载时就解析成 api_key：下游只认 api_key，不必各自记住"有的家 key 在文件里"
    for p in providers:
        if not p.api_key and p.api_key_file:
            try:
                p.api_key = Path(p.api_key_file).expanduser().read_text(encoding="utf-8").strip()
            except OSError as e:
                p._key_file_error = f"{p.api_key_file}（{e.strerror or e}）"
    # 隐式 provider：三个环境变量齐全即视为最小配置（无需 config.yaml 就能跑）
    if not providers:
        env_base, env_key, env_model = (os.environ.get(k, "") for k in ("DISTILL_API_BASE", "DISTILL_API_KEY", "DISTILL_MODEL"))
        if env_base and env_key and env_model:
            providers = [ProviderConfig(name="env", api_base=env_base, api_key=env_key, model=env_model)]
    stt = STTConfig(**{"backend": "faster-whisper", "model": "small",
                       **_known_keys(STTConfig, raw.get("stt"))})

    return AppConfig(
        root=root, notes_dir=notes, data_dir=data, cache_dir=cache,
        providers=providers, stt=stt, pipeline=raw.get("pipeline") or {},
        download=DownloadConfig(**_known_keys(DownloadConfig, raw.get("download"))),
    )
