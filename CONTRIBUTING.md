# 贡献指南

感谢你愿意改进蒸馏工坊。这个项目的定位很窄：**只做抖音、本地优先、笔记里的事实要能回查到原素材**。
提交前请先确认你的改动没有把范围扩成"通用视频下载器/多平台蒸馏"。

## 开发环境

- Python **3.10 – 3.13**（CI 覆盖 3.10 / 3.12，Windows 跑 3.12）
- 处理视频需要系统里可用的 **ffmpeg**（本地文件路径）；只做单元测试不需要
- 本地 STT 需要可选依赖 `faster-whisper`；B 站分流需要 `yt-dlp`；抖音浏览器兜底需要 `playwright`

```bash
git clone https://github.com/zhj-521-521/dystill.git
cd dystill

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"          # 本体 + pytest/ruff/build/twine
# 需要本地转写时：pip install -e ".[dev,local-stt]"
```

> 占位符 `OWNER` 发布前会替换成真实仓库地址。

## 跑测试与检查

提交 PR 前这两条必须全绿：

```bash
pytest -q
ruff check .
```

- 测试全部用临时目录和假 provider，不需要真实 API key，也不应该联网。
- 新增行为请补测试；纯注释/文档改动可以说明原因。
- 修 bug 时优先补一个能复现旧行为的测试，再改实现。

改配置模板时注意：**根目录 `config.example.yaml` 与 `src/distill_workshop/config.example.yaml`
必须逐字节一致**（`tests/test_cli.py::test_config_example_templates_stay_in_sync` 会兜底）。
`cli init` 读的是包内那份，README 让人看根目录那份，漂移会让用户改了模板却不生效。

## 打包相关改动

动了 `pyproject.toml`、`MANIFEST.in`、包内数据文件或依赖时，请本地验证：

```bash
python -m build                 # 产出 dist/*.whl 与 dist/*.tar.gz
twine check dist/*
python -m venv /tmp/smoke && /tmp/smoke/bin/pip install dist/*.whl
/tmp/smoke/bin/dystill --version
```

注意 `license = "MIT"` + `license-files` 走的是 PEP 639，要求 `setuptools>=77`；
降级 setuptools 会生成缺少 `License-Expression` 的 METADATA。

## 如何提 PR

1. Fork 仓库，从 `main`（或维护者指定的发布分支）切一个短分支：`fix/xxx`、`feat/xxx`、`docs/xxx`。
2. 一个 PR 只做一件事；提交信息写清"为什么"，不要只写"update"。
3. 填写 PR 模板里的自测清单，贴 `pytest -q` / `ruff check .` 的结果。
4. 修 bug 请关联 issue（`Closes #123`）；新功能请先开 issue 对齐范围，避免白写。
5. 维护者可能会要求补测试、拆 PR 或调整命名；合并前 CI 必须绿。

## 绝对不要提交

以下内容属于用户隐私或本机产物，仓库已用 `.gitignore` 挡住，但请不要用 `git add -f` 绕过：

- `config.yaml`、`config.yaml.bak*`、`.env`、`*.key`、`*_key.txt`、`*_token.txt`、cookie 文件
- 任何 **API key / token / cookie / 代理账号**
- `data/`、`cache/`、`output/`、`outputs/`、`notes/`（含用户视频与笔记原文）
- `bench/`（基准集与标注，含个人内容）
- `external/`（第三方 checkout，例如 douyin-downloader，请按 README 自行拉取）
- `build/`、`dist/`、`.venv/`、`__pycache__/`

提交前建议自查：

```bash
git status --short
git diff --cached | Select-String -Pattern "sk-|api_key|token|cookie"   # PowerShell
# 或：git diff --cached | grep -nE "sk-|api_key|token|cookie"
```

## 许可证

提交代码即表示你同意以本仓库的 **MIT License** 授权你的贡献。
涉及第三方代码/资源时，请在 `THIRD_PARTY_NOTICES.md` 补一条来源与许可证说明。
