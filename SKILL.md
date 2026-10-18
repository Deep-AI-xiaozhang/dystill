---
name: dystill
description: 把抖音视频（或本地媒体文件）蒸馏成「不瞎编、可回查」的笔记，并用核对引擎逐条验证：每条声明回原视频找出处，产出 ✅已抽核/❌不一致/⚠️存疑 + 时间戳报告。Use when 用户发来抖音链接要总结/存笔记、要求"蒸馏"视频、要求核对/验真已有 AI 笔记、或提到 dystill / 蒸馏工坊；也用于带用户从零安装并配置本工具。
version: 0.1.0
---

# 蒸馏工坊（dystill）· Agent Skill

CLI 是本 skill 的执行层：你负责编排与向用户转述，流水线与判定纪律由 `dystill` 保证。
**不要绕过 CLI 自己"总结视频"**——那正好丢掉了本产品唯一的核心价值（可回查的核对）。

## 上手引导（agent 必须按顺序执行，不要跳步）

读完这一节就该能开工：每个阶段都写了**跑什么命令 / 怎么算成功 / 失败了问用户什么 / 对应哪个配置项**。
新环境先跑 `dystill doctor --json`，按返回的 `need_setup` 逐项补，不要凭印象猜缺什么。
`doctor --json` 结构固定：`{"ok": bool, "need_setup": [...], "checks": [...]}`，每项是 `{name, status, detail, fix}`，`status` 取 `ok / NEED_SETUP / missing`；`dystill doctor` 退出码 0 = 可以开跑，2 = 还差配置。

### 阶段 0 · 环境体检（装之前）

```bash
python --version     # 需要 3.10+
git --version        # 拉源码、拉下载层要用
ffmpeg -version      # 切帧 / 抽音轨 / 分段转写都依赖它
```

- **成功判据**：三条都打印出版本；Python ≥ 3.10。
- **失败了怎么问用户**：一次问全，别挤牙膏——「你的系统是 Windows 还是 macOS/Linux？有没有权限装 ffmpeg（Windows `winget install Gyan.FFmpeg`，Linux `apt install ffmpeg`，macOS `brew install ffmpeg`）？Python 低于 3.10 的话能不能换一个版本？」
- **配置项**：无。ffmpeg 只要求进 PATH（阶段 2 的 doctor 会复查）。

### 阶段 1 · 装本体与系统依赖

```bash
git clone <本仓库地址> distill-workshop && cd distill-workshop
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -e .                 # 本体；Pillow / scenedetect 随它装
pip install -e '.[local-stt]'    # 推荐：本地转写 faster-whisper（自动下模型，无需 key）
dystill --version
```

- **成功判据**：`dystill --version` 打印 `dystill 0.1.0`。
- **失败了怎么问用户**：先分清网络还是编译——「把 pip 最后 20 行贴给我」；再问「有没有 NVIDIA 显卡」（只影响 `stt.device` / `stt.compute_type`，没有也能用 CPU 跑）。
- **配置项**：无。PyPI 尚未发布，**不要 `pip install dystill`**，只认源码安装。

### 阶段 2 · 配模型通道（唯一必填的一步）

```bash
dystill init            # 生成 config.yaml；已存在会拒绝，要覆盖加 --force（先备份）
dystill doctor --json   # 按 need_setup 逐项补
```

至少填一个**带视觉**的模型：`providers[0]` 的 `api_base` / `api_key` / `model`，并保持 `vision: true`。
也可以不写配置文件，改为 `export DISTILL_API_BASE=... DISTILL_API_KEY=... DISTILL_MODEL=...`（三个齐全才等效一个 provider；模板里的 `${DISTILL_*}` 占位也会自动展开）。

- **成功判据**：`doctor --json` 的 `ok` 为 `true`、`need_setup` 为空；`providers` 与 `provider(vision)` 两项都是 `ok`。
- **失败了怎么问用户**：按 `need_setup[].name` 分支问，别一次抛十个问题——
  - `providers`：想用哪家免费通道（见下方「免费通道」表）？请用户自己注册，key 自己填。
  - `provider(vision)`：你填的模型能读图吗？不能就把 `vision` 改成 `false`，再补一个能读图的；**至少一个 `vision: true` 是硬要求**，否则视频的画面理解必断。
  - `provider(keyfile)`：`api_key_file` 指向的文件读不到，核对路径和文件是否还在。
  - `stt(faster-whisper)`：装 `pip install -e '.[local-stt]'`，或把 `stt.backend` 换成云端 STT。
  - `pillow` / `scenedetect`：重装本体 `pip install -e .`。
  - `ffmpeg`：回阶段 0。
- **配置项**：`providers[].api_base` / `api_key` / `model` / `vision` / `max_tokens` / `max_attempts` / `api_key_file` / `extra_body`；环境变量 `DISTILL_API_BASE` / `DISTILL_API_KEY` / `DISTILL_MODEL`。

### 阶段 3 · 抖音下载层（可选，只处理抖音链接时需要）

```bash
python scripts/setup_downloader.py          # 默认走公开 GitHub 加速前缀；直连加 --direct
# 把脚本打印的两行填进 config.yaml 的 download 段（模板里已留了空段）：
# download:
#   downloader_dir: "<仓库>/external/douyin-downloader"   # 含 run.py 的目录
#   config_yml: ""
```

- **成功判据**：脚本打印 `[✓] 下载层就位：<路径>`，且 `<路径>/run.py` 存在。**doctor 不检查 `download` 段**（它只体检模型/转写/切帧），这步以脚本输出为准；真验证要跑一条抖音链接。
- **失败了怎么问用户**：「clone 失败多半是网络。要不要 `--direct` 直连 GitHub，或手动 clone 到 `external/douyin-downloader` 再用 `--dest` 指过去？」
- **配置项**：`download.downloader_dir` / `download.config_yml`；`external/` 已在 `.gitignore`，不进仓库。
- 可选兜底：`pip install -e '.[douyin-browser]'` + `playwright install chromium`（真浏览器取播放地址）；仍不行就改走本地文件，**别再折腾 cookie**。

### 阶段 4 · 验证（一条最小链路）

```bash
dystill doctor --probe                       # 真调一次供应商，验证 key 与连通（花少量 token）
dystill run <本地视频或音频文件> --json       # 不碰下载层的最小验证
```

- **成功判据**：`doctor --probe` 出现 `provider_probe: ... 实测连通`；`run` 退出码 0，`--json` 结果里 `status` 为 `ok`（`skipped` = 之前跑过、笔记有效），且 `notes_dir/<记录>/note.md` 存在非空。
- **失败了怎么问用户**：把 run 最后一段错误原样贴出来，再按类分支——401/403 查 key 与 `api_base`；HTTP 200 但正文为空换下家或调大 `max_tokens`；画面理解全失败查 `vision: true` 那家是否真能读图；转写失败确认 `faster-whisper` 已装。
- **配置项**：`providers[].max_tokens` / `extra_body`、`stt.model` / `stt.device`。

### 阶段 5 · 开跑

```bash
dystill run <抖音链接> --check          # 蒸馏 + 顺手核对（--check 默认 standard，可接 fast/deep）
dystill check <记录目录|笔记文件|链接> --mode standard
dystill kb --check                      # 知识库自检；--frontmatter 补字段，-o 指定索引路径
dystill clean                           # 只报告；--apply 才真删，--cache-days N 改保留天数
```

- **成功判据**：`run` 打印 `✅ 完成（…）：<记录名>` 且退出码 0；`check` 产出 `check.md` 徽章；`kb --check` 退出码 0（没有 ERROR）。
- **失败了怎么问用户**：429/限流时问「要不要把 `--delay` 调大（如 5），或用 `--retry-failed` 续跑？成功条目不会重复花钱。」
- **配置项**：`pipeline.judge_provider` / `check_batch_delay` / `check_max_claims` / `cache_keep_days` / `workers`；`workspace.notes_dir` / `data_dir` / `cache_dir`。

## 免费通道（优先让用户走这条，全流程可以不花钱）

| 通道 | API 地址 | 读图 | 适合 |
|---|---|---|---|
| AgnesAI | `https://api.agnes-ai.cn/v1` | ✅ | 画面理解主力，也能写笔记/判定 |
| 商汤日日新 | `https://token.sensenova.cn/v1` | ✅ | 备选；限流时换 |
| 硅基流动 | `https://api.siliconflow.cn/v1` | ❌ 纯文本 | 写笔记 + 判定（`vision: false`） |
| 魔搭 ModelScope | `https://api-inference.modelscope.cn/v1` | ❌ 纯文本 | 备选判定 / 写笔记 |

参考模型名（**以各家控制台当前列表为准**）：AgnesAI `agnes-3.0-flash`、商汤 `sensenova-6.8-flash-lite`、硅基流动/魔搭 `DeepSeek-V4-Flash`。免费额度、模型上下架都会变，本文件不写死额度与速度；换模型先跑 `scripts/vlm_probe.py` 实测。

**配法建议**：一个带视觉的（画面理解）+ 一个纯文本的（写笔记/判定），各自标对 `vision`；`pipeline.judge_provider` 填 `providers[].name`，逗号分隔多个即判定专用降级链——建议挂**两家不同供应商**互为备份，防单家日限流。名字必须与 `providers[].name` 一致。

## 每个环节需要什么

| 环节 | 需要的组件 | 是否必需 | 免费方案 | 配置位置 |
|---|---|---|---|---|
| 转写 | faster-whisper（本地）或云端 STT | 视频/音频必需；图文帖不需要 | 本地 faster-whisper，自动下模型、无需 key | `stt.backend` / `stt.model` / `stt.device` / `stt.compute_type` |
| 画面理解 | 带视觉的对话模型（`vision: true`） | 视频必需；纯音频、图文帖不走这路 | AgnesAI、商汤日日新 | `providers[].vision` / `pipeline.vlm_max_frames` / `pipeline.vlm_batch_frames` / `pipeline.workers` |
| 笔记生成 | 对话模型（可与画面理解同一个） | 必需 | 硅基流动、魔搭、AgnesAI | `providers[].api_base` / `api_key` / `model`（按列表顺序降级） |
| 核对判定 | 判定模型，可单独一条链 | 用 `--check` 时必需 | 硅基流动、魔搭；再挂一家不同供应商备份 | `pipeline.judge_provider` / `pipeline.check_max_claims` / `pipeline.check_batch_delay` |
| 抖音下载 | douyin-downloader（第三方 checkout） | 仅抖音链接需要；本地文件不需要 | 开源（MIT），`scripts/setup_downloader.py` 拉取 | `download.downloader_dir` / `download.config_yml` |
| 知识库 | 无模型调用，只读本地笔记 | 可选 | 免费 | `workspace.notes_dir` / `workspace.data_dir`；`dystill kb -o` |

系统依赖（ffmpeg、Pillow、scenedetect）不占配置项，见阶段 1；其中 ffmpeg 必须在 PATH 里。

## 凭据纪律（key 怎么给才安全）

- **让用户自己填**：写进 `config.yaml`（已被 `.gitignore` 覆盖），或放 `*_key.txt` 用 `providers[].api_key_file` 指过去（同样被忽略），或 `export DISTILL_API_KEY=...`。
- **你不索要明文 key**，也不要让用户把 key 贴进对话、日志、issue。用户已经贴了 → 提醒他轮换。
- 不要 `echo` key、不要写进任何文件、不要提交。`doctor --probe` 只验证连通，不会打印 key。
- 收尾自检：`git status --short` 里不应出现 `config.yaml`、`*_key.txt`（出现了说明 `.gitignore` 被改过，先停手问用户）。

## 主流程

```bash
dystill run <抖音链接|本地文件> [--check fast|standard|deep]   # 蒸馏（可批量），顺手核对
dystill check <记录目录|笔记文件|链接> --mode standard          # 只核对已有笔记
dystill kb                                                    # 知识库：汇总所有笔记成索引 + 自检
```

流水线六步：① 下载（抖音/本地文件）→ ② 转写 → ③ 切帧 → ④ 画面理解 → ⑤ 生成笔记 → ⑥ 核对。

- 默认 `run --check`（standard）一步到位；用户急用可先 `run` 后 `check`。
- 素材类型自动识别：**视频** / **纯音频**（ffprobe 复核容器内部流，不看后缀）/ **抖音图文帖**（逐图理解，没有语音维度）。图文帖同样生成笔记并走同一套核对，只是没有转写那一路证据。
- 下载缓存默认保留 7 天（`pipeline.cache_keep_days`），超期文件在 `run` 时自动清理；`dystill clean` 可随时查看/清理（默认只报告，`--apply` 才真删）。
- 模式选择：批量/低价值 → `fast`；日常 → `standard`；关键学习/工作视频 → `deep`。不确定就 standard，**不要替用户静默降模式**。
- 长笔记超上限会按章节轮转抽样并写进报告——把"哪些没核"如实转述，不许只报 ✅ 数。
- 429/限流：加大 `--delay`（如 5）或用 `--retry-failed` 续跑，成功条目不重复花钱；`--resume` 完全复用上轮判定、只重建报告。

## 产物与转述纪律

| 文件 | 给谁 | 你怎么用 |
|---|---|---|
| `note.md`（顶部含徽章） | 用户 | 给路径 + 徽章一句话总结 |
| `check.md` | 用户复核 | 转述 ❌ 与 ⚠️ 条目及理由 |
| `check.json` | 机器 | 你要精确统计时读它 |

- 徽章措辞是「**已抽核**」，转述时保持这个词（降级理由见 `docs/DECISIONS.md` 的《徽章措辞降级：「已核对」→「已抽核」》）。
- `--rewrite` 只有用户明确要求"把不一致的句子改掉"时才加；默认不改写笔记正文。
- 向用户交代边界：核对只保证"每句话在原视频有出处"，不保证"概括得准"；基于转写文本，转写错误不在核对范围。

## 已知坑（doctor 未必都报得出来）

- 抖音单视频接口有平台级风控：CLI 会自动切真浏览器兜底；仍失败就走本地文件，**别再折腾 cookie**（见 `docs/DECISIONS.md` 的《抖音单视频下载已被平台封堵》与《抖音风控先回真浏览器兜底》）。
- B 站链接是可选组件：需 `pip install -e '.[bilibili]'`，缺了 CLI 会报可行动提示。
- **HTTP 200 但 `content` 为空 = 失败**，必须换下家。思考型模型开了思考档位时要同时调大 `max_tokens`，否则思考吃掉额度、只剩空正文，会被当成调用失败。
- `stt.compute_type` 别写 `int8_float16`：该量化会静默丢整段内容；留空让工具按设备自动选。
- 模型选型按实测不按新旧，`config.example.yaml` 注释里留了当前结论；换模型先跑 `scripts/vlm_probe.py` + `scripts/model_pick.py`。

## 验收/开发类请求

用户要"看核对引擎准不准"：跑 `scripts/bench_check.py`（单轮）+ `scripts/bench_summary.py`（口径 v2：多题材、多轮取最差）。
通过线与口径以 `docs/` 里的验收文档为准，本文件不写死数字。
**不要为了刷分顺手改闸门/词表/提示词纪律。**
