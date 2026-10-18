# 第三方组件与许可证声明

本项目以 **MIT License** 发布（见 `LICENSE`）。下面列出运行期/可选路径实际使用或对接的第三方项目。
各项目的版权与许可证归其原作者所有；本文件只做归属说明，不改变上游许可证。
发布二进制包或再分发时，请保留上游要求的版权与许可证文本。

## 核心运行依赖

| 项目 | 许可证 | 用途 | 引入方式 |
| --- | --- | --- | --- |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | 读取 `config.yaml` 与缓存/元数据 | `pyproject.toml` 硬依赖 |
| [requests](https://github.com/psf/requests) | Apache-2.0 | 调用 OpenAI 兼容接口、下载素材 | 硬依赖 |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND（Historical Permission Notice and Disclaimer，PIL 系） | 抽帧缩放、感知哈希、图片编码 | 硬依赖 |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | BSD-3-Clause | 镜头切分（切帧主路径） | 硬依赖 `scenedetect` |

## 可选路径 / 外部工具

| 项目 | 许可证 | 用途 | 引入方式 |
| --- | --- | --- | --- |
| [douyin-downloader](https://github.com/jiji262/douyin-downloader) | MIT（© jiji262 及贡献者） | 抖音视频下载（含风控兜底链路） | `external/` 第三方 checkout，不随本仓库分发 |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Unlicense | B 站链接分流下载 | 可选依赖 `[bilibili]` |
| [Playwright (Python)](https://github.com/microsoft/playwright-python) | Apache-2.0 | 真浏览器取抖音播放地址 | 可选依赖 `[douyin-browser]`，另需 `playwright install chromium` |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | 本地语音转写（本地 STT 路径） | 可选依赖 `[local-stt]` |
| [FFmpeg](https://ffmpeg.org/) | LGPL-2.1+ / GPL（取决于构建选项） | 音视频解码、抽音频、切片 | **系统外部程序**，不随本仓库分发 |

## 说明

- `douyin-downloader` 以 git checkout 形式放在 `external/`（已被 `.gitignore` 排除），
  本仓库不内嵌其源码；按 README 自行拉取时请遵守其 MIT 许可证与免责声明。
  抖音下载涉及平台条款与版权，使用者需自行确保对素材拥有合法使用权。
- `faster-whisper` 首次运行会从 Hugging Face 下载模型权重；模型权重有各自的许可证，
  以模型卡（model card）为准。
- FFmpeg 的许可证取决于其编译选项（`--enable-gpl` 等），分发前请确认所用构建的许可条款。
- 如果你新增了第三方代码/数据/模型，请在 PR 里同步更新本文件。
