#!/usr/bin/env python
"""一键拉取第三方下载层（douyin-downloader），M3「全新机器 30 分钟跑通」的其中一步。

用法：
    python scripts/setup_downloader.py                      # 默认走 gh-proxy 加速（国内可用）
    python scripts/setup_downloader.py --direct             # 不走代理，直连 GitHub
    python scripts/setup_downloader.py --dest /path/to/douyin-downloader

为什么不把 douyin-downloader vendor 进本仓库：
1. 它是**别人的 MIT 项目**（jiji262），自带 .git——混进本仓库会让 license 归属和贡献者预期都变乱；
2. 它的更新节奏跟着抖音签名算法走，vendor 进来就意味着"平台一变，更新责任变成我们的"；
3. 本仓库会因为它多出几千个文件，噪声远大于便利。

为什么不自动改 config.yaml：那里存着你的 API key。脚本只克隆 + **打印**该填什么，
凭据文件永远由你自己动（这条纪律和 .gitignore 里"凭据永不入库"是同一件事）。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/jiji262/douyin-downloader"
PROXY = "https://gh-proxy.com/"          # 国内常见加速前缀；换镜像只改这一行
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "external" / "douyin-downloader"


def main() -> int:
    ap = argparse.ArgumentParser(description="拉取 douyin-downloader 到 external/（不入 git）")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="目标目录（默认 <repo>/external/douyin-downloader）")
    ap.add_argument("--direct", action="store_true", help="直连 GitHub（不走加速前缀）")
    ap.add_argument("--branch", default="", help="指定分支/标签（默认跟默认分支）")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    if not shutil.which("git"):
        print("[x] 没找到 git。先装 Git 再来（winget install Git.Git / apt install git）")
        return 2
    if (dest / ".git").is_dir():
        print(f"[=] 已经在了：{dest}")
        print("    想更新就进去 git pull；下面是配置片段（不变）")
        return _print_config(dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"[x] {dest} 已存在且非空，但不是 git 仓库。为避免覆盖你的东西，脚本停手——")
        print("    要么手动清空它，要么用 --dest 指定别处。")
        return 2

    url = REPO if args.direct else PROXY + REPO
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"] + (["--branch", args.branch] if args.branch else []) + [url, str(dest)]
    print(f"[*] 克隆：{url}")
    try:
        r = subprocess.run(cmd, check=False, timeout=900)
    except subprocess.TimeoutExpired:
        print("[x] 克隆超时（900s）。国内网络不稳是常态，重跑一次或换 --direct。")
        return 1
    if r.returncode != 0:
        print("[x] 克隆失败。常见原因：网络被墙 / 代理不可用。可试：")
        print("    · python scripts/setup_downloader.py --direct")
        print(f"    · 或自行 clone {REPO} 到 {dest}")
        return 1
    if not (dest / "run.py").is_file():
        print(f"[!] 克隆完成但没看到 run.py，请确认上游结构是否变了：{dest}")
    print(f"[✓] 下载层就位：{dest}")
    return _print_config(dest)


def _print_config(dest: Path) -> int:
    print("\n把下面两行填进 config.yaml 的 download 段（**文件里存着 key，脚本不替你改**）：")
    print("  download:")
    print(f'    downloader_dir: "{dest}"        # 含 run.py 的目录')
    print('    config_yml: ""                        # 留空即可（用下载器自带配置）')
    print("\n填完跑一次 dystill doctor 确认；本地文件（mp4/音频）不需要这一步就能蒸馏。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
