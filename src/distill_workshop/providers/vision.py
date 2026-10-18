"""VLM 多模态调用：图片 → base64 data URL，复用 llm.chat 的降级纪律。

**一帧一次调用，不是把全部帧塞进一次请求**（2026-09-17 实测确立，见 DECISIONS D7）：
老工具（vision_tool.py + distill.step_vision）就是这么干的，实测跑过 200+ 条视频；
新管线一度把 48 帧塞进一个请求，三个后果都实测到了：
  - 输出被单次 max_tokens 截断（glm-4v-flash 硬上限 1024 → 多图×3 就撞顶）
  - 块与块之间信息互相稀释（同一模型多图时，能复现的画面文字反而比单图少）
  - **一次失败 = 整条画面维度全丢**，而逐帧只丢那一帧

关键纪律继承：HTTP 200 空 content = 失败（商汤空壳 6 篇事故），
单帧多次重试仍失败只标记该帧，绝不编造画面描述。
"""
from __future__ import annotations

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import ProviderConfig
from .llm import ChatResult, chat

_MAX_SIDE = 1024

# 单帧提问：让模型知道这是一帧、不是整段视频，避免它去"总结全片"
PROMPT_SINGLE = (
    "这是视频里的一帧画面。请提取：画面中出现的文字（尽量完整转录）、"
    "人物在做什么、界面/场景是什么。只描述看到的，不要推测没有的内容。"
)

# 多帧提问（batch>1 时用；单帧提问里的"一帧"会让模型只描述第一张）
PROMPT_MULTI = (
    "这是同一个视频按场景切出的关键帧（按时间顺序）。请逐张提取：画面中出现的文字、"
    "人物在做什么、关键信息。只描述看到的，不要推测没有的内容。"
)

# 图文帖：图片就是全部内容（没有语音轨），必须说明白——否则模型会当它只是配图，
# 把图片里的信息当成"次要素材"而漏掉（图文帖的正文本来就全在图上）。
PROMPT_GALLERY = (
    "这是同一条图文帖里的图片（按顺序排列）。**图片本身就是这条内容的全部素材，没有语音**。"
    "请逐张提取：图中出现的文字（尽量完整转录，正文往往就是图上的字）、画面内容、"
    "以及图片传达的关键信息。只描述看到的，不要推测没有的内容。"
)


def _image_data_url(path: Path, quality: int = 70) -> str:
    """图片压缩到 max_side 内转 base64（控制 token，画面信息不丢）。"""
    from io import BytesIO

    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.size) > _MAX_SIDE:
        r = _MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _chunks(seq: list, n: int) -> list[list]:
    n = max(1, n)
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _marker(lo: int, hi: int, names: list[str]) -> str:
    """帧标记：`第 N 张图` 是老工具沿用的锚点，证据层靠它把画面片段映射回时间戳。

    单帧带文件名（回原片核对时直接找得到）；多帧只标首尾（块内确实无法逐帧切分，
    宁可标粗也不假装精确）。
    """
    if lo == hi:
        return f"--- 第 {lo} 张图（{names[0]}） ---"
    return f"--- 第 {lo}-{hi} 张图 ---"


def describe_images(image_paths: list[Path], prompt: str,
                    providers: list[ProviderConfig], timeout: int = 180,
                    *, batch_size: int = 1, workers: int = 2,
                    max_tokens: int | None = None, on_step=None) -> ChatResult:
    """逐帧（或按 batch_size 小批）理解画面，返回拼好的 ChatResult。

    batch_size=1 是默认，也是老工具验证过的形态；调大只换来"少几次请求"，
    代价是截断/稀释/一损俱损（理由见模块 docstring）。
    某一帧全供应商都失败 → 只标记这一帧，其余帧照常产出；全帧皆失败才抛错。
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("没有抽帧可理解（画面维度无法建立）")
    if not providers:
        raise RuntimeError("画面理解需要能读图的模型供应商：运行 dystill doctor 按提示补配置")

    groups = _chunks(paths, batch_size)
    results: list[str | None] = [None] * len(groups)
    used: dict[str, str] = {}
    failures: list[str] = []
    lock = threading.Lock()
    done = 0

    def work(gi: int) -> None:
        nonlocal done
        g = groups[gi]
        lo = gi * max(1, batch_size) + 1
        hi = lo + len(g) - 1
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in g:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(p)}})
        try:
            r = chat([{"role": "user", "content": content}], providers,
                     timeout=timeout, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 —— 单帧失败不该拖垮整条画面维度
            with lock:
                failures.append(f"第 {lo}{'' if lo == hi else f'-{hi}'} 张图：{type(e).__name__}: {e}")
                results[gi] = f"（画面理解失败：{type(e).__name__}）"
                done += 1
                n = done
            if on_step:
                on_step(f"   [!] 第 {lo} 张图理解失败（{type(e).__name__}），继续其余帧 [{n}/{len(groups)}]")
            return
        with lock:
            used[r.provider] = r.model
            results[gi] = _marker(lo, hi, [p.name for p in g]) + "\n" + r.content.strip()
            done += 1
            n = done
        if on_step and len(groups) > 1:
            on_step(f"   画面理解 [{n}/{len(groups)}]")

    if workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as ex:
            list(ex.map(work, range(len(groups))))
    else:
        for gi in range(len(groups)):
            work(gi)

    if len(failures) == len(groups):
        raise RuntimeError("画面理解全部失败（{} 帧）：{}".format(len(paths), " | ".join(failures[:3])))

    parts = [txt or "" for txt in results]
    names = ",".join(used) or ",".join(p.name for p in providers)
    models = ",".join(sorted(set(used.values()))) or providers[0].model
    return ChatResult(content="\n\n".join(parts), provider=names, model=models)
