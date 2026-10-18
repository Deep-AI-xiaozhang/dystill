"""身份层：aweme_id / 内容指纹是唯一身份；编号（v<N>）只是显示名，永不参与逻辑。

来源：实测事故 v123 重复、v171 假记录、v226→v230 编号漂移等事故的根因收敛。
规则：
1. 去重、断点续跑、台账、核对报告关联，一律用 record_key（aweme:<id> 或 fp:<hash>）。
2. 编号仅在"给人看 / 文件命名"时分配，允许空洞，允许重复出现于显示层。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

_AWEME_PATTERNS = (
    re.compile(r"/video/(\d{6,})"),
    re.compile(r"/note/(\d{6,})"),
    re.compile(r"[?&]aweme_id=(\d{6,})"),
    re.compile(r"[?&]item_ids=(\d{6,})"),
)


def extract_aweme_id(url: str) -> str | None:
    """从抖音链接中提取 aweme_id；短链（v.douyin.com）需在下载层展开后重跑本函数。"""
    for pat in _AWEME_PATTERNS:
        m = pat.search(url or "")
        if m:
            return m.group(1)
    return None


def content_fingerprint(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """本地文件内容指纹（sha256，流式）。用于本地视频/音频/图文等无 aweme_id 的素材身份。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def record_key(aweme_id: str | None = None, fingerprint: str | None = None) -> str:
    """唯一身份键。优先 aweme:<id>；无平台 id 时退回内容指纹 fp:<hash>。"""
    if aweme_id:
        return f"aweme:{aweme_id}"
    if fingerprint:
        return f"fp:{fingerprint}"
    raise ValueError("record_key 需要 aweme_id 或 fingerprint 之一")
