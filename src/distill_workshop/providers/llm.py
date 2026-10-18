"""LLM 供应商抽象：OpenAI 兼容接口 + 顺序降级。

两条铁律（来自实测 2026-09-02 商汤空壳事故复盘）：
1. HTTP 200 但 content 为空 = 失败（EmptyContentError），必须降级，绝不当成功。
2. 每个 provider 最多试 max_attempts 次，试完换下家——"降级是换墙撞，不是同墙撞更久"。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import requests

from ..config import ProviderConfig


class LLMError(RuntimeError):
    """所有供应商都失败。"""


class EmptyContentError(LLMError):
    """HTTP 200 但 content 为空 —— 当失败处理。"""


@dataclass
class ChatResult:
    content: str
    provider: str
    model: str


def _chat_once(provider: ProviderConfig, messages: list[dict], temperature: float,
               timeout: int, max_tokens: int | None = None) -> str:
    url = provider.api_base.rstrip("/") + "/chat/completions"
    payload = {"model": provider.model, "messages": messages,
               "temperature": temperature, "stream": False}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if provider.extra_body:
        # 供应商专属参数原样透传（如思考档位 reasoning_effort）。
        # 放最后合并：显式配置优先，配了就能覆盖上面的默认字段。
        payload.update(provider.extra_body)
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {provider.api_key}", "Connection": "close"},
        json=payload,
        timeout=(15, timeout),  # (connect_timeout, read_timeout)
    )
    resp.raise_for_status()
    content = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise EmptyContentError(f"{provider.name} 返回空 content")
    return content


# 这些 HTTP 状态在同家重试没有意义：401/403 = key 无效或无权限，404 = api_base/模型名写错。
# 原地退避重试只会白等（3→6→12s）还把真实死因淹没在降级链里——直接换下家。
_NON_RETRYABLE_STATUS = frozenset({401, 403, 404})


def chat(messages: list[dict], providers: list[ProviderConfig],
         temperature: float = 0.3, timeout: int = 120,
         max_tokens: int | None = None) -> ChatResult:
    """按配置顺序调用供应商列表，任一成功即返回；全失败抛 LLMError（附各次错误）。"""
    if not providers:
        raise LLMError("没有可用的 provider：请配置 config.yaml 或设置 DISTILL_API_KEY/DISTILL_API_BASE/DISTILL_MODEL")
    errors: list[str] = []
    for p in providers:
        # 调用方没指定时，退到该 provider 自己的硬上限（免费模型常有 [1,1024] 这类限制）
        mt = max_tokens if max_tokens is not None else (p.max_tokens or None)
        for attempt in range(1, p.max_attempts + 1):
            try:
                content = _chat_once(p, messages, temperature, timeout, mt)
                return ChatResult(content=content, provider=p.name, model=p.model)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                errors.append(f"{p.name}#{attempt}: HTTP {status}: {e}")
                if status in _NON_RETRYABLE_STATUS:
                    errors[-1] += "（key 无效/无权限或 api_base 写错，不再重试这家）"
                    break   # 确定性失败：换下家，不原地退避
                if attempt < p.max_attempts:
                    time.sleep(min(3 * 2 ** (attempt - 1), 12) + random.uniform(0, 1))
            except Exception as e:  # noqa: BLE001 —— 统一计入降级日志
                wait = min(3 * 2 ** (attempt - 1), 12) + random.uniform(0, 1)
                errors.append(f"{p.name}#{attempt}: {type(e).__name__}: {e}")
                if attempt < p.max_attempts:
                    time.sleep(wait)
    raise LLMError("全部供应商失败：" + " | ".join(errors))
