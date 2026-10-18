"""VLM 供应商层测试：锁定「一帧一次调用 + 单帧失败隔离」（DECISIONS D7）。

这一层的设计是被实测逼出来的：新管线一度把 48 帧塞进一次请求，后果是
输出被 max_tokens 截断、块间信息互相稀释、一帧失败整条画面维度全丢。
老工具（vision_tool.py + distill.step_vision）是逐帧调用 + 帧级缓存，
实测跑过 200+ 条视频。这些用例防止它被"优化"回去。
"""
from __future__ import annotations

import pytest
from PIL import Image

from distill_workshop.config import AppConfig, ProviderConfig
from distill_workshop.core.check.evidence_retriever import parse_vlm
from distill_workshop.providers import vision as vlm
from distill_workshop.providers.llm import ChatResult


def _frames(tmp_path, n=3):
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    out = []
    for i in range(1, n + 1):
        p = d / f"{i:03d}.jpg"
        Image.new("RGB", (64, 48), (i * 30, 0, 0)).save(p)
        out.append(p)
    return out


class _Spy:
    """记录每次调用带了几张图，并返回预设文本。"""

    def __init__(self, fail_on: set[int] | None = None):
        self.calls: list[int] = []
        self.fail_on = fail_on or set()

    def __call__(self, messages, providers, **kw):
        content = messages[0]["content"]
        n_imgs = sum(1 for c in content if c.get("type") == "image_url")
        self.calls.append(n_imgs)
        if len(self.calls) in self.fail_on:
            raise RuntimeError("429 too many requests")
        return ChatResult(content=f"画面文字 {len(self.calls)}（{n_imgs} 张）",
                          provider="p", model="m")


def test_default_sends_one_frame_per_call(tmp_path, monkeypatch):
    """默认逐帧：3 帧 = 3 次调用，每次只带 1 张图（老工具验证过的形态）。"""
    spy = _Spy()
    monkeypatch.setattr(vlm, "chat", spy)
    frames = _frames(tmp_path, 3)
    r = vlm.describe_images(frames, "prompt", [ProviderConfig(name="p", api_base="u", api_key="k", model="m")])
    assert spy.calls == [1, 1, 1]
    assert r.content.count("第 ") == 3


def test_marker_maps_back_to_frame_name(tmp_path, monkeypatch):
    """帧标记必须带序号+文件名，且能被证据层的画面解析认出来（否则时间戳映射不到帧）。"""
    monkeypatch.setattr(vlm, "chat", _Spy())
    frames = _frames(tmp_path, 2)
    r = vlm.describe_images(frames, "prompt", [ProviderConfig(name="p", api_base="u", api_key="k", model="m")])
    assert "第 1 张图（001.jpg）" in r.content
    assert "第 2 张图（002.jpg）" in r.content

    segs = parse_vlm(r.content)
    assert [s.frame for s in segs] == [1, 2]          # 序号被证据层正确解析


def test_one_frame_failure_does_not_kill_the_rest(tmp_path, monkeypatch):
    """单帧失败只标记该帧：整条画面维度不能因为一帧被限流就全丢。"""
    spy = _Spy(fail_on={1})
    monkeypatch.setattr(vlm, "chat", spy)
    frames = _frames(tmp_path, 3)
    r = vlm.describe_images(frames, "prompt", [ProviderConfig(name="p", api_base="u", api_key="k", model="m")])
    assert "画面理解失败" in r.content
    assert "画面文字 2" in r.content and "画面文字 3" in r.content
    assert len(spy.calls) == 3


def test_all_frames_failed_raises(tmp_path, monkeypatch):
    """全帧失败必须抛错——否则写出一份空 vlm.txt，看起来像"这个视频没有画面"。"""
    monkeypatch.setattr(vlm, "chat", _Spy(fail_on={1, 2}))
    frames = _frames(tmp_path, 2)
    with pytest.raises(RuntimeError) as e:
        vlm.describe_images(frames, "prompt", [ProviderConfig(name="p", api_base="u", api_key="k", model="m")])
    assert "全部失败" in str(e.value)


def test_batch_size_groups_frames(tmp_path, monkeypatch):
    """显式调大 batch 时按批发（速度换质量的取舍，标记退化为首尾区间）。"""
    spy = _Spy()
    monkeypatch.setattr(vlm, "chat", spy)
    frames = _frames(tmp_path, 4)
    r = vlm.describe_images(frames, "prompt", [ProviderConfig(name="p", api_base="u", api_key="k", model="m")],
                            batch_size=2)
    assert spy.calls == [2, 2]
    assert "第 1-2 张图" in r.content and "第 3-4 张图" in r.content


def test_empty_frames_raises_with_reason():
    with pytest.raises(ValueError) as e:
        vlm.describe_images([], "prompt", [])
    assert "抽帧" in str(e.value)


def test_no_vision_provider_raises():
    with pytest.raises(RuntimeError) as e:
        vlm.describe_images([__file__], "prompt", [])
    assert "能读图的模型" in str(e.value)


def test_provider_max_tokens_is_applied(tmp_path, monkeypatch):
    """provider 自己的 max_tokens 上限要自动生效（glm-4v-flash 硬上限 1024 就是这么踩到的）。"""
    from distill_workshop.providers import llm

    seen = {}

    def fake_once(provider, messages, temperature, timeout, max_tokens=None):
        seen[provider.name] = max_tokens
        return "ok"

    monkeypatch.setattr(llm, "_chat_once", fake_once)
    ps = [ProviderConfig(name="zhipu", api_base="u", api_key="k", model="m", max_tokens=1024),
          ProviderConfig(name="other", api_base="u", api_key="k", model="m")]
    llm.chat([{"role": "user", "content": "x"}], ps)
    assert seen == {"zhipu": 1024}          # 第一家就把上限带上，且成功后不再打第二家
    seen.clear()
    llm.chat([{"role": "user", "content": "x"}], ps, max_tokens=64)
    assert seen == {"zhipu": 64}            # 调用方显式指定时覆盖 provider 默认


def test_pipeline_vlm_uses_one_frame_prompt_and_config(tmp_path, monkeypatch):
    """配置里的 vlm_batch_frames / workers 真的被读走，且默认走单帧提问。"""
    from distill_workshop.core import pipeline
    seen = {}

    def fake_describe(paths, prompt, providers, **kw):
        seen["n"] = len(paths)
        seen["prompt"] = prompt
        seen["kw"] = kw
        return ChatResult(content="x", provider="p", model="m")

    monkeypatch.setattr("distill_workshop.providers.vision.describe_images", fake_describe)
    app = AppConfig(root=tmp_path, notes_dir=tmp_path, data_dir=tmp_path, cache_dir=tmp_path,
                    providers=[ProviderConfig(name="p", api_base="u", api_key="k", model="m")],
                    pipeline={"vlm_batch_frames": 1, "workers": 3})
    frames = _frames(tmp_path, 2)
    pipeline._default_vlm(frames, app, lambda m: None)
    assert seen["n"] == 2 and seen["kw"]["batch_size"] == 1 and seen["kw"]["workers"] == 3
    assert "一帧" in seen["prompt"]
