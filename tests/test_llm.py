"""供应商降级链测试（F0 最要命的一条纪律，此前 0 覆盖）。

来历：实测事故 2026-09-02 商汤空壳事故——HTTP 200 但 content 为空被当成功，
整篇笔记成了空壳。F0 把它写成铁律：「空返回当失败 + 按顺序降级」。
这条规则一破，后面所有"绝不空手"的承诺全部落空，所以要锁死。
"""
import pytest
import requests

from distill_workshop.config import ProviderConfig
from distill_workshop.providers import llm
from distill_workshop.providers.llm import EmptyContentError, LLMError, chat


class _Resp:
    def __init__(self, content, status=200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _Calls:
    """记录每次调用打到哪个 provider；_chat_once 的契约是「返回 str 或抛异常」。"""

    def __init__(self, plan):
        self.plan = list(plan)
        self.hits = []

    def __call__(self, provider, messages, temperature, timeout, max_tokens=None):
        self.hits.append(provider.name)
        item = self.plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _p(name, attempts=3):
    return ProviderConfig(name=name, api_base=f"http://{name}.local/v1",
                          api_key="k", model="m", max_attempts=attempts)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """降级退避是真的会 sleep（3/6/12s）——测试里必须掐掉，否则一套跑几分钟。"""
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)


def test_fallback_to_next_provider_on_error(monkeypatch):
    calls = _Calls([requests.HTTPError("429"), "正常内容"])
    monkeypatch.setattr(llm, "_chat_once", calls)
    r = chat([{"role": "user", "content": "x"}], [_p("main", attempts=1), _p("backup")], timeout=1)
    assert r.content == "正常内容" and r.provider == "backup"
    assert calls.hits == ["main", "backup"]


def test_empty_content_error_triggers_fallback(monkeypatch):
    """HTTP 200 但 content 为空要当失败换下家——这条是商汤空壳事故的根因，重点锁。"""
    calls = _Calls([EmptyContentError("main 返回空 content"), "真内容"])
    monkeypatch.setattr(llm, "_chat_once", calls)
    r = chat([{"role": "user", "content": "x"}], [_p("main", attempts=1), _p("backup")], timeout=1)
    assert r.content == "真内容" and r.provider == "backup"


def test_chat_once_raises_on_whitespace_content(monkeypatch):
    """空返回的判定真身在这里：只有空白字符也算空，不许 strip 后当成功放行。"""
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp("   "))
    with pytest.raises(EmptyContentError):
        llm._chat_once(_p("main"), [{"role": "user", "content": "x"}], 0.0, 5)


def test_all_providers_fail_lists_every_error(monkeypatch):
    calls = _Calls([requests.HTTPError("boom"), TimeoutError("timeout")])
    monkeypatch.setattr(llm, "_chat_once", calls)
    with pytest.raises(LLMError) as e:
        chat([{"role": "user", "content": "x"}], [_p("a"), _p("b")], timeout=1)
    msg = str(e.value)
    assert "a#1" in msg and "b#1" in msg and "boom" in msg


def test_max_attempts_respected_before_switching(monkeypatch):
    """降级是换墙撞，不是同墙撞更久：a 试满 max_attempts 次才轮到 b。"""
    calls = _Calls([requests.HTTPError("x")] * 2 + ["ok"])
    monkeypatch.setattr(llm, "_chat_once", calls)
    r = chat([{"role": "user", "content": "x"}], [_p("a", attempts=2), _p("b")], timeout=1)
    assert calls.hits == ["a", "a", "b"] and r.provider == "b"


def test_no_provider_raises_clear_error():
    with pytest.raises(LLMError) as e:
        chat([{"role": "user", "content": "x"}], [])
    assert "DISTILL_API_KEY" in str(e.value)     # 报错要直接告诉人怎么补


def test_empty_content_error_is_llm_error():
    assert issubclass(EmptyContentError, LLMError)


def test_chat_once_parses_openai_shape(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp("hello"))
    out = llm._chat_once(_p("main"), [{"role": "user", "content": "x"}], 0.0, 5)
    assert out == "hello"


# ── 确定性失败不重试（D17 加的 _NON_RETRYABLE_STATUS，当时零覆盖）────────
def _http_error(status):
    """带 .response 的 HTTPError（真实 requests 抛出来的形态）。"""
    e = requests.HTTPError(f"{status} Client Error")
    e.response = type("_R", (), {"status_code": status})()
    return e


def test_non_retryable_status_switches_provider_without_backoff(monkeypatch):
    """401/403/404 = key 无效/无权限或 api_base 写错，同家重试纯属白等（退避 3/6/12s）
    还把真实死因淹在降级链里——必须一次就换下家，且不退避。"""
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))   # 覆盖 autouse 的 no-sleep
    calls = _Calls([_http_error(403), "正常内容"])
    monkeypatch.setattr(llm, "_chat_once", calls)
    r = chat([{"role": "user", "content": "x"}], [_p("main", attempts=3), _p("backup")], timeout=1)
    assert r.provider == "backup"
    assert calls.hits == ["main", "backup"], "确定性失败不许在同家重试"
    assert slept == [], "确定性失败不许退避等待"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_each_non_retryable_status_reports_reason_and_gives_up_early(monkeypatch, status):
    calls = _Calls([_http_error(status)])
    monkeypatch.setattr(llm, "_chat_once", calls)
    with pytest.raises(LLMError) as e:
        chat([{"role": "user", "content": "x"}], [_p("only", attempts=3)], timeout=1)
    msg = str(e.value)
    assert f"HTTP {status}" in msg and "不再重试这家" in msg      # 报错要能看出怎么修
    assert calls.hits == ["only"], "attempts=3 也只能打一次"


def test_http_error_without_response_keeps_retrying(monkeypatch):
    """HTTPError 也可能没有 .response（旧测试的构造形态）——不能因此炸掉，
    要按普通错误退避重试（空返回/超时那套降级策略不变）。"""
    calls = _Calls([requests.HTTPError("boom"), requests.HTTPError("boom"), "正常内容"])
    monkeypatch.setattr(llm, "_chat_once", calls)
    r = chat([{"role": "user", "content": "x"}], [_p("main", attempts=2), _p("backup")], timeout=1)
    assert r.provider == "backup"
    assert calls.hits == ["main", "main", "backup"]


def test_extra_body_is_passed_through(monkeypatch):
    """供应商专属参数（如思考档位 reasoning_effort）必须原样进请求体。"""
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Resp("ok")

    monkeypatch.setattr(llm.requests, "post", fake_post)
    p = ProviderConfig(name="t", api_base="https://x/v1", api_key="k", model="m",
                       extra_body={"reasoning_effort": "none"})
    chat([{"role": "user", "content": "hi"}], [p])
    assert captured["reasoning_effort"] == "none"
    assert captured["model"] == "m" and captured["stream"] is False


def test_extra_body_absent_by_default(monkeypatch):
    """没配 extra_body 时，请求体里不该冒出多余键。"""
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Resp("ok")

    monkeypatch.setattr(llm.requests, "post", fake_post)
    p = ProviderConfig(name="t", api_base="https://x/v1", api_key="k", model="m")
    chat([{"role": "user", "content": "hi"}], [p])
    assert "reasoning_effort" not in captured
