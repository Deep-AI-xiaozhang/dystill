"""笔记生成（移植自早期实现的笔记层，受众改为可配置）。

模板取向：只做**纯知识蒸馏**（结论 / 口播要点 / 画面要点 / 知识要点 / 适用场景），
不分析营销套路——禁止出现变现、引流、转化链路、内容公式这类栏目。

保留的实测资产：
- 类型判定（实操教程 / 观点认知）+ 单模板，两种类型同一结构，缺项写"无"
- 长转写分块提炼再汇总（16-20min 转写直接喂会超输出预算切半截）
- 初稿自检 + 定向重试：模板照抄 / 占位符残留 / 正文过短 / 结构跑偏 / 截断检测
  （「占位符即答案」：弱模型会把模板里的示例当答案照抄——模板只放"要填什么"）
- 无可用供应商时降级为原始素材草稿，绝不空手
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..config import AppConfig
from ..providers.llm import chat
from .artifacts import read_if_valid, write_if_valid

TRANSCRIPT_CHUNK_CHARS = 4000
TRANSCRIPT_CHUNK_MAX = 8

_CHUNK_PROMPT = ("这是某视频语音转写的一段（共 {total} 段）。请提炼这一段的核心要点：保留具体方法、"
                 "数字、步骤、指令话术，不要遗漏任何信息，不加入你的联想。用简洁条目输出。\n\n"
                 "【转写段落】\n{chunk}")

_PLACEHOLDER_KW = ("禁止写蒸馏工具链", "<从视频", "<一句话概括", "如：头条写作")
# 模板占位符残留的通用特征：`类型：<...>` / `主题：<...>` 这种"冒号后紧跟尖括号"。
# 实测（2026-10-13）模型会把尖括号原样留在成品里，而原来的关键词表漏检（主题内容被改写后
# 就不再含"<一句话概括"）。这条按位置特征抓，抓不到正文里正常的 < > 用法。
_PLACEHOLDER_POS_RE = re.compile(r"(类型|主题)[：:]\s*<")
_POLLUTE_RE = re.compile(
    r"[｜|]\s*主题：\s*(douyin-downloader|faster-whisper|PySceneDetect|scenedetect|视觉理解|LLM 汇总|faster_whisper)")
_EXAMPLE_RE = re.compile(r"[｜|：:]\s*如：")
_RETRY_PROMPTS = (
    ("注意：上一版把蒸馏工具链当成了视频的\"主题\"，或把模板里的占位符原样抄进了笔记，"
    "属于照抄模板。请重新生成：「主题」一栏必须用一句话概括视频内容本身，禁止出现尖括号占位符和模板示例文字；正文要有信息量。"),
    "注意：上一版内容过于简短、是模板空话。请重新基于转写与画面理解，把视频里具体的步骤、数字、话术、工具、数据全部提炼出来。",
)


# 提示词纪律：不写内部决策号（如 D23）——模型看不懂编号，只会把它当成要输出的内容。
# 上面「禁补视频外背景常识」那句的来历：外部背景常识是 L3 假阳性的最大单一来源
# （预热测 215 条里 3 条），从笔记生成的**上游**断掉（DECISIONS D23）。
# 改动这段动词/范围前先看 D15：判定提示词纪律不许随手改，要定点回放验证方向。
def build_prompt(title_hint: str, transcript: str, vlm_text: str,
                 audience: str = "想从视频里学到有用知识的学习者") -> str:
    return f"""你是一个视频知识蒸馏助手。你要把这段视频里**有用的知识**提炼成一份干净的结构化蒸馏笔记，读者是"{audience}"。

【核心原则】只蒸馏知识本身：概念、事实、步骤、工具、数据、对比、注意事项。不要分析视频的营销套路，不要出现"变现""引流""转化链路""内容公式"这类栏目。内容宁长勿短：把视频里出现的具体数字、步骤、工具名、指令话术、参数、平台规则全部保留，不要概括成一句话。区分"听到的"和"看到的"：口播要点来自转写，画面要点来自画面理解，不要混写。视频里没有的信息不要编。**也不要补充视频未提及的背景知识**——历史年份、人物生平、行业常识、"这其实是XX"式的对照说明，哪怕你确定是对的也不写：它们看似正确，但视频里查无出处，核对环节只能记为不一致或存疑。

【不要过度概括】这份笔记会被**逐句回原素材核对**，下列写法会被判"存疑"，请避免（都是实测踩到的）：
- **不加原话里没有的限定**：原素材没说的括号补充、修饰词、范围一律不写。例：原话只说"很难在一个窗口说清"，就不要写成"复杂内容（素材、代码）难以说清"。
- **不把不同位置的信息拼成一句推断**：宁可拆成两条，也不要把 A 处的现象和 B 处的功能合成一句结论或因果。例：原话只有"会先理解视频""合适加动画"，就不要写成"先理解内容再定制生成动画"。
- **不下"全程/所有/从不"这类覆盖性断言**：只看到部分素材时（画面往往只抽了几帧），不写"视频全程为口播""视频无人物出镜""全篇没有演示"。
- 概括允许换说法，但**不许加原素材里没有的判断**。

【不许把推测写成事实】作者在视频里**猜的、问的、说"可能"的**，笔记里必须保持"这是推测"的语气，不能写成确定结论（实测踩到：作者说"这有没有一种可能是更多的搜索流量，或者说是收藏"，笔记写成了"新评估维度包括 7 天后的搜索/收藏流量占比"——把疑问变成了确定事实，核对判不一致）。
- 原话有"可能/也许/大概/我猜/是不是/有没有一种可能是"这类词时，笔记里保留"作者推测""可能是"的措辞。
- 不要替作者把话说满：他问"是不是"，你就写"作者猜测是不是"，别写"就是"。

【画面字段照抄，不要漏、不要改写】画面理解里的**分类名、字段名、按钮名、进度数字**是硬证据，必须原样保留（实测踩到两类错）：
- **不要改写专有名词**：画面写"落库池·连接"，就不能写成"落库池·连板"；画面写"题材·问答"，就不能加字成"现题材·问答"。
- **不要漏掉画面上的具体字段**：画面列出"5板是空的"这类细节时，必须写进笔记，不能概括成"只有 6 板和 2 板"（漏掉 5 板就与证据矛盾）。
- 拿不准某个字/词时，**照抄画面里的写法**，不要按自己的理解"修正"它。

【按模板填】模板里的尖括号 `< >` 是**占位符标记**，填完必须删掉：类型写"实操教程"或"观点认知"，主题写一句实际概括——不要保留尖括号。**括号里的说明文字（如"核心，逐项提取…""按时间顺序…"）也是给你看的填写要求，不要写进笔记**——小节标题只保留标题本身。**日期一栏照抄模板里给定的那个日期，不要改写、不要自己编**。

【第一步·类型判定】二选一：
- **实操教程**：有具体步骤/流程/配置/数据/工具，可以照做
- **观点认知**：只有框架/观点/结论，无具体操作

【第二步·按类型选模板】两种类型都用【标准模板】；「观点认知」类型里"实操步骤"等确实没有的小项写"无"。禁止混入营销分析。

【标准模板】
# 蒸馏笔记 · {title_hint}

> 蒸馏日期：{time.strftime('%Y-%m-%d')} ｜ 类型：<实操教程/观点认知> ｜ 主题：<一句话概括视频主题；禁止写蒸馏工具链>

## 一、结论先行
3-5 句话概括这条视频讲了什么、最有价值的结论是什么。

## 二、视频内容
- 口播要点：按时间顺序，保留具体数字、步骤、工具名、参数
- 画面/操作要点

## 三、知识要点
- **关键概念**：视频里出现的名词/技术/功能，各自是什么意思
- **实操步骤**：可以直接照做的步骤/流程/配置
- **工具与资源**：提到的工具、模型、插件、价格、链接
- **对比与数据**：横向对比结论、数据基准、各自的优缺点
- **注意事项**：坑、限制、风险提示、前提条件

## 四、总结与适用场景
这些知识什么时候用得上、对谁有用，2-4 句话。

---
【语音转写】
{transcript}

【画面理解】
{vlm_text}
"""


def chunk_text(text: str, max_chars: int, cap: int) -> list[str]:
    """按行分块；块数超 cap 时按行均衡重切（限制 LLM 调用次数）。"""
    lines = text.splitlines()
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        if cur and cur_len + len(ln) > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    if len(chunks) > cap and len(chunks) > 1:
        per = (len(lines) + cap - 1) // cap
        chunks = ["\n".join(lines[i:i + per]) for i in range(0, len(lines), per)][:cap]
    return chunks


def check_note_quality(note: str) -> list[str]:
    """初稿自检：过短空话 / 模板照抄 / 占位符残留 / 示例残留 / 结构跑偏 / 截断。空列表=通过。"""
    problems = []
    low = 400  # 标准模板（纯知识蒸馏）正文下限
    if len(note) < low:
        problems.append(f"正文过短（{len(note)} 字），疑似模板空话")
    if _POLLUTE_RE.search(note):
        problems.append("「主题」字段写成了蒸馏工具链，疑似模板照抄")
    for kw in _PLACEHOLDER_KW:
        if kw in note:
            problems.append(f"模板占位符未替换（{kw}）")
    if _PLACEHOLDER_POS_RE.search(note):
        problems.append("模板占位符未替换（类型/主题一栏的尖括号没删掉）")
    if _EXAMPLE_RE.search(note):
        problems.append("疑似照抄模板示例（出现「如：」样例残留）")
    if "结论先行" not in note:
        problems.append("模板结构跑偏，缺「结论先行」板块")
    if "四、" not in note:
        problems.append("笔记疑似被截断（缺第四板块，只写到前三节）")
    tail = [ln.strip() for ln in note.splitlines() if ln.strip()]
    if tail and re.search(r"[-:：、,，;；\u2026]$", tail[-1]):
        problems.append(f"笔记疑似被截断（末行以「{tail[-1][-1]}」收尾）")
    return problems


def generate_note(transcript: str, vlm_text: str, title_hint: str, app: AppConfig,
                  on_step=None, chat_fn=chat) -> str:
    """生成笔记：长转写先分块提炼；初稿自检不通过带定向指令重试；全失败降级素材草稿。"""
    providers = app.ready_providers
    if not providers:
        return (f"# 蒸馏笔记（草稿）· {title_hint}\n\n"
                f"> 未配置模型供应商，以下为原始素材。运行 dystill doctor 补配置。\n\n"
                f"## 语音转写\n{transcript}\n\n## 画面理解\n{vlm_text}\n")

    def llm(prompt: str) -> str:
        return chat_fn([{"role": "user", "content": prompt}], providers, max_tokens=6000).content

    audience = app.pipeline.get("audience", "想从视频里学到有用知识的学习者")
    if not (transcript or "").strip():
        # 图文帖没有语音轨：要写明，否则模型会以为素材缺失而自己补内容
        transcript_part = "（本条是图文帖，没有语音内容，请完全依据图片理解来蒸馏）"
    elif len(transcript) > TRANSCRIPT_CHUNK_CHARS:
        chunks = chunk_text(transcript, TRANSCRIPT_CHUNK_CHARS, TRANSCRIPT_CHUNK_MAX)
        if on_step:
            on_step(f"   长转写分块摘要（{len(chunks)} 段）...")
        joined = "\n\n".join(f"【段 {i + 1}】\n{llm(_CHUNK_PROMPT.format(total=len(chunks), chunk=c))}"
                             for i, c in enumerate(chunks))
        transcript_part = f"【语音转写（分段提炼，共 {len(chunks)} 段）】\n{joined}"
    else:
        transcript_part = transcript

    prompt = build_prompt(title_hint, transcript_part, vlm_text, audience=audience)
    note, problems = None, []
    for attempt in range(len(_RETRY_PROMPTS) + 1):
        suffix = "" if attempt == 0 else "\n\n" + _RETRY_PROMPTS[attempt - 1]
        try:
            note = llm(prompt + suffix)
        except Exception as e:  # noqa: BLE001
            if on_step:
                on_step(f"   [!] LLM 汇总失败（{e}），降级为原始素材草稿")
            # 降级产物必须是素材本身，不能是 prompt：prompt 里的模板说明句
            # （"3-5 句话概括…"含数字）会被核对引擎 L1 当成视频声明去核，产出假报告
            return (f"# 蒸馏笔记（草稿）· {title_hint}\n\n"
                    f"> LLM 汇总失败，以下为原始素材。请检查供应商配置后重跑（dystill doctor）。\n\n"
                    f"## 语音转写\n{transcript}\n\n## 画面理解\n{vlm_text}\n")
        problems = check_note_quality(note)
        if not problems:
            return note
        if on_step:
            on_step(f"   初稿自检不通过：{'；'.join(problems)}（第 {attempt + 1} 版，将重试）")
    return note + "\n\n---\n> ⚠️ 初稿质量自检未通过（" + "；".join(problems) + "），需人工校对补深度\n"


def clean_title(stem: str) -> str:
    """从下载文件名提取主题：去抖音日期前缀/BV号前缀/尾部 aweme id（早期实现）。"""
    s = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)
    s = re.sub(r"^(BV[0-9A-Za-z]{8,12})_", "", s)
    s = re.sub(r"_\d{15,20}$", "", s)
    return s.strip()


def safe_title(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n]+', "", s).strip()
    return s[:30] or "未命名"


def default_note_fn(app: AppConfig, on_step=None):
    """流水线 note_fn 默认实现：读产物 → 生成笔记 → frontmatter 落盘。"""
    def fn(record_dir: Path, artifacts: dict, app_: AppConfig, log) -> None:
        rd = Path(record_dir)
        transcript = read_if_valid(rd / "transcript.txt", "transcript") or ""
        vlm = read_if_valid(rd / "vlm.txt", "vlm") or ""
        # 标题/身份来源：优先用 process_record 开头就放进 artifacts 的 media/key/source
        # （meta.json 由流程成功末尾才写，首次运行时读它拿不到，会退化成"视频"）；
        # artifacts 缺失时（直接调用/测试）回退读 meta.json。
        media = artifacts.get("media") or ""
        if not media:
            try:
                meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, TypeError):
                meta = {}
            media = meta.get("media") or ""
            artifacts = {**artifacts, "key": meta.get("key", ""), "source": meta.get("source", "")}
        title_hint = safe_title(clean_title(Path(media).stem))
        if on_step:
            on_step(f"   标题：{title_hint}")
        note = generate_note(transcript, vlm, title_hint, app_, on_step=log)
        fm = (f"---\ntitle: {title_hint}\nkey: {artifacts.get('key', '')}\n"
              f"source: {artifacts.get('source', '')}\ndate: {time.strftime('%Y-%m-%d')}\n---\n")
        write_if_valid(rd / "note.md", fm + note, "note")
    return fn
