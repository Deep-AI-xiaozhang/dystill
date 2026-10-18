"""L3 逐条判定（核对四层第三层）。

判定三值 + 两道硬闸，闸门的方向都是"宁存疑，不误判"：
1. **零证据不许判一致**（所有模式生效）：L2 一条证据都没回查到，就说明原素材里找不到出处，
   此时模型若给 ✅，那是模型在凭常识补——正是"瞎编"要防的东西，一律压回 ⚠️ 存疑。
2. **覆盖度阈值**（仅深度模式）：LLM 判 ✅ 但 L2 覆盖度低于阈值 → 降 ⚠️ 存疑（双信号的第二只眼睛）。

为什么不是"LLM 判完就完事"：核对工具最致命的失败模式是把没核到的当核过了（假阳性），
用户看到 ✅ 就不再回看原片，错误就此固化。所以闸门只可能把 ✅ 往下压，永远不会把 ⚠️ 抬成 ✅。
"""
from __future__ import annotations

import json
import re
import time

from .models import VERDICT_DOUBT, VERDICT_MISMATCH, VERDICT_OK, Claim
from .modes import CheckMode

# 模型漏返某条判定时的占位理由——带这个理由的条目会被"单独补问"一次，
# 因为这是机械失败（漏输出），不是判断结果，不该直接记成 ⚠️ 存疑白丢一条
MISSING_REASON = "模型未返回该条判定"

_VERDICT_ALIAS = {
    "ok": VERDICT_OK, "一致": VERDICT_OK, "支持": VERDICT_OK, "true": VERDICT_OK, "✅": VERDICT_OK,
    "mismatch": VERDICT_MISMATCH, "不一致": VERDICT_MISMATCH, "矛盾": VERDICT_MISMATCH,
    "false": VERDICT_MISMATCH, "❌": VERDICT_MISMATCH,
    "doubt": VERDICT_DOUBT, "存疑": VERDICT_DOUBT, "无法确认": VERDICT_DOUBT,
    "unknown": VERDICT_DOUBT, "⚠️": VERDICT_DOUBT, "": VERDICT_DOUBT,
}

# ── 证据预排序（2026-10-16 改进）──
# 优先级：钉住 > visual/opening > contrast > spoken > neutral
# 让模型先看到最关键的证据，减少被无关证据干扰

_EVI_TYPE_PRIORITY = {
    'visual': 1,     # 画面证据最可靠
    'opening': 1,    # 开头定位同样重要
    'contrast': 2,   # 对比结构需要两段都确认
    'spoken': 3,     # 口语表达允许措辞差异
    'neutral': 4,    # 普通证据
}


def _sort_evidence_by_priority(evidences: list) -> list:
    """按优先级排序证据：钉住 > 类型优先级 > fact_hit > score"""
    return sorted(
        evidences,
        key=lambda e: (
            not e.pinned,  # pinned=True 排在前面
            _EVI_TYPE_PRIORITY.get(e.evi_type or 'neutral', 4),
            -e.fact_hit,
            -e.score
        )
    )


_PROMPT = """你是视频笔记的事实核对员。任务：判断笔记里的每条声明，**在下面给出的原视频证据里有没有出处**。

判定只有三种：
- "ok"：证据支持这条声明（数字、名称、说法对得上）
- "mismatch"：**证据明确与声明矛盾**（数字不符、说法相反、评级/结论不符）
- "doubt"：证据不足以判断（只提到一半、指代不清、给到的片段里找不到对应内容）

判定纪律（每一条都是踩过坑才写上的，请严格照做）：
1. **你看到的只是回查到的片段，不是完整原素材**。片段里没找到 ≠ 原视频没有。
   这种情形一律判 "doubt"，**不要**判 "mismatch"——把"我没看到"说成"原视频没有"是本工具最严重的误判。
2. **转写是 ASR 产物，同音错字很常见**：英文产品名常被听写成读音相近的另一串字母
   （同一款产品在转写里出现几种拼写很正常），中文专有名词的同音字替换同理。
   名字看着不一样、但读音接近且语境一致时，按**一致**处理；
   只有当名字差异**没法用音近解释**（证据说的与声明说的明显是两个不同的东西）才算不符。
   ⚠️ 同理，**不要因为"证据里的名字和声明里的名字拼写不同"就判不符**——
   先判断它们是不是同一款产品的不同听写（读音接近 + 语境一致，基本就是同一个）。
3. **笔记的声明多是概括**，措辞和口播不一样是正常的。只要证据支持这个说法（换了说法也算），就判 "ok"；
   不要因为"没逐字出现"就判存疑。
4. **只依证据判，不许用你自己的常识补**：证据里完全没有的内容不能判 ok。
5. **不评价观点对错、不评价概括是否到位**，只判"原视频里有没有这么说过"。
6. 每条会给出「原素材中查不到的数字/专名」清单。**数字清单是强信号**（数字很少被听错）：
   清单里的数字可判 "mismatch" 或 "doubt"，但理由里必须写明这个数字在原素材里查不到；
   专名清单只作参考（ASR 会听错名字）。
7. **小心「推理性改写」——本工具最容易漏的一类假 ✅**：原视频说的是作者**自己**怎么用，
   笔记却写成**对外的服务/商业行为**（例：视频说"我用它做调研竞品、测软件这三件事"，
   笔记写成"可基于它提供竞品调研服务、软件测试服务"）。这类声明里的工具名、场景名
   都能在原素材里找到，看着完全"有出处"——**但错的是推理方向**。
   出现这种升格（提供服务/对外/客户/接单/变现等）时应判 "doubt"，
   reason 写"原素材只说了作者自己在用，没有对外提供这层意思"。
8. 判 "mismatch" 或 "doubt" 时给出 fix：按证据能支持的说法该怎么写；证据不足就写"建议回看原片确认"。
9. **看栏目名再理解声明**：每条声明都标了它在笔记里的栏目（如「测试任务」「问题与表现」「第11-13张」）。
   「测试任务」说的是"这次要测什么"，**不是在断言任务成功**；
   「问题与表现」说的是"这款工具的表现"，工具由它所在的小节决定。
   把栏目名算进去再判，不要只读半句话。
10. **多工具对比类视频尤其要小心证据归属**：一段话可能在讲 A 工具，而声明在讲 B 工具。
    如果给你的证据通篇在讲**另一款工具**，那它既不能证明也不能否证这条声明 → 判 "doubt"，
    并在 reason 里写明"给到的证据讲的是另一款工具"。
11. **画面文字优先于转写文本**：画面OCR提取的文字（vlm证据里的「画面文字转录」「画面中出现的文字」）
    比转写更可靠。当画面明确显示某个产品名、数字或参数时，优先以画面为准。
    转写常因ASR错误把专有名词听错（如"Qwen"听成"千问"或"北大"），但画面文字是视频作者故意放上去的，
    基本不会错。**如果画面说A、转写说B，优先相信画面**。
12. **准确追踪上下文主体**：证据里提到"缺点""优点""需要折腾""省心"等评价时，
    **必须判断这句话是在说谁**。常见坑：
    - 上一段在讲工具A的缺点，下一段开始讲工具B，但你把A的缺点当成B的。
    - "还有一个缺点"这句话前后可能换了主语，要看上下文。
    仔细看证据的时间戳和前后文：如果某段在讲A，下一段明确切换到B（如"接下来看B""那么B呢"），
    就不要把A的评价算到B头上。**拿不准时判"doubt"，不要瞎猜主体**。
13. **证据内部矛盾时优先判"doubt"**：如果给到的多条证据互相矛盾（例如不同时间点的画面显示不同价格、
    或者转写和画面说法不一致），**不要自己选边站判"mismatch"**。
    这种情况说明原视频本身前后不一致，或者某个证据来源有误，应判"doubt"并在reason里写明
    "证据内部矛盾（画面XX显示A，画面YY显示B），无法判定"。
14. **"mismatch" 必须是明确否定，不是"没明说"**：
    只有当证据**明确说了相反的事**才判 "mismatch"。常见误判：
    - 声明说"X是A"，证据没提A但也没说X不是A → 这是 "doubt"，不是 "mismatch"
    - 声明说"在1500以内无缺点"，证据提到"缺点"但没说是不是在1500以内 → "doubt"
    - 声明说"价格90"，证据显示80 → 这才是 "mismatch"（明确矛盾）
    **判 mismatch 前问自己：证据是"明确否定了声明"，还是"只是没确认声明"？**
    后者应判 "doubt"。宁可多判存疑，不要把"信息不足"当成"明确矛盾"。

只返回 JSON 数组，不要解释、不要 markdown 围栏：
[{{"i": 编号, "verdict": "ok|mismatch|doubt", "reason": "一句话依据", "fix": "修正建议或空串"}}]

待核对（每条含笔记声明 + 回查到的原视频证据 + 缺失硬事实）：

{items}
"""


def _fmt_evidence(c: Claim) -> str:
    if not c.evidence:
        return "    （无证据）"
    # 按优先级排序证据：钉住 > visual/opening > contrast > spoken > neutral
    # 让模型先看到最关键的证据（2026-10-16 改进）
    sorted_evi = _sort_evidence_by_priority(c.evidence)
    lines = []
    for e in sorted_evi:
        tag = "转写" if e.source == "transcript" else "画面"
        pos = f"{e.ts} " if e.ts else ""
        mark = "★含全部硬事实 " if e.fact_hit >= 1.0 else ""
        lines.append(f"    - [{tag}{pos}]{mark}{e.text[:400]}")
    return "\n".join(lines)


def _fmt_missing(c: Claim) -> str:
    bits = []
    if c.miss_numbers:
        bits.append(f"原素材中查不到的数字：{'、'.join(c.miss_numbers)}")
    if c.miss_names:
        bits.append(f"原素材中查不到的专名：{'、'.join(c.miss_names)}（转写可能听错，仅参考）")
    if c.miss_biz:
        bits.append(f"这条声明把它写成对外的服务/商业行为（{'、'.join(c.miss_biz)}），"
                    f"但整条原素材里没有这层意思 → 这是笔记的推论，应判 doubt")
    return "\n".join(f"    ! {b}" for b in bits)


def build_items(claims: list[Claim]) -> str:
    blocks = []
    for i, c in enumerate(claims, 1):
        missing = _fmt_missing(c)
        # 栏目名必须带上：`**测试任务**：找到下载目标栏并上传 pio 文件` 剥掉栏目名后，
        # 模型会把它读成"断言上传成功了"（实测假 ❌ 的元凶）
        head = f"{i}. 声明（笔记的「{c.context}」栏）" if c.context else f"{i}. 声明"
        blocks.append(f"{head}：{c.text}\n   证据：\n{_fmt_evidence(c)}"
                      + (f"\n{missing}" if missing else ""))
    return "\n\n".join(blocks)


def parse_verdicts(raw: str, n: int) -> dict[int, dict]:
    """解析 [{i,verdict,reason,fix}]。不可解析的条目直接忽略（上层按存疑兜底）。"""
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return {}
    try:
        rows = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    out: dict[int, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        if not (1 <= i <= n):
            continue
        raw_v = str(row.get("verdict") or "").strip().lower()
        out[i] = {
            "verdict": _VERDICT_ALIAS.get(raw_v, VERDICT_DOUBT),
            "reason": str(row.get("reason") or "").strip()[:200],
            "fix": str(row.get("fix") or "").strip()[:200],
        }
    return out


def apply_gates(c: Claim, mode: CheckMode) -> Claim:
    """四道闸（只会把 ok 往下压）：
    1. 零证据不许判一致
    2. 声明里的数字在原素材里查不到 → 不许判一致（编造数字是这类工具最危险的输出）
    3. 声明用了"对外提供/买卖"措辞而原素材里连这词都没有 → 不许判一致（推理性改写，见 D9）
    4. 深度模式再加覆盖度阈值
    """
    if c.verdict != VERDICT_OK:
        return c
    if not c.evidence:
        c.verdict = VERDICT_DOUBT
        c.reason = "原素材里没回查到出处，不足以判一致"
        c.fix = c.fix or "建议回看原片确认"
        return c
    if c.miss_numbers:
        c.verdict = VERDICT_DOUBT
        c.reason = f"声明里的数字 {'、'.join(c.miss_numbers)} 在原素材里查不到，不足以判一致"
        c.fix = c.fix or "建议回看原片确认数字"
        return c
    if c.miss_biz:
        c.verdict = VERDICT_DOUBT
        c.reason = (f"声明把它写成了对外的服务/商业行为（{ '、'.join(c.miss_biz) }），"
                    f"但整条原素材里没有这层意思，属于笔记自己的推论")
        c.fix = c.fix or "改回原素材的说法：这是作者自己在用，不是对外提供的服务"
        return c
    if mode.double_signal and c.score < mode.min_score:
        c.verdict = VERDICT_DOUBT
        c.reason = f"字面证据太弱（覆盖度 {c.score:.2f} < {mode.min_score}），深度模式判存疑"
        c.fix = c.fix or "建议回看原片确认"
    return c


def _one_batch(batch: list[Claim], mode: CheckMode, chat_fn):
    """跑一批判定。返回 (mapping, 是否抛错, provider, 末次错误摘要)。抛错与"模型返回空"必须区分：
    前者值得等一会儿重试（典型是 429/限流），后者重试也是白跑。错误摘要必须透传到日志——
    "疑似限流"和"余额不足"的处置完全不同，吞掉真实死因会把人引向错误排查方向（D21 实测）。"""
    import time
    start_time = time.time()
    prompt = _PROMPT.format(items=build_items(batch))
    last_err = ""
    try:
        r = chat_fn([{"role": "user", "content": prompt}])
        result = parse_verdicts(getattr(r, "content", r) or "", len(batch)), False, \
            getattr(r, "provider", "") or "", ""
        return result
    except Exception as e:  # noqa: BLE001 —— 判定失败一律落存疑，绝不当一致
        elapsed = time.time() - start_time
        # 失败批次耗时并进错误摘要：区分"快速 429"和"超时"靠它；先截异常正文，保证耗时不被 200 字符上限挤掉
        last_err = (f"{type(e).__name__}: {e}"[:160] + f"（{elapsed:.1f}s）")[:200]
        return {}, True, "", last_err


def verify_claims(claims: list[Claim], mode: CheckMode, chat_fn, on_step=None,
                  batch_delay: float = 0.0, retry_pause: float = 20.0) -> str:
    """分批判定，就地写回 claim.verdict/reason/fix。返回实际使用的供应商名（可为空）。

    限流兜底（2026-09-17 实测商汤 429）：某批抛错时先等 retry_pause 秒重试一次，
    仍失败才把该批记存疑。免费/低配额通道连续调用容易被掐，等一次比直接降级划算得多。
    """
    if not claims:
        return ""
    provider_used = ""
    batches = [claims[i:i + mode.batch_size] for i in range(0, len(claims), mode.batch_size)]
    for bi, batch in enumerate(batches, 1):
        if on_step:
            on_step(f"   开始第 {bi}/{len(batches)} 批（{len(batch)} 条）...")
        if bi > 1 and batch_delay > 0:
            time.sleep(batch_delay)
        mapping, raised, prov, err = _one_batch(batch, mode, chat_fn)
        provider_used = prov or provider_used
        if raised:                      # 抛错才重试（"模型返回空数组"重试也是白跑）
            if retry_pause > 0:
                if on_step:
                    on_step(f"   [!] 第 {bi} 批判定失败，{retry_pause:.0f}s 后重试一次（{err or '疑似限流'}）")
                time.sleep(retry_pause)
            mapping, raised, prov, err = _one_batch(batch, mode, chat_fn)
            provider_used = prov or provider_used
        if raised:
            for c in batch:
                c.failed = True          # 打标记：下次可 --retry-failed 只重判这几条
            if on_step:
                on_step(f"   [!] 第 {bi}/{len(batches)} 批判定仍失败 → 该批 {len(batch)} 条记为存疑"
                        f"（{err or '原因未知'}）")
        for i, c in enumerate(batch, 1):
            got = mapping.get(i)
            if got:
                c.verdict, c.reason, c.fix = got["verdict"], got["reason"], got["fix"]
            elif not c.verdict:
                c.verdict = VERDICT_DOUBT
                # 失败原因必须带真实错误原文：「余额不足」和「限流」处置完全不同，
                # 只写「限流/网络」会把人引向错误排查方向（D21 追记当时只打到日志，
                # 产物里丢掉了——2026-10-04 复查补上）。"调用失败"字样保留：
                # engine.prev_failed 靠它兼容没有 failed 字段的旧报告。
                c.reason = c.reason or (f"该批模型调用失败（{err or '限流/网络'}），未能判定" if raised
                                        else MISSING_REASON)
                c.fix = c.fix or "建议回看原片确认"
            apply_gates(c, mode)

    # 补问：模型偶尔会漏返某几条（实测 B016：理由写着"模型未返回该条判定"）。
    # 这是**机械失败，不是判断**——它会把一条本该 ✅ 的声明记成 ⚠️，白丢分。
    # 单独再问一次这几条，代价极小（只有那几条，不带整批）。
    missing = [c for c in claims
               if c.reason == MISSING_REASON and not c.failed]
    if missing and chat_fn is not None:
        if on_step:
            on_step(f"   [i] {len(missing)} 条模型漏返 → 分批补问")
        # **按正常批大小分组重问**，不把漏返的全塞进一次：实测（2026-10-14）
        # 7 条一次补问，补完仍有 8 条停在"模型未返回"——一批越大越容易再漏。
        for start in range(0, len(missing), mode.batch_size):
            group = missing[start:start + mode.batch_size]
            mapping, raised, prov, err = _one_batch(group, mode, chat_fn)
            provider_used = prov or provider_used
            if raised:
                # 补问也抛错：错误摘要不能吞（D21），标成「没判成」供 --retry-failed 恢复
                for c in group:
                    c.failed = True
                    c.reason = f"该批模型调用失败（{err or '限流/网络'}），未能判定"
                continue
            for i, c in enumerate(group, 1):
                got = mapping.get(i)
                if got and got["verdict"]:
                    c.verdict, c.reason, c.fix = got["verdict"], got["reason"], got["fix"]
                    apply_gates(c, mode)
        if on_step:
            # 只算**真正拿到判定**的：failed（调用失败）的不算，否则会把失败报成成功
            got_n = sum(1 for c in missing
                        if not c.failed and c.reason != MISSING_REASON)
            if got_n:
                on_step(f"   [i] 补问结果：{got_n}/{len(missing)} 条拿到判定")
            else:
                # 错误摘要不能吞（D21）：「限流」和「上游 500」的处置完全不同
                errs = "；".join(sorted({c.reason for c in missing if c.failed}))
                on_step(f"   [!] 补问仍未拿到判定（保持存疑）：{errs or '原因未知'}")
    # 兜底：任何仍未判定的声明一律存疑（绝不留在空值，空值会被下游当成没事）
    for c in claims:
        if not c.verdict:
            c.verdict, c.reason = VERDICT_DOUBT, "未取得判定"
            apply_gates(c, mode)
    return provider_used
