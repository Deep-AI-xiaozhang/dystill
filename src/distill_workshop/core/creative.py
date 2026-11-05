"""自媒体衍生创作引擎：基于无幻觉事实笔记，一键衍生小红书/微头条/自媒体口播脚本。

纪律：
1. 严禁凭空编造事实：所有衍生文案的核心论点、数据、步骤必须严格源自蒸馏笔记本身；
2. 针对不同社媒受众定制语言风格：
   - xiaohongshu: 视觉化标题、Emoji点缀、分段呼吸感、核心看板前置、末尾带标签组；
   - weitoutiao: 强冲突/强认知反差标题、故事起承转合、接地气、引发评论区讨论；
   - script: 去除原作者痕迹的个人口播文案，带【镜头语言/画面提示】与自然断句。
"""
from __future__ import annotations

from ..config import AppConfig
from ..providers.llm import chat

PROMPT_XIAOHONGSHU = """你是一个顶流小红书爆款博主。请根据下面这份经过严格核对的【视频真实知识手记】，将其改编成一篇高点赞、高收藏的小红书图文文案。

【小红书风格要求】：
1. 标题：提供 3 个极具吸引力、带 Emoji 的爆款主副标题（反常识、拿结果、利他感强）。
2. 开头：用 1-2 句话直接戳中痛点或揭晓最有价值的答案（拒绝废话客套）。
3. 正文：结构化提炼，多用 Emoji 做小标，每个要点精炼干脆，直接给出可执行的步骤或数据参数。
4. 结尾：给出一句强行动口号或引导讨论。
5. 标签组：末尾附带 5-8 个精准热门话题标签（如 #知识薯 #干货分享 等）。

【真实知识手记内容】：
{note}
"""

PROMPT_WEITOUTIAO = """你是一个今日头条百万爆款创作者。请根据下面这份经过严格核对的【视频真实知识手记】，将其改写为一篇引人深思、高互动率的微头条文章。

【微头条风格要求】：
1. 标题：语不惊人死不休，具备强认知反差，制造悬念或直击普通人现实痛点。
2. 结构：经典的“黄金前三行”抓人，中间层层剖析，用大白话讲透底层逻辑，节奏紧凑。
3. 结尾：抛出启发性思考或争议性问题，刺激读者在评论区互动留言。
4. 字数：控制在 400-600 字之间，短句多，无AI腔。

【真实知识手记内容】：
{note}
"""

PROMPT_SCRIPT = """你是一个专业的短视频口播内容教练。请根据下面这份经过严格核对的【视频真实知识手记】，将其改编为一份适合创作者自己对着镜头拍摄的【个人原创短视频口播逐字稿】。

【口播脚本要求】：
1. 彻底去除原视频作者的个人身份、昵称与痕迹，变成使用者的第一人称原创视角。
2. 标注【镜头/画面提示】与【语气提示】，让拍摄者拿到就能直接读。
3. 黄金 3 秒黄金开头：第一句话就留住观众。
4. 中间把核心方法、参数和避坑要点讲透，口语化表达，不拗口。
5. 尾部引导点赞关注并提供行动指引。

【真实知识手记内容】：
{note}
"""

STYLES = {
    "xiaohongshu": ("小红书爆款", PROMPT_XIAOHONGSHU),
    "weitoutiao": ("今日头条/知乎", PROMPT_WEITOUTIAO),
    "script": ("短视频口播脚本", PROMPT_SCRIPT),
}


def rewrite_note(note_text: str, style_key: str, app: AppConfig) -> str:
    """根据选择的风格生成自媒体文案."""
    if style_key not in STYLES:
        raise ValueError(f"未知风格: {style_key}，可选: {list(STYLES.keys())}")

    _, prompt_tmpl = STYLES[style_key]
    prompt = prompt_tmpl.format(note=note_text[:6000])

    providers = app.ready_providers
    if not providers:
        raise RuntimeError("未配置模型供应商")

    # 二创文案一般在 500-1000 字以内，控制 max_tokens=1500 避免被大模型过度思考或长上下文拖慢；
    # timeout 设为 60s
    res = chat([{"role": "user", "content": prompt}], providers, max_tokens=1500, temperature=0.6, timeout=60)
    return res.content.strip()
