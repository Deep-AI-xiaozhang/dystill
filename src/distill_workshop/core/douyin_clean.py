"""抖音口水词清洗(F3 第一阶段)。

转写后、笔记生成前过滤无信息量的填充词,保留核心口播内容。
"""
from __future__ import annotations

import re

# 抖音高频口水词库(按类型分组)
_EMOTION_FILLERS = [
    "老铁们", "家人们", "宝子们", "兄弟们", "姐妹们",
    "给大家分享一下", "跟大家说一下", "和大家聊一聊",
    "这个真的是太", "真的是绝了", "你敢信", "我的天呐",
    "这也太", "简直不要太", "真的爱了", "直接爱了",
]

_TRAFFIC_FILLERS = [
    "点赞收藏", "点个赞", "点点关注", "关注我", "关注一下",
    "评论区告诉我", "评论区扣", "弹幕走起", "转发给",
    "下期接着说", "咱们下期见", "下集更精彩",
    "三连走起", "一键三连",
]

_NONSENSE_FILLERS = [
    "那么问题来了", "接下来重点来了", "划重点了",
    "废话不多说", "长话短说", "简单来说就是",
    "怎么说呢", "怎么讲呢", "咋说呢",
    "对不对", "是不是", "有没有",
    "你看啊", "你听我说", "听我说", "你看",
    "emmm", "嗯嗯", "啊这", "这个吧",
]

_REPEAT_FILLERS = [
    "对对对", "是是是", "好好好", "行行行",
    "就是就是", "真的真的", "非常非常",
    "这个这个", "那个那个", "然后然后",
]

# 合并全部口水词(按长度排序避免误删)
_ALL_FILLERS = sorted(
    _EMOTION_FILLERS + _TRAFFIC_FILLERS + _NONSENSE_FILLERS + _REPEAT_FILLERS,
    key=len,
    reverse=True
)

# 重复句式模式("就是就是就是" → "就是")
_REPEAT_PATTERN = re.compile(r"(.{1,4})\1{2,}")

# 保留的结构词(不清洗)
_KEEP_STRUCTURE = {
    "第一", "第二", "第三", "第四", "第五",
    "首先", "其次", "然后", "最后", "另外",
    "比如说", "举个例子", "具体来说",
}


def clean_douyin_transcript(transcript: str) -> str:
    """清洗抖音转写中的口水词和重复句式。

    Args:
        transcript: 原始转写文本(通常带时间戳分段)

    Returns:
        清洗后的转写文本(保留时间戳格式,移除空行)
    """
    if not transcript or not transcript.strip():
        return transcript

    lines = transcript.splitlines()
    cleaned_lines = []
    last_timestamp = None

    for line in lines:
        # 保留时间戳行(但只在有后续内容时才加入)
        if line.strip().startswith("[") and "]" in line[:10]:
            last_timestamp = line
            continue

        # 清洗内容行
        cleaned = _clean_line(line)
        if cleaned.strip():  # 清洗后还有内容才保留
            # 如果有待处理的时间戳,先加入时间戳再加内容
            if last_timestamp is not None:
                cleaned_lines.append(last_timestamp)
                last_timestamp = None
            cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines)


def _clean_line(text: str) -> str:
    """清洗单行文本。"""
    if not text.strip():
        return text

    # 1. 移除口水词(先清洗,再合并重复)
    for filler in _ALL_FILLERS:
        if filler in _KEEP_STRUCTURE:
            continue

        # 五种匹配场景(全覆盖):
        escaped = re.escape(filler)

        # 场景1:句首(后面可以是任意字符)
        text = re.sub(f"^{escaped}", "", text)
        # 场景2:句尾(前面可以是任意字符)
        text = re.sub(f"{escaped}$", "", text)
        # 场景3:前后都有标点或空格(支持中英文标点)
        text = re.sub(f"([，。、；：！？,.;:!?\\s]){escaped}([，。、；：！？,.;:!?\\s])", r"\1\2", text)
        # 场景4:前面有标点,后面紧跟内容(如",老铁们今天")
        text = re.sub(f"([，。、；：！？,.;:!?\\s]){escaped}", r"\1", text)
        # 场景5:前面是内容,后面有标点(如"今天老铁们,")
        text = re.sub(f"{escaped}([，。、；：！？,.;:!?\\s])", r"\1", text)

    # 2. 合并重复句式(在口水词清洗后做,避免误删有意义重复)
    text = _REPEAT_PATTERN.sub(r"\1", text)

    # 3. 清理多余空格和标点
    text = re.sub(r"\s{2,}", " ", text)  # 多空格→单空格
    text = re.sub(r"[，。、,]{2,}", "，", text)  # 多标点→单标点（支持中英文逗号）
    text = text.strip("，。、, ")

    # 如果清洗后只剩标点/空格,返回空串
    if not re.search(r"[\u4e00-\u9fa5a-zA-Z0-9]", text):
        return ""

    return text


def stats_cleaning(original: str, cleaned: str) -> dict[str, int]:
    """统计清洗效果(用于测试和调试)。"""
    return {
        "original_chars": len(original),
        "cleaned_chars": len(cleaned),
        "removed_chars": len(original) - len(cleaned),
        "removal_rate": round((len(original) - len(cleaned)) / len(original) * 100, 1)
                        if original else 0.0,
        "original_lines": len([ln for ln in original.splitlines() if ln.strip()]),
        "cleaned_lines": len([ln for ln in cleaned.splitlines() if ln.strip()]),
    }
