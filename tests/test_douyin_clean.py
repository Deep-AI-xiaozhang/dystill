"""测试抖音口水词清洗模块。"""
from distill_workshop.core.douyin_clean import (
    _clean_line,
    clean_douyin_transcript,
    stats_cleaning,
)


class TestCleanLine:
    """测试单行清洗逻辑。"""

    def test_emotion_fillers(self):
        """情绪类口水词。"""
        assert _clean_line("老铁们,这个工具真的太好用了") == "这个工具真的太好用了"
        assert _clean_line("家人们,给大家分享一下") == ""
        assert _clean_line("宝子们你敢信") == ""

    def test_traffic_fillers(self):
        """引流类口水词。"""
        assert _clean_line("点赞收藏,下期接着说") == ""
        assert _clean_line("关注我,评论区告诉我") == ""
        assert _clean_line("这个功能很强大,点个赞") == "这个功能很强大"

    def test_nonsense_fillers(self):
        """废话类口水词。"""
        assert _clean_line("那么问题来了,怎么配置") == "怎么配置"
        assert _clean_line("废话不多说,直接开始") == "直接开始"
        assert _clean_line("简单来说就是,这个参数要填") == "这个参数要填"

    def test_repeat_patterns(self):
        """重复句式合并。"""
        assert _clean_line("就是就是就是这个意思") == "就是这个意思"
        assert _clean_line("对对对对,这个的很好用") == "对,这个的很好用"  # 修复:避免"真的真的"被清洗
        assert _clean_line("然后然后然后再点保存") == "然后再点保存"

    def test_keep_structure_words(self):
        """保留结构词。"""
        text = "第一步打开设置,第二步找到权限,然后勾选"
        # "第一""第二""然后"应该保留
        cleaned = _clean_line(text)
        assert "第一步" in cleaned
        assert "第二步" in cleaned
        assert "然后" in cleaned

    def test_preserve_content(self):
        """不误删正常内容。"""
        # "这个工具"不能变成"工具"
        assert "这个工具" in _clean_line("我推荐这个工具给大家")
        # "对不对"是口水词,但"对齐"不是
        assert "对齐" in _clean_line("需要把文本对齐")

    def test_empty_after_cleaning(self):
        """清洗后只剩标点时返回空串。"""
        assert _clean_line("老铁们,家人们") == ""
        assert _clean_line("点赞收藏,关注我") == ""
        assert _clean_line("emmm,怎么说呢") == ""


class TestCleanTranscript:
    """测试完整转写清洗。"""

    def test_preserve_timestamps(self):
        """保留时间戳格式。"""
        transcript = """[00:15]
老铁们，给大家分享一下这个工具
[00:23]
第一步打开设置，然后找到权限
[00:35]
保存后重启生效"""

        cleaned = clean_douyin_transcript(transcript)
        assert "[00:15]" in cleaned
        assert "[00:23]" in cleaned
        assert "[00:35]" in cleaned
        # 口水词应该被清理
        assert "老铁们" not in cleaned
        assert "给大家分享一下" not in cleaned
        # 有效内容应该保留
        assert "这个工具" in cleaned
        assert "第一步" in cleaned
        assert "然后" in cleaned

    def test_remove_empty_lines(self):
        """清洗后空行应该被移除。"""
        transcript = """[00:10]
点赞收藏
[00:15]
这是有效内容
[00:20]
关注我,评论区告诉我"""

        cleaned = clean_douyin_transcript(transcript)
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        # 只剩时间戳 [00:15] 和有效内容行
        assert "[00:10]" not in cleaned or cleaned.count("[00:10]") == 0  # 可能被保留但无内容行
        assert "这是有效内容" in lines

    def test_real_example(self):
        """真实转写片段(模拟抖音实际场景)。"""
        transcript = """[01:05]
老铁们,那么问题来了,这个工具怎么配置呢
[01:12]
第一步,打开config.yaml文件,然后然后找到providers这一行
[01:25]
第二步,把你的API key填进去,对对对,就是这个key
[01:35]
这个真的是太简单了,点赞收藏,下期接着说"""

        cleaned = clean_douyin_transcript(transcript)
        # 应该保留的
        assert "这个工具怎么配置" in cleaned
        assert "第一步" in cleaned
        assert "config.yaml" in cleaned
        assert "providers" in cleaned
        assert "第二步" in cleaned
        assert "API key" in cleaned
        # 应该清理的
        assert "老铁们" not in cleaned
        assert "那么问题来了" not in cleaned
        assert "然后然后" not in cleaned
        assert "对对对" not in cleaned
        assert "这个真的是太" not in cleaned
        assert "点赞收藏" not in cleaned


class TestStatsCollecting:
    """测试清洗统计。"""

    def test_basic_stats(self):
        """基本统计数据。"""
        original = "老铁们,这是一段测试文本,点赞收藏"
        cleaned = "这是一段测试文本"

        stats = stats_cleaning(original, cleaned)
        assert stats["original_chars"] == len(original)
        assert stats["cleaned_chars"] == len(cleaned)
        assert stats["removed_chars"] > 0
        assert 0 < stats["removal_rate"] < 100

    def test_multiline_stats(self):
        """多行文本统计。"""
        original = """老铁们
这是内容
点赞收藏"""
        cleaned = "这是内容"

        stats = stats_cleaning(original, cleaned)
        assert stats["original_lines"] == 3
        assert stats["cleaned_lines"] == 1


# 集成测试:验证清洗不会破坏有效信息
class TestIntegration:
    """集成测试:确保清洗后信息完整。"""

    def test_technical_content_preserved(self):
        """技术类内容应完整保留。"""
        transcript = """[00:10]
老铁们,给大家分享一下,这个faster-whisper模型
[00:18]
它的large-v3版本,准确率能达到95%以上,对对对
[00:25]
配置参数是beam_size设为5,temperature设为0,就是就是这样"""

        cleaned = clean_douyin_transcript(transcript)
        # 技术细节必须保留
        assert "faster-whisper" in cleaned
        assert "large-v3" in cleaned
        assert "95%" in cleaned
        assert "beam_size" in cleaned
        assert "temperature" in cleaned
        # 数字应该保留
        assert "5" in cleaned
        assert "0" in cleaned

    def test_step_by_step_tutorial(self):
        """步骤类教程应保持结构。"""
        transcript = """[00:05]
废话不多说,直接开始
[00:08]
第一步,打开命令行,然后输入pip install
[00:15]
第二步,等待安装完成,大概需要30秒
[00:22]
第三步,运行python test.py验证,点赞收藏"""

        cleaned = clean_douyin_transcript(transcript)
        assert "第一步" in cleaned
        assert "第二步" in cleaned
        assert "第三步" in cleaned
        assert "pip install" in cleaned
        assert "30秒" in cleaned
        assert "python test.py" in cleaned
