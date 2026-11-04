"""CLI 入口测试：重点盯 Windows GBK 控制台。

为什么值得单测：报告/帮助里带着 ✅❌⚠️，Windows 默认控制台（chcp 936）编码是 gbk，
一打印就 UnicodeEncodeError 直接崩。本机实机复现过 —— `dystill doctor` 与
`dystill check --help` 都炸，而 `--json` 与 `--version`（纯 ASCII）不炸。
早期实现早有 `sys.stdout.reconfigure` 兜底，移植到开源版时漏了。
"""
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_workshop.cli import _force_utf8_stdio, main


def _gbk_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")


def test_force_utf8_stdio_makes_emoji_printable(monkeypatch):
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", _gbk_stream())

    _force_utf8_stdio()
    assert stream.encoding.lower().replace("-", "") == "utf8"

    print("✅ 已抽核 12/15 · ⚠️ 2 条存疑")  # 修复前这里抛 UnicodeEncodeError
    stream.flush()
    assert buf.getvalue().decode("utf-8").startswith("✅")


def test_help_survives_gbk_console(monkeypatch):
    # argparse 的 help/usage 也打印 ✅❌（check 子命令），所以 reconfigure 必须在
    # parse_args 之前 —— 放在 main() 末尾或子命令里就来不及了。
    monkeypatch.setattr(sys, "stdout", _gbk_stream())
    monkeypatch.setattr(sys, "stderr", _gbk_stream())

    with pytest.raises(SystemExit) as exc:
        main(["check", "--help"])
    assert exc.value.code == 0


def test_tolerates_stream_without_reconfigure(monkeypatch):
    # stdout 可能是没有 reconfigure 的对象（被重定向/被测试框架替换），不能因此崩掉
    monkeypatch.setattr(sys, "stdout", io.BytesIO())
    monkeypatch.setattr(sys, "stderr", io.BytesIO())
    _force_utf8_stdio()


def test_version_still_works(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _gbk_stream())
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_config_example_templates_stay_in_sync():
    """根目录模板与包内模板必须一致。

    cli.init 读的是**包内**那份（importlib.resources），而 README / doctor 让人看
    **根目录**那份。两份一旦漂移，用户照着根目录改、实际生成出来的却是另一份。
    """
    from importlib.resources import files

    root_copy = Path(__file__).resolve().parents[1] / "config.example.yaml"
    if not root_copy.is_file():
        pytest.skip("非源码检出（装在 site-packages），跳过模板同步检查")
    pkg_copy = files("distill_workshop").joinpath("config.example.yaml").read_text(encoding="utf-8")
    assert root_copy.read_text(encoding="utf-8") == pkg_copy


def test_init_writes_packaged_template_and_respects_force(tmp_path, capsys):
    from distill_workshop import cli

    args = SimpleNamespace(path=str(tmp_path), force=False)
    assert cli.cmd_init(args) == 0
    config = tmp_path / "config.yaml"
    assert config.is_file()
    assert "providers:" in config.read_text(encoding="utf-8")

    assert cli.cmd_init(args) == 1
    assert "--force" in capsys.readouterr().out
    assert cli.cmd_init(SimpleNamespace(path=str(tmp_path), force=True)) == 0


def test_clean_reports_stale_dirs_without_deleting(tmp_path, monkeypatch, capsys):
    """clean 默认只报告：失败残留目录（只有 meta.json）要被列出来，但一个都不许删。"""
    from distill_workshop import cli
    from distill_workshop.config import AppConfig

    stale = tmp_path / "data" / "aweme_123"
    stale.mkdir(parents=True)
    (stale / "meta.json").write_text("{}", encoding="utf-8")
    app = AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                    data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    monkeypatch.setattr("distill_workshop.config.load_config", lambda *a, **k: app)

    assert cli.cmd_clean(SimpleNamespace(apply=False, cache_days=7)) == 0
    out = capsys.readouterr().out
    assert "aweme_123" in out
    assert stale.is_dir()  # 默认不删

    assert cli.cmd_clean(SimpleNamespace(apply=True, cache_days=7)) == 0
    assert not stale.exists()  # --apply 才真删


def test_clean_keeps_evidence_unless_purge_flag(tmp_path, monkeypatch, capsys):
    """常规 clean 只清大文件；文本证据必须由 --purge-evidence 显式授权。"""
    from distill_workshop import cli
    from distill_workshop.config import AppConfig

    data = tmp_path / "data"
    record = data / "aweme_123"
    frames = record / "frames"
    frames.mkdir(parents=True)
    evidence = record / "transcript.txt"
    frame = frames / "001.jpg"
    evidence.write_text("足够长的转写证据" * 20, encoding="utf-8")
    frame.write_bytes(b"x")
    old = time.time() - 30 * 86400
    os.utime(evidence, (old, old))
    os.utime(frame, (old, old))
    app = AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                    data_dir=data, cache_dir=tmp_path / "cache")
    monkeypatch.setattr("distill_workshop.config.load_config", lambda *a, **k: app)

    assert cli.cmd_clean(SimpleNamespace(
        apply=True, cache_days=7, purge_evidence=False)) == 0
    assert evidence.exists()
    assert not frame.exists()                    # 大文件已按常规规则清掉

    assert cli.cmd_clean(SimpleNamespace(
        apply=True, cache_days=7, purge_evidence=True)) == 0
    assert not evidence.exists()
    assert "--purge-evidence" in capsys.readouterr().out


def test_clean_finished_audio_runs_at_end_of_task(tmp_path, capsys):
    """run 结束时也要立即清音轨，不能留到下一次任务。"""
    from distill_workshop.cli import _clean_finished_audio

    record = tmp_path / "aweme_123"
    record.mkdir()
    (record / "transcript.txt").write_text(
        "足够长的有效转写正文，用来证明音轨已经用完，可以安全删除。" * 2,
        encoding="utf-8")
    audio = record / "audio.wav"
    audio.write_bytes(b"x" * 1024)

    _clean_finished_audio(tmp_path, announce=True)

    assert not audio.exists()
    assert "转写成果保留" in capsys.readouterr().out


def test_clean_purge_orphans_requires_explicit_flag(tmp_path, monkeypatch, capsys):
    """清笔记后留下的素材孤儿，只有显式 --purge-orphans 才删。"""
    from distill_workshop import cli
    from distill_workshop.config import AppConfig

    data = tmp_path / "data"
    record = data / "aweme_orphan"
    record.mkdir(parents=True)
    transcript = record / "transcript.txt"
    transcript.write_text("足够长的有效转写正文，用来模拟清掉笔记后留下的素材目录。" * 2,
                          encoding="utf-8")
    old = time.time() - 30 * 86400
    os.utime(transcript, (old, old))
    app = AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                    data_dir=data, cache_dir=tmp_path / "cache")
    monkeypatch.setattr("distill_workshop.config.load_config", lambda *a, **k: app)

    assert cli.cmd_clean(SimpleNamespace(
        apply=True, cache_days=7, purge_evidence=False, purge_orphans=False)) == 0
    assert record.exists()

    assert cli.cmd_clean(SimpleNamespace(
        apply=True, cache_days=7, purge_evidence=False, purge_orphans=True)) == 0
    assert not record.exists()
    assert "purge-orphans" in capsys.readouterr().out


def test_run_without_stt_gives_install_command(tmp_path, monkeypatch, capsys):
    """跳过 doctor 直接 run 时，缺转写也要给人能照着执行的命令。"""
    from distill_workshop import cli
    from distill_workshop.config import AppConfig

    app = AppConfig(root=tmp_path, notes_dir=tmp_path / "notes",
                    data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    monkeypatch.setattr("distill_workshop.config.load_config", lambda *a, **k: app)
    monkeypatch.setattr(cli, "find_spec", lambda name: None)

    code = cli.cmd_run(SimpleNamespace(
        urls=[str(tmp_path / "video.mp4")], json=True, check=None))
    result = json.loads(capsys.readouterr().out)

    assert code == 2
    assert "pip install faster-whisper" in result[0]["error"]
