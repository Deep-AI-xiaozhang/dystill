"""安全批量将现有具备转写和VLM素材的记录刷新为新版高颜值笔记。

纪律：
1. 自动备份现有的 note.md（到 backup_notes/），避免不可逆覆盖；
2. 保持原有的 frontmatter (title, key, source, date 等)；
3. 纯本地利用已有 transcript.txt 和 vlm.txt 生成，不调用下载与转写（省时省算力）；
4. 输出每篇生成耗时与状态。
"""
import json
import time
import shutil
from pathlib import Path
from distill_workshop.config import load_config
from distill_workshop.core.notes import generate_note, safe_title
from distill_workshop.core.frontmatter import yaml_scalar

def run_refresh():
    app = load_config()
    data_dir = Path("data")
    backup_dir = Path("backup_notes") / time.strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for d in sorted(data_dir.glob("aweme_*")):
        tr_file = d / "transcript.txt"
        vlm_file = d / "vlm.txt"
        note_file = d / "note.md"
        meta_file = d / "meta.json"
        if tr_file.is_file() and vlm_file.is_file() and tr_file.stat().st_size > 50:
            records.append(d)

    print(f"==================================================")
    print(f"🚀 开始批量升级笔记排版（共 {len(records)} 篇有效记录）")
    print(f"📦 历史旧笔记将备份至: {backup_dir}")
    print(f"==================================================")

    success_count = 0
    fail_count = 0

    for idx, d in enumerate(records, 1):
        print(f"\n[{idx}/{len(records)}] 正在处理: {d.name} ...")
        t0 = time.time()
        tr = (d / "transcript.txt").read_text(encoding="utf-8")
        vlm = (d / "vlm.txt").read_text(encoding="utf-8")

        # 提取原标题与身份
        orig_title = "未命名"
        orig_key = d.name
        orig_source = "douyin"
        note_file = d / "note.md"

        if note_file.is_file():
            # 备份旧笔记
            bak_path = backup_dir / f"{d.name}_note.md"
            shutil.copy2(note_file, bak_path)
            lines = note_file.read_text(encoding="utf-8").splitlines()
            for ln in lines[:10]:
                if ln.startswith("title:"):
                    orig_title = ln.split(":", 1)[1].strip().strip("'\"")
                    break

        meta_file = d / "meta.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                orig_key = meta.get("key") or orig_key
                orig_source = meta.get("source") or orig_source
            except Exception:
                pass

        try:
            # 调用新模板生成
            body = generate_note(tr, vlm, orig_title, app, on_step=None)

            # 拼装安全的 frontmatter
            fm = (f"---\n"
                  f"title: {yaml_scalar(orig_title)}\n"
                  f"key: {yaml_scalar(orig_key)}\n"
                  f"source: {yaml_scalar(orig_source)}\n"
                  f"date: {time.strftime('%Y-%m-%d')}\n"
                  f"---\n")

            note_file.write_text(fm + body, encoding="utf-8")
            elapsed = time.time() - t0
            print(f"  ✅ 升级成功！耗时: {elapsed:.1f}s | 字数: {len(body)}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            fail_count += 1

    print(f"\n==================================================")
    print(f"🎉 批量升级结束！成功: {success_count} 篇，失败: {fail_count} 篇")
    print(f"==================================================")

if __name__ == "__main__":
    run_refresh()
