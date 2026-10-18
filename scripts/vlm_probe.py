"""VLM 实测探针：区分「单图」与「多图同请求」两种形态，量出模型/通道的真实能力。

为什么需要它：`providers/vision.py` 把**全部抽帧塞进一次请求**，而老工具
（vision_tool.py）是**一帧一次调用**。同一个模型在两种形态下表现可能完全不同——
一次失败就断言"这个模型不能读图"是错的（2026-09-17 实测教训）。

用法：
    set MODELSCOPE_API_KEY=ms-xxx
    python scripts/vlm_probe.py --record <记录目录> --models deepseek-ai/DeepSeek-V4.1-Flash ...
    python scripts/vlm_probe.py --record <记录目录> --frames 8 --single   # 只测单图形态
    # 换成别家（任何 OpenAI 兼容端点）：--base + --key-env/--key-file
    python scripts/vlm_probe.py --record <记录> --models agnes-3.0-flash ^
        --base https://api.agnes-ai.cn/v1/chat/completions --key-file /path/to/your_key.txt

输出：每个模型 × 每种形态的 状态/耗时/是否真的读到画面文字（按能否复现帧内文字判定）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_BASE = "https://api-inference.modelscope.cn/v1/chat/completions"
PROMPT = "逐条列出这张图里出现的所有文字，只输出文字。"


def data_url(p: Path, max_side: int = 1024) -> str:
    from io import BytesIO

    from PIL import Image
    img = Image.open(p).convert("RGB")
    if max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def call(base: str, key: str, model: str, images: list[str], timeout: int) -> dict:
    """非流式调用；返回 {status, sec, chars, text, note}。"""
    content: list[dict] = [{"type": "text", "text": PROMPT}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": 1500}).encode()
    req = urllib.request.Request(base, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        return {"status": e.code, "sec": round(time.time() - t, 1), "chars": 0,
                "text": "", "note": e.read().decode("utf-8", "replace")[:120]}
    except Exception as e:  # noqa: BLE001
        return {"status": "ERR", "sec": round(time.time() - t, 1), "chars": 0,
                "text": "", "note": f"{type(e).__name__}: {e}"[:120]}
    sec = round(time.time() - t, 1)

    # 兼容三种返回：正常 JSON / SSE 流（多个 JSON 拼一起）/ 异步任务壳
    if raw.count('{"id"') > 1 or raw.startswith("data:"):
        texts = []
        for line in raw.replace("data:", "\n").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = d.get("choices") or []
            if ch:
                m = ch[0].get("message") or ch[0].get("delta") or {}
                texts.append(m.get("content") or "")
        txt = "".join(texts).strip()
        note = "SSE 流" if txt else "SSE 流但正文为空（choices 全程 null）"
    else:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": status, "sec": sec, "chars": 0, "text": "",
                    "note": "非 JSON（既不是单体也不是 SSE）"}
        ch = d.get("choices")
        if ch is None:
            return {"status": status, "sec": sec, "chars": 0, "text": "",
                    "note": f"异步任务壳 id={d.get('id')}"}
        m = ch[0].get("message") or {}
        txt = (m.get("content") or "").strip()
        note = ""
    return {"status": status, "sec": sec, "chars": len(txt), "text": txt, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM 单图/多图形态实测")
    ap.add_argument("--record", required=True, help="含 frames/ 的记录目录")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=8, help="多图形态一次送几张")
    ap.add_argument("--single", action="store_true", help="只测单图形态")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--key-env", default="MODELSCOPE_API_KEY",
                    help="从哪个环境变量读 key（默认 MODELSCOPE_API_KEY）")
    ap.add_argument("--key-file", default="",
                    help="也可以直接给 key 文件路径（老工具各家 key 都是这种 *.txt）")
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    key = ""
    if args.key_file:
        key = Path(args.key_file).read_text(encoding="utf-8").strip()
    else:
        key = os.environ.get(args.key_env, "").strip()
    if not key:
        print(f"[x] 没有 key：设 {args.key_env} 环境变量，或用 --key-file 指向 key 文件")
        return 2

    frames = sorted((Path(args.record) / "frames").glob("*.jpg"))
    if not frames:
        print(f"[x] 没找到抽帧：{Path(args.record) / 'frames'}")
        return 2
    multi = frames[:max(1, args.frames)]
    # 用画面里真实出现的文字做判定锚点（读不到就说明没真读图）
    anchor = ["Lumen", "Reader", "File", "Edit", "View", "Window"]

    print(f"记录：{args.record}")
    print(f"抽帧：{len(frames)} 张，多图形态取前 {len(multi)} 张")
    print(f"锚点文字（应能在输出里复现）：{anchor}\n")

    for m in args.models:
        print(f"■ {m}")
        forms = [("单图", [data_url(frames[6 if len(frames) > 6 else 0])])]
        if not args.single:
            forms.append((f"多图×{len(multi)}", [data_url(f) for f in multi]))
        for label, imgs in forms:
            r = call(args.base, key, m, imgs, args.timeout)
            hit = [a for a in anchor if a.lower() in r["text"].lower()]
            verdict = ("✅ 真读到画面文字" if len(hit) >= 3 else
                       "⚠️ 有返回但没复现锚点" if r["chars"] > 0 else "❌ 无有效正文")
            print(f"   {label:10} HTTP {r['status']} {r['sec']:>6}s 正文 {r['chars']:>5} 字  {verdict}"
                  f"{' | ' + r['note'] if r['note'] else ''}")
            if r["chars"]:
                print(f"      命中锚点 {hit} → 片段：{r['text'][:110].replace(chr(10), ' / ')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
