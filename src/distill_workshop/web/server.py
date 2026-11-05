"""Dystill 极简高颜值本地 Web 服务端。

设计纪律：
1. 纯 Python 标准库（http.server + json + threading），零额外 pip 依赖，极速秒开；
2. 桥接已有的 dystill pipeline 与 check 核心，不重复造轮子；
3. 支持实时 SSE（Server-Sent Events）推送流水线进度日志；
4. 支持列出历史笔记、查看单篇详情、回放原视频与图片画廊。
"""
from __future__ import annotations

import json
import mimetypes
import queue
import re
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import load_config
from ..core.check import run_check
from ..core.notes import default_note_fn
from ..core.pipeline import process_record

STATIC_DIR = Path(__file__).parent / "static"


class DystillWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        # 保持控制台干净，不输出频繁的静态资源 GET 日志
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/history":
            self._handle_api_history()
        elif path.startswith("/api/record/"):
            rec_id = path.replace("/api/record/", "").strip("/")
            self._handle_api_record_detail(rec_id)
        elif path.startswith("/media/"):
            # 提供本地原视频/抽帧图片的跨域流式访问
            self._handle_media_file(path.replace("/media/", ""))
        elif path.startswith("/api/stream/"):
            task_id = path.replace("/api/stream/", "").strip("/")
            self._handle_sse_stream(task_id)
        else:
            # 静态页面路由（默认 index.html）
            if path in ("", "/"):
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/distill":
            self._handle_api_distill()
        elif parsed.path == "/api/creative":
            self._handle_api_creative()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _send_json(self, data: dict | list, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_api_history(self):
        """列出全部已蒸馏笔记."""
        app = load_config()
        data_dir = Path(app.data_dir)
        records = []
        if data_dir.is_dir():
            for d in sorted(data_dir.iterdir(), reverse=True):
                note_file = d / "note.md"
                if d.is_dir() and note_file.is_file():
                    title = "未命名"
                    badge_info = ""
                    txt = note_file.read_text(encoding="utf-8", errors="replace")
                    for line in txt.splitlines()[:10]:
                        if line.startswith("title:"):
                            title = line.split(":", 1)[1].strip().strip("'\"")
                            break
                    for line in txt.splitlines()[:15]:
                        if "✅ 已抽核" in line:
                            badge_info = line.strip("> \t")
                            break
                    records.append({
                        "id": d.name,
                        "title": title,
                        "badge": badge_info,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(note_file.stat().st_mtime)),
                    })
        self._send_json(records)

    def _handle_api_record_detail(self, rec_id: str):
        """获取某条笔记的详情、原片路径与核对报告."""
        app = load_config()
        d = Path(app.data_dir) / rec_id
        if not d.is_dir():
            self._send_json({"error": "记录不存在"}, status=HTTPStatus.NOT_FOUND)
            return

        note_file = d / "note.md"
        meta_file = d / "meta.json"
        check_file = d / "check.json"

        note_content = note_file.read_text(encoding="utf-8", errors="replace") if note_file.is_file() else ""
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        check = json.loads(check_file.read_text(encoding="utf-8")) if check_file.is_file() else {}

        # 处理本地媒体链接
        media_url = ""
        raw_media = meta.get("media")
        if raw_media:
            p = Path(raw_media)
            if p.is_file():
                media_url = f"/media/{urllib.parse.quote(str(p.resolve()))}"

        # 处理图片列表（图文帖）
        images = []
        for img_p in meta.get("images", []):
            ip = Path(img_p)
            if ip.is_file():
                images.append(f"/media/{urllib.parse.quote(str(ip.resolve()))}")

        self._send_json({
            "id": rec_id,
            "note": note_content,
            "meta": meta,
            "check": check,
            "media_url": media_url,
            "images": images,
        })

    def _handle_media_file(self, full_path_encoded: str):
        """流式发送本地音视频或图片文件."""
        file_path = Path(urllib.parse.unquote(full_path_encoded))
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"
        stat = file_path.stat()
        size = stat.st_size

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with open(file_path, "rb") as f:
            while chunk := f.read(64 * 1024):
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break

    def _handle_api_distill(self):
        """触发异步蒸馏流水线并返回 taskId."""
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(raw_body)
        except (json.JSONDecodeError, TypeError, ValueError):
            self._send_json({"error": "非法 JSON 请求"}, status=HTTPStatus.BAD_REQUEST)
            return

        urls_input = req.get("urls")
        single_url = req.get("url", "").strip()
        do_check = req.get("check", True)

        urls = []
        if isinstance(urls_input, list):
            urls = [str(u).strip() for u in urls_input if str(u).strip()]
        elif single_url:
            # 支持单行粘贴多行文本（自动切行或按空白切分）
            lines = re.split(r"[\r\n]+", single_url)
            for line in lines:
                m = re.search(r"https?://[^\s\"'<>]+", line)
                if m:
                    urls.append(m.group(0))
                elif line.strip():
                    urls.append(line.strip())

        if not urls:
            self._send_json({"error": "缺少有效 URL 链接"}, status=HTTPStatus.BAD_REQUEST)
            return

        task_id = f"task_{int(time.time()*1000)}"
        TASK_QUEUES[task_id] = queue.Queue()

        # 启动后台线程执行单条或批量流水线
        threading.Thread(target=_worker_distill_batch, args=(task_id, urls, do_check), daemon=True).start()
        self._send_json({"task_id": task_id, "count": len(urls)})

    def _handle_api_creative(self):
        """一键二创自媒体爆款文案接口."""
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(raw_body)
        except (json.JSONDecodeError, TypeError, ValueError):
            self._send_json({"error": "非法 JSON 请求"}, status=HTTPStatus.BAD_REQUEST)
            return

        note_text = req.get("note", "").strip()
        style = req.get("style", "xiaohongshu").strip()

        if not note_text:
            self._send_json({"error": "未提供有效笔记正文"}, status=HTTPStatus.BAD_REQUEST)
            return

        app = load_config()
        try:
            from ..core.creative import rewrite_note
            content = rewrite_note(note_text, style, app)
            self._send_json({"content": content, "style": style})
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"文案生成失败: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_sse_stream(self, task_id: str):
        """SSE 实时输出流."""
        q = TASK_QUEUES.get(task_id)
        if not q:
            self.send_error(HTTPStatus.NOT_FOUND, "Task not found")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        while True:
            try:
                msg = q.get(timeout=30)
                event_data = f"data: {json.dumps(msg, ensure_ascii=False)}\n\n".encode()
                self.wfile.write(event_data)
                self.wfile.flush()
                if msg.get("done"):
                    break
            except queue.Empty:
                # 发送心跳包保活
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break


TASK_QUEUES: dict[str, queue.Queue] = {}


def _worker_distill_batch(task_id: str, urls: list[str], do_check: bool):
    q = TASK_QUEUES[task_id]
    app = load_config()

    def log(msg: str):
        q.put({"type": "log", "message": msg})

    total = len(urls)
    success_records = []
    failed_records = []

    log(f"📦 收到批量蒸馏任务（共 {total} 个视频）...")

    try:
        from ..core.stt import FasterWhisperSTT
        stt = FasterWhisperSTT(app.stt.model, device=app.stt.device, compute_type=app.stt.compute_type)

        for i, u in enumerate(urls, 1):
            prefix = f"[{i}/{total}] " if total > 1 else ""
            log(f"\n{prefix}▶ 开始处理: {u}")
            try:
                res = process_record(u, app, stt=stt, note_fn=default_note_fn(app), on_step=log)
                rec_dir = Path(res["record_dir"])
                if do_check:
                    log(f"{prefix}🔍 笔记生成完毕，开始四层原子事实核对...")
                    run_check(rec_dir, app, mode="standard", on_step=log)
                success_records.append(rec_dir.name)
                log(f"{prefix}✅ 该视频蒸馏完成: {rec_dir.name}")
            except Exception as e:  # noqa: BLE001
                log(f"{prefix}❌ 该视频处理失败: {type(e).__name__}: {e}")
                failed_records.append({"url": u, "error": str(e)})

        last_id = success_records[-1] if success_records else None
        if total > 1:
            log(f"\n🎉 批量任务全部结束！成功: {len(success_records)}/{total}，失败: {len(failed_records)}")

        q.put({
            "type": "finish",
            "done": True,
            "success": len(success_records) > 0,
            "record_id": last_id,
            "success_count": len(success_records),
            "total_count": total,
        })
    except Exception as e:  # noqa: BLE001
        log(f"❌ 批量主进程异常: {type(e).__name__}: {e}")
        q.put({
            "type": "finish",
            "done": True,
            "success": False,
            "error": str(e),
        })


def start_server(port: int = 8765):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, DystillWebHandler)
    msg = f"蒸馏工坊 Web 界面已启动: http://127.0.0.1:{port}"
    try:
        print("\n==================================================")
        print(f"✨ {msg}")
        print("==================================================\n")
    except UnicodeEncodeError:
        print("\n==================================================")
        print(f"[*] {msg}")
        print("==================================================\n")
    httpd.serve_forever()


if __name__ == "__main__":
    start_server()
