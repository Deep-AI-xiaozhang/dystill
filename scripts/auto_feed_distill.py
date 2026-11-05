# -*- coding: utf-8 -*-
"""在可视化浏览器中打开抖音，用户浏览或自动滚屏时实时捕获视频链接并自动提交蒸馏。"""
import json
import time
import re
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

COOKIE_FILE = Path("cache/config/cookies.json")

def start_interactive_harvest(target_count=2):
    print("=" * 60)
    print("🎬 正在启动可视化抖音浏览器窗口...")
    print("💡 提示：在打开的窗口中正常浏览、刷视频，系统会自动识别并捕获你正在看的高清视频！")
    print("=" * 60)
    
    collected = {}
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 800}
        )
        
        if COOKIE_FILE.is_file():
            cookies_data = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
            context.add_cookies([
                {"name": k, "value": str(v), "domain": ".douyin.com", "path": "/"}
                for k, v in cookies_data.items()
            ])
            print("🔑 账号登录态已载入")
            
        page = context.new_page()
        page.goto("https://www.douyin.com/?recommend=1", wait_until="domcontentloaded")
        
        last_url = ""
        start_t = time.time()
        
        print("\n👀 监听中...（遇到滑块验证请直接在浏览器中手动滑一下）")
        
        while len(collected) < target_count and (time.time() - start_t < 180):
            url = page.url
            if url != last_url:
                last_url = url
                # 尝试从 URL 识别当前正在播放的视频
                m = re.search(r"/(?:video|note)/(\d{15,})", url)
                if not m:
                    m = re.search(r"[?&]modal_id=(\d{15,})", url)
                    
                if m:
                    aid = m.group(1)
                    if aid not in collected:
                        # 尝试获取视频描述
                        title = page.title()
                        clean_url = f"https://www.douyin.com/video/{aid}"
                        collected[aid] = clean_url
                        print(f"\n✨ 成功捕获第 {len(collected)}/{target_count} 个视频: {clean_url}")
                        print(f"   页面标题: {title[:40]}")
                        
            time.sleep(1)
            
        browser.close()
        
    return list(collected.values())

if __name__ == "__main__":
    links = start_interactive_harvest(target_count=2)
    print(f"\n🎉 捕获完成，准备自动送入蒸馏流水线，共 {len(links)} 条:")
    for l in links:
        print("  -", l)
        
    if links:
        out_f = Path("cache/captured_links.json")
        out_f.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
