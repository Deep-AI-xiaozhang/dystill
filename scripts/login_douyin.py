# -*- coding: utf-8 -*-
"""抖音扫码登录脚本：启动可见浏览器，等待用户扫码登录后保存Cookies。"""
import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

COOKIE_DIR = Path("cache/config")
COOKIE_DIR.mkdir(parents=True, exist_ok=True)
COOKIE_FILE = COOKIE_DIR / "cookies.json"

def main():
    print("=" * 60)
    print("🚀 正在启动浏览器窗口，请在弹出的浏览器中用【抖音 App】扫码登录...")
    print("=" * 60)
    
    with sync_playwright() as p:
        # 启动可视化窗口
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()
        page.goto("https://www.douyin.com/", wait_until="domcontentloaded")
        
        print("\n⏳ 窗口已打开，等待登录完成（检测登录状态最长等待 180 秒）...")
        
        logged_in = False
        start_time = time.time()
        
        while time.time() - start_time < 180:
            cookies = context.cookies()
            cookie_dict = {c["name"]: c["value"] for c in cookies}
            
            # 检测是否包含核心登录凭据 sessionid
            if "sessionid" in cookie_dict or "sessionid_ss" in cookie_dict or "LOGIN_STATUS" in cookie_dict:
                # 再次确认是否有用户登录态
                if any(k in cookie_dict for k in ["sessionid", "sessionid_ss", "passport_csrf_token"]):
                    logged_in = True
                    break
            time.sleep(2)
            
        if logged_in:
            import json
            cookies = context.cookies()
            cookie_dict = {c["name"]: c["value"] for c in cookies}
            COOKIE_FILE.write_text(json.dumps(cookie_dict, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n✅ 登录成功！Cookie 已安全保存至: {COOKIE_FILE}")
            print(f"📊 共获取 {len(cookies)} 项凭据（含 sessionid 登录态）")
        else:
            print("\n❌ 等待超时或未完成登录，请重新运行。")
            
        browser.close()

if __name__ == "__main__":
    main()
