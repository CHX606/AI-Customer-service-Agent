"""Exercise the real frontend in an isolated Chromium container.

The host reads management credentials privately and passes them over stdin;
the browser container never receives the application's API configuration.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def management_check(browser, settings):
    from playwright.sync_api import expect

    admin_context = browser.new_context(
        viewport={"width": 1440, "height": 1000}, locale="zh-CN",
        http_credentials={"username": settings["username"], "password": settings["password"]},
    )
    admin = admin_context.new_page()
    errors = []
    admin.on("pageerror", lambda error: errors.append(str(error)))
    response = admin.goto(settings["url"].rstrip("/") + "/manage", wait_until="networkidle", timeout=60000)
    assert response and response.status == 200, "Management authentication failed"
    admin.get_by_role("button", name="管理后台", exact=True).click()
    expect(admin.get_by_text("企业客服管理", exact=True)).to_be_visible(timeout=30000)
    company = admin.get_by_label("企业全称", exact=True)
    expect(company).to_be_visible(timeout=30000)
    expect(company).not_to_have_value("", timeout=30000)
    admin.get_by_role("tab", name="知识库文档").click()
    expect(admin.get_by_text("可乐云客服操作文档.docx", exact=True)).to_be_visible(timeout=30000)
    expect(admin.locator(".knowledge-hint")).to_contain_text("DOCX 图片会按服务器配置进行语义解析")
    admin.screenshot(path="/tmp/ai-customer-service-management.png", full_page=True)
    assert not errors, "Uncaught JavaScript error in management frontend"
    print("PASS: browser management authentication, profile and knowledge list", flush=True)
    admin_context.close()


def browser_check(settings):
    from playwright.sync_api import expect, sync_playwright

    origin = settings["url"].rstrip("/")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/usr/bin/chromium", headless=True,
            args=["--disable-dev-shm-usage", "--disable-gpu"],
        )
        if settings.get("admin_only"):
            management_check(browser, settings)
            browser.close()
            return
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="zh-CN")
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto(origin, wait_until="networkidle", timeout=60000)
        assert response and response.status == 200, "Frontend navigation failed"
        expect(page.get_by_text("客服在线", exact=True)).to_be_visible(timeout=45000)
        print("PASS: browser rendered the real frontend with backend online", flush=True)
        question = "续费之后为什么流量没有重置？"
        expect(page.get_by_role("button", name="上传故障截图", exact=True)).to_be_visible()
        page.get_by_placeholder("描述你遇到的问题，或粘贴一张截图…", exact=True).fill(question)
        started = time.monotonic()
        with page.expect_response(lambda response: response.url.endswith("/api/chat/stream"), timeout=300000) as pending:
            page.get_by_role("button", name="发送消息", exact=True).click()
        reply = pending.value
        assert reply.status == 200, "Chat HTTP request failed"
        events = [json.loads(line) for line in reply.body().splitlines() if line.strip()]
        assert events and not any(event.get("type") == "error" for event in events)
        final = next(event for event in reversed(events) if event.get("type") == "final")
        assert "重置" in final["answer"] and any(word in final["answer"] for word in ("有效期", "延长", "期限"))
        expect(page.locator(".message-list .message-text").last).to_contain_text("重置", timeout=15000)
        page.wait_for_function("() => (JSON.parse(localStorage.getItem('customer-service-conversations') || '[]')).some(c => c.messages.some(m => m.role === 'assistant' && m.content.includes('重置')))", timeout=15000)
        elapsed = round(time.monotonic() - started, 2)
        page.reload(wait_until="networkidle", timeout=60000)
        expect(page.locator(".message-list .message-text").last).to_contain_text("重置", timeout=15000)
        page.screenshot(path="/tmp/ai-customer-service-chat.png", full_page=True)
        assert not errors, "Uncaught JavaScript error in public frontend"
        print(json.dumps({"check": "browser_chat_and_reload_persistence", "status": "PASS", "chat_seconds": elapsed, "session_id": final["session_id"]}), flush=True)

        page.get_by_role("button", name="新建对话", exact=True).click()
        with page.expect_file_chooser(timeout=15000) as chooser:
            page.get_by_role("button", name="上传故障截图", exact=True).click()
        chooser.value.set_files("/tmp/ai-customer-service-unknown-image.png")
        expect(page.locator('.composer-area img[alt="ai-customer-service-unknown-image.png"]')).to_be_visible(timeout=15000)
        page.get_by_placeholder("描述你遇到的问题，或粘贴一张截图…", exact=True).fill("截图里这个导入问题怎么处理？")
        started = time.monotonic()
        with page.expect_response(lambda response: response.url.endswith("/api/chat/image"), timeout=300000) as pending:
            page.get_by_role("button", name="发送消息", exact=True).click()
        image_reply = pending.value
        assert image_reply.status == 200, "Browser image chat failed"
        image_answer = image_reply.json()["answer"]
        assert len(image_answer) > 20 and any(word in image_answer for word in ("订阅", "导入", "TLS"))
        expect(page.locator(".message-list .message-text").last).to_contain_text(image_answer[:15], timeout=15000)
        page.screenshot(path="/tmp/ai-customer-service-image-chat.png", full_page=True)
        assert not errors, "Uncaught JavaScript error in image upload"
        print(json.dumps({"check": "browser_image_upload_and_answer", "status": "PASS", "seconds": round(time.monotonic() - started, 2)}), flush=True)
        context.close()

        management_check(browser, settings)
        browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--admin-only", action="store_true", help="Verify management without making more model requests")
    args = parser.parse_args()
    if args.inside:
        settings = json.load(sys.stdin)
        try:
            browser_check(settings)
        except Exception as error:
            # Do not emit a browser call log that could contain credentials.
            print(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
                              "frames": [{"function": frame.name, "line": frame.lineno} for frame in traceback.extract_tb(error.__traceback__)],
                              "assertion": str(error).splitlines()[0][:160] if isinstance(error, AssertionError) else None}), flush=True)
            raise SystemExit(1)
        return
    assert args.url, "Supply the deployed origin"
    root = Path(__file__).resolve().parents[2]
    credentials = dict(line.split(": ", 1) for line in
                       (root / "deployment-secrets.txt").read_text().splitlines() if ": " in line)
    container = "ai-customer-service-browser-tools"
    subprocess.run(["docker", "cp", str(Path(__file__).resolve()), container + ":/tmp/verify_browser.py"], check=True)
    result = subprocess.run(
        ["docker", "exec", "-i", container, "python", "/tmp/verify_browser.py", "--inside"],
        input=json.dumps({"url": args.url, "username": credentials["Username"], "password": credentials["Password"],
                          "admin_only": args.admin_only}), text=True,
    )
    if result.returncode == 0:
        for name in (("management",) if args.admin_only else ("chat", "image-chat", "management")):
            subprocess.run(["docker", "cp", f"{container}:/tmp/ai-customer-service-{name}.png",
                            str(root / f"deployment-browser-{name}.png")], check=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
