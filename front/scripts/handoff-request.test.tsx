// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { App as AntApp } from "antd";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { HandoffRequest } from "../src/components/chat/handoff-request";
import { submitHandoff } from "../src/api/handoff";

vi.mock("../src/api/handoff", () => ({ submitHandoff: vi.fn() }));
let host: HTMLDivElement;
let root: Root;
const onClose = vi.fn();
const defaults = { open: true, tenantId: "default", sessionId: "session-a", initialIssue: "订单没有到账", onClose };

beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  window.matchMedia = vi.fn().mockImplementation((media: string) => ({
    matches: false, media, onchange: null, addListener() {}, removeListener() {},
    addEventListener() {}, removeEventListener() {}, dispatchEvent: () => true,
  }));
  vi.clearAllMocks();
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

async function render(props = defaults) {
  await act(async () => { root.render(<AntApp><HandoffRequest {...props} /></AntApp>); });
}
function input() { return document.querySelector<HTMLTextAreaElement>("textarea")!; }
async function edit(value: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input(), value);
    input().dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText" }));
  });
}
async function click(label: string) {
  const button = [...document.querySelectorAll("button")].find((item) => item.textContent?.replace(/\s/g, "") === label);
  expect(button).toBeDefined();
  await act(async () => { button!.click(); await new Promise(resolve => setTimeout(resolve, 30)); });
}

it("prefills the issue, validates empty input and asks for no contact or password fields", async () => {
  await render();
  expect(input().value).toBe("订单没有到账");
  expect(document.querySelectorAll("textarea")).toHaveLength(1);
  expect(document.querySelector("input[type=password]")).toBeNull();
  await edit("   ");
  await click("提交申请");
  expect(submitHandoff).not.toHaveBeenCalled();
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 250)); });
  expect(document.body.textContent).toContain("请填写需要处理的事情");
});

it("keeps a failed submission and retries the same issue with the same id", async () => {
  vi.mocked(submitHandoff).mockRejectedValue(new Error("lost response"));
  await render();
  await click("提交申请");
  expect(input().value).toBe("订单没有到账");
  expect(document.body.textContent).toContain("提交暂未确认");
  await click("提交申请");
  const calls = vi.mocked(submitHandoff).mock.calls;
  expect(calls).toHaveLength(2);
  expect(calls[0][0]).toEqual(calls[1][0]);
  expect(calls[0][0].submission_id).toBeTruthy();
});

it("uses a new id after editing a failed issue and submits only once while pending", async () => {
  vi.mocked(submitHandoff).mockRejectedValueOnce(new Error("offline"));
  await render();
  await click("提交申请");
  const firstId = vi.mocked(submitHandoff).mock.calls[0][0].submission_id;
  await edit("需要核查第二笔订单");
  let complete!: (receipt: Awaited<ReturnType<typeof submitHandoff>>) => void;
  vi.mocked(submitHandoff).mockImplementation(() => new Promise(resolve => { complete = resolve; }));
  await click("提交申请");
  await click("提交申请");
  expect(submitHandoff).toHaveBeenCalledTimes(2);
  expect(vi.mocked(submitHandoff).mock.calls[1][0].submission_id).not.toBe(firstId);
  await act(async () => { complete({ ticket_id: "ticket-a", status: "pending", notification_status: "unconfigured", message: "recorded" }); });
  expect(document.body.textContent).toContain("申请已保存，通知暂未发送");
  expect(document.body.textContent).toContain("ticket-a");
  expect(document.body.textContent).not.toContain("邮件已发送");
});

it("starts a new application after finishing an existing application", async () => {
  vi.mocked(submitHandoff).mockResolvedValue({ ticket_id: "ticket-a", status: "pending", notification_status: "sent", message: "recorded" });
  await render();
  await click("提交申请");
  const firstId = vi.mocked(submitHandoff).mock.calls[0][0].submission_id;
  await click("完成");
  expect(onClose).toHaveBeenCalledOnce();
  await render({ ...defaults, open: false });
  await render({ ...defaults, initialIssue: "另外一件事" });
  expect(input().value).toBe("另外一件事");
  await click("提交申请");
  expect(vi.mocked(submitHandoff).mock.calls[1][0].submission_id).not.toBe(firstId);
});

it("keeps a failed issue across reopening but resets it when switching sessions", async () => {
  vi.mocked(submitHandoff).mockRejectedValue(new Error("offline"));
  await render();
  await edit("我补充的故障详情");
  await click("提交申请");
  await render({ ...defaults, open: false });
  await render();
  expect(input().value).toBe("我补充的故障详情");
  await render({ ...defaults, sessionId: "session-b", initialIssue: "新会话的问题" });
  expect(input().value).toBe("新会话的问题");
});
