// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { App as AntApp } from "antd";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AdminHandoffs } from "../src/components/admin-handoffs";
import { fetchHandoff, fetchHandoffs, retryHandoffNotification, updateHandoff } from "../src/api/handoff";
import type { HandoffDetail } from "../src/types/handoff";

vi.mock("../src/api/handoff", () => ({
  fetchHandoff: vi.fn(), fetchHandoffs: vi.fn(), retryHandoffNotification: vi.fn(), updateHandoff: vi.fn(),
}));
const ticket: HandoffDetail = {
  tenant_id: "default", session_id: "s", ticket_id: "ticket-a", issue: "订单需要核查", status: "pending",
  notification_status: "failed", notification_error: "邮件通知暂未成功", created_at: "2026-09-30T12:00:00Z",
  handling_note: "", messages: [{ role: "user", content: "付款后没有到账" }, { role: "assistant", content: "可以提交人工处理" }],
};
let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  window.matchMedia = vi.fn().mockImplementation((media: string) => ({
    matches: false, media, onchange: null, addListener() {}, removeListener() {},
    addEventListener() {}, removeEventListener() {}, dispatchEvent: () => true,
  }));
  vi.clearAllMocks();
  vi.mocked(fetchHandoffs).mockResolvedValue([ticket]);
  vi.mocked(fetchHandoff).mockResolvedValue(ticket);
  vi.mocked(updateHandoff).mockResolvedValue({ ...ticket, status: "resolved", handling_note: "已核实并处理" });
  vi.mocked(retryHandoffNotification).mockResolvedValue({ ticket_id: ticket.ticket_id, status: ticket.status, notification_status: "sent", message: "已记录" });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});
async function render(tenantId = "default", active = true) {
  await act(async () => { root.render(<AntApp><AdminHandoffs tenantId={tenantId} active={active} /></AntApp>); });
}
async function click(label: string) {
  const button = [...host.querySelectorAll("button")].find(item => item.textContent?.replace(/\s/g, "") === label || item.getAttribute("aria-label") === label);
  expect(button).toBeDefined();
  await act(async () => { button!.click(); });
}
async function openDetails() { await click("查看事项 ticket-a"); }
async function editNote(value: string) {
  const input = host.querySelector<HTMLTextAreaElement>("textarea")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new InputEvent("input", { bubbles: true }));
  });
}
async function selectResolved() {
  const selector = host.querySelector<HTMLElement>(".ant-form-item [role=combobox]")!;
  await act(async () => { selector.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })); });
  const option = [...document.querySelectorAll<HTMLElement>(".ant-select-item-option")].find(item => item.textContent === "已处理");
  expect(option).toBeDefined();
  await act(async () => { option!.click(); });
}

it("loads only when the tab is active and shows saved conversation details", async () => {
  await render("default", false);
  expect(fetchHandoffs).not.toHaveBeenCalled();
  await render();
  expect(fetchHandoffs).toHaveBeenCalledWith("default", "pending", 0, 50);
  await openDetails();
  expect(host.textContent).toContain("付款后没有到账");
  expect(host.textContent).toContain("通知发送失败");
  expect(fetchHandoff).toHaveBeenCalledWith("ticket-a", "default");
});

it("saves the selected processing status and handling note", async () => {
  await render();
  await openDetails();
  await editNote("已核实并处理");
  await selectResolved();
  await click("保存处理结果");
  expect(updateHandoff).toHaveBeenCalledWith("ticket-a", "default", "resolved", "已核实并处理");
  expect(host.textContent).toContain("已处理");
});

it("keeps the handling note after an API failure and supports retrying the notification", async () => {
  vi.mocked(updateHandoff).mockRejectedValue(new Error("保存失败"));
  await render();
  await openDetails();
  await editNote("需要保留的处理记录");
  await click("保存处理结果");
  expect(host.textContent).toContain("保存失败");
  expect(host.querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("需要保留的处理记录");
  await click("重试邮件通知");
  expect(retryHandoffNotification).toHaveBeenCalledWith("ticket-a", "default");
  expect(host.textContent).toContain("通知已发送");
  expect(host.querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("需要保留的处理记录");
});

it("does not display a stale detail response after switching the tenant", async () => {
  let resolveDetail!: (result: HandoffDetail) => void;
  vi.mocked(fetchHandoff).mockImplementation(() => new Promise(resolve => { resolveDetail = resolve; }));
  await render();
  await openDetails();
  vi.mocked(fetchHandoffs).mockResolvedValue([]);
  await render("other-tenant");
  await act(async () => { resolveDetail(ticket); });
  expect(host.textContent).not.toContain("付款后没有到账");
  expect(host.textContent).not.toContain("订单需要核查");
});

it("shows list errors and recovers on refresh", async () => {
  vi.mocked(fetchHandoffs).mockRejectedValueOnce(new Error("后台暂不可用"));
  await render();
  expect(host.textContent).toContain("后台暂不可用");
  await click("刷新待办");
  expect(host.textContent).toContain("订单需要核查");
});

it("fetches pending items from the server and loads the next page without losing older requests", async () => {
  const firstPage = Array.from({ length: 50 }, (_, index) => ({ ...ticket, ticket_id: `ticket-${index}`, issue: `事项 ${index}` }));
  vi.mocked(fetchHandoffs).mockResolvedValueOnce(firstPage).mockResolvedValueOnce([{ ...ticket, ticket_id: "old-ticket", issue: "较早的待处理事项" }]);
  await render();
  expect(fetchHandoffs).toHaveBeenNthCalledWith(1, "default", "pending", 0, 50);
  await click("加载更多事项");
  expect(fetchHandoffs).toHaveBeenNthCalledWith(2, "default", "pending", 50, 50);
  expect(host.textContent).toContain("较早的待处理事项");
});

it("accepts notification receipts without replacing detail and observes background delivery", async () => {
  await render();
  await openDetails();
  await editNote("还没保存的记录");
  vi.mocked(retryHandoffNotification).mockResolvedValue({ ticket_id: "ticket-a", status: "pending", notification_status: "pending", message: "通知待发送" });
  vi.mocked(fetchHandoff).mockResolvedValue({ ...ticket, notification_status: "sent", notification_error: null });
  await click("重试邮件通知");
  expect(host.textContent).toContain("通知已发送");
  expect(host.textContent).toContain("付款后没有到账");
  expect(host.querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("还没保存的记录");
});
