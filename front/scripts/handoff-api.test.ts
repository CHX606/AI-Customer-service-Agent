// @vitest-environment jsdom
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { fetchHandoff, fetchHandoffs, retryHandoffNotification, submitHandoff, updateHandoff } from "../src/api/handoff";

const fetchMock = vi.fn();
beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset().mockResolvedValue({ ok: true, json: async () => ({ ticket_id: "a" }) });
});
afterEach(() => vi.unstubAllGlobals());

it("submits the issue with tenant auth and a stable caller-provided submission id", async () => {
  sessionStorage.setItem("tenant_token", "short-session-token");
  const request = { tenant_id: "default", session_id: "session", submission_id: "stable-id", issue: "订单核查" };
  await submitHandoff(request);
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toMatch(/\/handoff$/);
  expect(init.headers["X-Tenant-Token"]).toBe("short-session-token");
  expect(JSON.parse(init.body)).toEqual(request);
});

it("encodes admin identifiers and sends only handling data during updates", async () => {
  await fetchHandoffs("tenant/name");
  await fetchHandoff("ticket/id", "tenant/name");
  await updateHandoff("ticket/id", "tenant/name", "resolved", "处理完毕");
  await retryHandoffNotification("ticket/id", "tenant/name");
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
    expect.stringMatching(/\/admin\/handoffs\?tenant_id=tenant%2Fname&limit=50&offset=0$/),
    expect.stringMatching(/\/admin\/handoffs\/ticket%2Fid\?tenant_id=tenant%2Fname$/),
    expect.stringMatching(/\/admin\/handoffs\/ticket%2Fid\?tenant_id=tenant%2Fname$/),
    expect.stringMatching(/\/admin\/handoffs\/ticket%2Fid\/notification\/retry\?tenant_id=tenant%2Fname$/),
  ]);
  expect(fetchMock.mock.calls[2][1].method).toBe("PUT");
  expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual({ tenant_id: "tenant/name", status: "resolved", handling_note: "处理完毕" });
  expect(fetchMock.mock.calls[3][1].method).toBe("POST");
});

it("propagates API errors so the form can retain and retry the issue", async () => {
  fetchMock.mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: "服务暂不可用" }) });
  await expect(submitHandoff({ tenant_id: "default", session_id: "s", submission_id: "id", issue: "问题" })).rejects.toThrow("服务暂不可用");
});
