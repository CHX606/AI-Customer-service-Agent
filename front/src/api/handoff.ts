import { requestJson } from "./client";
import { getTenantToken } from "./chat";
import type { HandoffDetail, HandoffReceipt, HandoffStatus, HandoffSubmission, HandoffTicket } from "../types/handoff";

function ticketPath(ticketId: string, tenantId: string): string {
  return `/admin/handoffs/${encodeURIComponent(ticketId)}?tenant_id=${encodeURIComponent(tenantId)}`;
}

export function submitHandoff(request: HandoffSubmission): Promise<HandoffReceipt> {
  const token = getTenantToken();
  return requestJson<HandoffReceipt>("/handoff", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { "X-Tenant-Token": token } : {}) },
    body: JSON.stringify(request),
  }, "提交人工处理失败");
}

export function fetchHandoffs(tenantId: string, status?: HandoffStatus, offset = 0, limit = 50): Promise<HandoffTicket[]> {
  const query = new URLSearchParams({ tenant_id: tenantId, limit: String(limit), offset: String(offset) });
  if (status) query.set("status", status);
  return requestJson<HandoffTicket[]>(`/admin/handoffs?${query}`, {}, "加载人工待办失败");
}

export function fetchHandoff(ticketId: string, tenantId: string): Promise<HandoffDetail> {
  return requestJson<HandoffDetail>(ticketPath(ticketId, tenantId), {}, "加载事项详情失败");
}

export function updateHandoff(ticketId: string, tenantId: string, status: HandoffStatus, handlingNote: string): Promise<HandoffDetail> {
  return requestJson<HandoffDetail>(ticketPath(ticketId, tenantId), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tenant_id: tenantId, status, handling_note: handlingNote }),
  }, "保存处理结果失败");
}

export function retryHandoffNotification(ticketId: string, tenantId: string): Promise<HandoffReceipt> {
  return requestJson<HandoffReceipt>(
    `/admin/handoffs/${encodeURIComponent(ticketId)}/notification/retry?tenant_id=${encodeURIComponent(tenantId)}`,
    { method: "POST" }, "重试邮件通知失败",
  );
}
