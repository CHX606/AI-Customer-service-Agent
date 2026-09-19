import { API_BASE_URL, requestJson } from "./client";

export { API_BASE_URL } from "./client";

export interface ChatRequest {
  tenant_id?: string;
  message: string;
  session_id: string;
  turn_id?: string;
  regenerate?: boolean;
  previous_answer_hash?: string;
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  tenant_id?: string;
}

export type ChatStreamEvent =
  | { type: "status"; message: string }
  | { type: "token"; delta: string }
  | ({ type: "final" } & ChatResponse)
  | { type: "error"; message: string };

/** 获取当前站点的短期租户凭证；不从 URL 或持久化存储读取密钥。 */
export function getTenantToken(): string | null {
  if (typeof window !== "undefined") {
    const sessionToken = sessionStorage.getItem("tenant_token");
    if (sessionToken) {
      return sessionToken;
    }
  }
  return null;
}

export function setTenantToken(token: string | null): void {
  if (token) {
    sessionStorage.setItem("tenant_token", token);
  } else {
    sessionStorage.removeItem("tenant_token");
  }
}

export async function checkBackendStatus(): Promise<{ online: boolean; imageChatEnabled: boolean }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3000);
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    if (!res.ok) return { online: false, imageChatEnabled: false };
    const data = await res.json() as { status?: string; features?: { image_chat?: boolean } };
    return { online: data.status === "ok", imageChatEnabled: data.features?.image_chat === true };
  } catch {
    return { online: false, imageChatEnabled: false };
  } finally {
    clearTimeout(timer);
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  return (await checkBackendStatus()).online;
}

export async function sendChatMessage(
  request: ChatRequest,
  image?: File,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const endpoint = image ? "/chat/image" : "/chat";
  const tenantId =
    request.tenant_id ||
    (import.meta.env.VITE_TENANT_ID as string | undefined) ||
    "default";

  const payload: ChatRequest = {
    ...request,
    tenant_id: tenantId,
  };

  const tenantToken = getTenantToken();
  const headers: Record<string, string> = {};
  if (tenantToken) {
    headers["X-Tenant-Token"] = tenantToken;
  }

  const requestInit: RequestInit = image
    ? {
        method: "POST",
        headers,
        body: buildImageFormData(payload, image),
        signal,
      }
    : {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...headers,
        },
        body: JSON.stringify(payload),
        signal,
      };

  return requestJson<ChatResponse>(endpoint, requestInit, "聊天请求失败");
}

/** 以 NDJSON 读取处理进度和最终客服回答。 */
export async function* streamChatMessage(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const tenantId =
    request.tenant_id ||
    (import.meta.env.VITE_TENANT_ID as string | undefined) ||
    "default";
  const tenantToken = getTenantToken();
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(tenantToken ? { "X-Tenant-Token": tenantToken } : {}),
    },
    body: JSON.stringify({ ...request, tenant_id: tenantId }),
    signal,
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail || `聊天请求失败：HTTP ${response.status}`);
  }
  if (!response.body) {
    throw new Error("浏览器未返回可读取的流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const parseLine = (line: string): ChatStreamEvent | null => {
    const trimmed = line.trim();
    if (!trimmed) return null;
    const event = JSON.parse(trimmed) as ChatStreamEvent;
    if (!event || !["status", "token", "final", "error"].includes(event.type)) {
      throw new Error("后端返回了无法识别的流式事件");
    }
    return event;
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const event = parseLine(line);
      if (event) yield event;
    }
    if (done) break;
  }

  const trailingEvent = parseLine(buffer);
  if (trailingEvent) yield trailingEvent;
}

function buildImageFormData(request: ChatRequest, image: File): FormData {
  const formData = new FormData();

  formData.append("message", request.message);
  formData.append("session_id", request.session_id);
  formData.append("tenant_id", request.tenant_id || "default");
  formData.append("image", image, image.name);
  if (request.turn_id) formData.append("turn_id", request.turn_id);

  return formData;
}
