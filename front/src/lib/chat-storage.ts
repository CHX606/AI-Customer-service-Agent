import type { ThreadMessageLike } from "@assistant-ui/react";


// ── Storage keys ──
const CONVERSATIONS_KEY = "customer-service-conversations";
const ACTIVE_ID_KEY = "customer-service-active-id";

export const CONVERSATIONS_UPDATED_EVENT =
  "customer-service-conversations-updated";
export const MAX_CONVERSATION_TITLE_LENGTH = 80;

// Legacy keys (for migration)
const LEGACY_SESSION_KEY = "customer-service-session-id";
const LEGACY_MESSAGES_KEY = "customer-service-chat-messages";


export interface StoredChatMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
}

export interface Conversation {
  id: string;
  title: string;
  isCustomTitle?: boolean;
  messages: StoredChatMessage[];
  createdAt: string;
}


// ── Helpers ──

function generateId(): string {
  return crypto.randomUUID();
}

function readConversations(): Conversation[] {
  const raw = localStorage.getItem(CONVERSATIONS_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as Conversation[];
  } catch {
    return [];
  }
}

function writeConversations(list: Conversation[]): void {
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(list));
}

function notifyConversationsUpdated(): void {
  window.dispatchEvent(new Event(CONVERSATIONS_UPDATED_EVENT));
}

function titleFromContent(text: string): string {
  const trimmed = text.replace(/\s+/g, " ").trim();
  return trimmed.length > 20 ? trimmed.slice(0, 20) + "…" : trimmed;
}


// ── Migration from single-conversation format ──

function migrateLegacy(): void {
  if (localStorage.getItem(CONVERSATIONS_KEY)) {
    // Already migrated – clean up residuals
    localStorage.removeItem(LEGACY_SESSION_KEY);
    localStorage.removeItem(LEGACY_MESSAGES_KEY);
    return;
  }

  const legacyId = localStorage.getItem(LEGACY_SESSION_KEY);
  const legacyRaw = localStorage.getItem(LEGACY_MESSAGES_KEY);

  if (!legacyId && !legacyRaw) return;

  let messages: StoredChatMessage[] = [];
  try {
    if (legacyRaw) messages = JSON.parse(legacyRaw);
  } catch { /* ignore */ }

  const id = legacyId || generateId();
  const firstUser = messages.find((m) => m.role === "user");
  const title = firstUser ? titleFromContent(firstUser.content) : "智能客服咨询";

  const conv: Conversation = {
    id,
    title,
    messages,
    createdAt: messages[0]?.createdAt || new Date().toISOString(),
  };

  writeConversations([conv]);
  localStorage.setItem(ACTIVE_ID_KEY, id);
  localStorage.removeItem(LEGACY_SESSION_KEY);
  localStorage.removeItem(LEGACY_MESSAGES_KEY);
}

migrateLegacy();


// ── Public API ──

export function getConversations(): Conversation[] {
  return readConversations();
}

export function getActiveConversationId(): string {
  const id = localStorage.getItem(ACTIVE_ID_KEY);
  if (id) {
    // Verify it actually exists
    const list = readConversations();
    if (list.some((c) => c.id === id)) return id;
  }
  // Fallback: create a new conversation
  return createNewConversation();
}

export function setActiveConversation(id: string): void {
  localStorage.setItem(ACTIVE_ID_KEY, id);
  notifyConversationsUpdated();
}

export function createNewConversation(): string {
  const id = generateId();
  const list = readConversations();

  list.unshift({
    id,
    title: "新对话",
    messages: [],
    createdAt: new Date().toISOString(),
  });

  writeConversations(list);
  localStorage.setItem(ACTIVE_ID_KEY, id);
  notifyConversationsUpdated();
  return id;
}

export function deleteConversation(id: string): void {
  let list = readConversations();
  list = list.filter((c) => c.id !== id);
  writeConversations(list);

  const activeId = localStorage.getItem(ACTIVE_ID_KEY);
  if (activeId === id) {
    if (list.length > 0) {
      localStorage.setItem(ACTIVE_ID_KEY, list[0].id);
    } else {
      createNewConversation();
      return;
    }
  }

  notifyConversationsUpdated();
}

export function renameConversation(id: string, title: string): void {
  const nextTitle = title.trim();
  if (!nextTitle) throw new Error("请输入会话名称");
  if (nextTitle.length > MAX_CONVERSATION_TITLE_LENGTH) {
    throw new Error(`会话名称不能超过 ${MAX_CONVERSATION_TITLE_LENGTH} 个字符`);
  }

  const list = readConversations();
  const conversation = list.find((item) => item.id === id);
  if (!conversation) throw new Error("该会话已不存在，请刷新列表");

  conversation.title = nextTitle;
  conversation.isCustomTitle = true;
  writeConversations(list);
  notifyConversationsUpdated();
}

/**
 * Alias kept for backward compatibility with demo-chat-adapter.
 */
export function getOrCreateSessionId(): string {
  return getActiveConversationId();
}


// ── Message persistence (scoped to active conversation) ──

export function loadStoredChatMessages(
  conversationId?: string,
): ThreadMessageLike[] {
  const list = readConversations();
  const targetId = conversationId ?? getActiveConversationId();
  const conv = list.find((c) => c.id === targetId);

  if (!conv?.messages.length) return [];

  // 为升级前的历史补稳定 ID，切换会话或刷新后仍能定位同一轮。
  if (conv.messages.some((msg) => !msg.id)) {
    conv.messages = conv.messages.map((msg) => ({ ...msg, id: msg.id || generateId() }));
    writeConversations(list);
  }

  return conv.messages.map((msg) => ({
    id: msg.id,
    role: msg.role,
    content: msg.content,
    createdAt: new Date(msg.createdAt),
  }));
}

export function saveStoredChatMessages(
  conversationId: string,
  messages: StoredChatMessage[],
): void {
  const list = readConversations();
  const idx = list.findIndex((c) => c.id === conversationId);
  if (idx === -1) return;

  list[idx].messages = messages;

  // Auto-title from first user message when still default
  if (!list[idx].isCustomTitle && list[idx].title === "新对话") {
    const firstUser = messages.find((m) => m.role === "user");
    if (firstUser) {
      list[idx].title = titleFromContent(firstUser.content);
    }
  }

  writeConversations(list);
  notifyConversationsUpdated();
}

/**
 * @deprecated  Kept for API compat; now just creates a fresh conversation.
 */
export function resetStoredChatSession(): void {
  createNewConversation();
}


// ── Export ──

export function exportChatHistory(companyName: string = "智能客服"): boolean {
  const list = readConversations();
  const activeId = getActiveConversationId();
  const conv = list.find((c) => c.id === activeId);

  if (!conv?.messages.length) return false;

  const { messages } = conv;

  const lines = [
    `# ${companyName} - 咨询记录导出`,
    `*导出时间：${new Date().toLocaleString("zh-CN")}*`,
    `*会话：${conv.title}*`,
    "",
    "---",
    "",
  ];

  messages.forEach((msg, idx) => {
    const speaker = msg.role === "user" ? "👤 用户" : `🤖 ${companyName}`;
    const time = new Date(msg.createdAt).toLocaleTimeString("zh-CN");
    lines.push(`### ${idx + 1}. ${speaker}（${time}）`);
    lines.push("");
    lines.push(msg.content);
    lines.push("");
  });

  const blob = new Blob([lines.join("\n")], {
    type: "text/markdown;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${companyName}_${conv.title}_${new Date().toISOString().slice(0, 10)}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  return true;
}
