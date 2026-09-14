// @vitest-environment jsdom
import { beforeEach, expect, it, vi } from "vitest";
import {
  CONVERSATIONS_UPDATED_EVENT,
  MAX_CONVERSATION_TITLE_LENGTH,
  createNewConversation,
  getActiveConversationId,
  getConversations,
  renameConversation,
  saveStoredChatMessages,
  type StoredChatMessage,
} from "../src/lib/chat-storage";

beforeEach(() => localStorage.clear());
const messages: StoredChatMessage[] = [
  { role: "user", content: "如何更新订阅？", createdAt: "2026-09-11T06:00:00Z" },
  { role: "assistant", content: "请打开订阅设置。", createdAt: "2026-09-11T06:00:01Z" },
];

it("persists an inactive conversation's new title without changing its messages or active selection", () => {
  const targetId = createNewConversation();
  saveStoredChatMessages(targetId, messages);
  const activeId = createNewConversation();
  const before = getConversations();
  const updated = vi.fn();
  window.addEventListener(CONVERSATIONS_UPDATED_EVENT, updated);
  try {
    renameConversation(targetId, "  订阅问题记录  ");
    expect(getConversations()).toEqual(before.map(item => item.id === targetId
      ? { ...item, title: "订阅问题记录", isCustomTitle: true }
      : item));
    expect(getActiveConversationId()).toBe(activeId);
    expect(updated).toHaveBeenCalledTimes(1);
  } finally {
    window.removeEventListener(CONVERSATIONS_UPDATED_EVENT, updated);
  }
});

it("preserves a manually chosen default-looking title when messages are saved", () => {
  const customId = createNewConversation();
  renameConversation(customId, "新对话");
  saveStoredChatMessages(customId, messages);
  expect(getConversations().find(item => item.id === customId)?.title).toBe("新对话");

  const automaticId = createNewConversation();
  saveStoredChatMessages(automaticId, messages);
  expect(getConversations().find(item => item.id === automaticId)?.title).toBe("如何更新订阅？");
});

it("rejects blank, excessive, or stale rename requests without changing stored conversations", () => {
  const id = createNewConversation();
  const before = getConversations();
  expect(() => renameConversation(id, "   ")).toThrow("请输入会话名称");
  expect(() => renameConversation(id, "a".repeat(MAX_CONVERSATION_TITLE_LENGTH + 1))).toThrow("不能超过");
  expect(() => renameConversation("missing", "新名称")).toThrow("已不存在");
  expect(getConversations()).toEqual(before);
  expect(getActiveConversationId()).toBe(id);
});
