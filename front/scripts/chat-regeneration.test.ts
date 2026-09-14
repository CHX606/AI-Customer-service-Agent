// @vitest-environment jsdom
import { webcrypto } from "node:crypto";
import { beforeEach, expect, it, vi } from "vitest";
import { chatAdapter } from "../src/lib/chat-adapter";
import { sendChatMessage, streamChatMessage } from "../src/api/chat";
import { createNewConversation, getConversations, loadStoredChatMessages, saveStoredChatMessages } from "../src/lib/chat-storage";

vi.mock("../src/api/chat", () => ({ sendChatMessage: vi.fn(), streamChatMessage: vi.fn() }));

const time = new Date("2026-09-12T00:00:00Z");
const user = (id: string, text = "问题") => ({ id, role: "user", content: [{ type: "text", text }], attachments: [], createdAt: time });
const assistant = (id: string, text = "旧答案") => ({ id, role: "assistant", content: [{ type: "text", text }], createdAt: time });

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  vi.stubGlobal("crypto", webcrypto);
  vi.mocked(streamChatMessage).mockImplementation(async function* (request) {
    yield { type: "final", answer: "新答案", session_id: request.session_id };
  });
});

async function run(messages: unknown[]) {
  const options = { messages, abortSignal: new AbortController().signal, unstable_assistantMessageId: "new-answer" };
  const generator = chatAdapter.run(options as Parameters<typeof chatAdapter.run>[0]) as AsyncGenerator;
  for await (const _event of generator) { /* consume */ }
}

it("sends a stable turn id and preserves the ordinary new-message operation", async () => {
  const id = createNewConversation();
  await run([user("u1")]);
  expect(streamChatMessage).toHaveBeenCalledWith(expect.objectContaining({ session_id: id, turn_id: "u1", regenerate: false }), expect.any(AbortSignal));
  expect(getConversations()[0].messages.map((message) => message.id)).toEqual(["u1", "new-answer"]);
});

it("regenerates an earlier answer and saves only the selected branch", async () => {
  const id = createNewConversation();
  saveStoredChatMessages(id, [
    { id: "u1", role: "user", content: "问题", createdAt: time.toISOString() },
    { id: "a1", role: "assistant", content: "旧答案", createdAt: time.toISOString() },
    { id: "u2", role: "user", content: "后续问题", createdAt: time.toISOString() },
    { id: "a2", role: "assistant", content: "后续答案", createdAt: time.toISOString() },
  ]);
  await run([user("u1")]);
  const request = vi.mocked(streamChatMessage).mock.calls[0][0];
  expect(request.turn_id).toBe("u1");
  expect(request.regenerate).toBe(true);
  expect(request.previous_answer_hash).toMatch(/^[a-f0-9]{64}$/);
  expect(getConversations()[0].messages.map((message) => message.content)).toEqual(["问题", "新答案"]);
});

it("does not confuse a new identical question with regeneration", async () => {
  const id = createNewConversation();
  saveStoredChatMessages(id, [
    { id: "u1", role: "user", content: "问题", createdAt: time.toISOString() },
    { id: "a1", role: "assistant", content: "旧答案", createdAt: time.toISOString() },
  ]);
  await run([user("u1"), assistant("a1"), user("u2")]);
  expect(vi.mocked(streamChatMessage).mock.calls[0][0].regenerate).toBe(false);
  expect(getConversations()[0].messages).toHaveLength(4);
});

it("keeps migrated history ids stable across reloads and retries images without reupload", async () => {
  const id = createNewConversation();
  saveStoredChatMessages(id, [
    { role: "user", content: "[已上传图片]", createdAt: time.toISOString() },
    { role: "assistant", content: "图片诊断", createdAt: time.toISOString() },
  ]);
  const first = loadStoredChatMessages(id);
  const second = loadStoredChatMessages(id);
  expect(first.map((message) => message.id)).toEqual(second.map((message) => message.id));
  await run([user(first[0].id!, "[已上传图片]")]);
  expect(sendChatMessage).not.toHaveBeenCalled();
  expect(vi.mocked(streamChatMessage).mock.calls[0][0].regenerate).toBe(true);
});

it("keeps saved history when regeneration fails", async () => {
  const id = createNewConversation();
  saveStoredChatMessages(id, [
    { id: "u1", role: "user", content: "问题", createdAt: time.toISOString() },
    { id: "a1", role: "assistant", content: "旧答案", createdAt: time.toISOString() },
  ]);
  const before = getConversations();
  vi.mocked(streamChatMessage).mockImplementation(async function* () {
    yield { type: "error", message: "服务暂不可用" };
  });
  await run([user("u1")]);
  expect(getConversations()).toEqual(before);
});
