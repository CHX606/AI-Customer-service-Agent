import type { ChatModelAdapter } from "@assistant-ui/react";

import {
  sendChatMessage,
  streamChatMessage,
  type ChatResponse,
} from "../api/chat";
import {
  getOrCreateSessionId,
  getConversations,
  saveStoredChatMessages,
  type StoredChatMessage,
} from "./chat-storage";


function getLatestQuestion(
  messages: Parameters<ChatModelAdapter["run"]>[0]["messages"],
) {
  const latestUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");

  const textPart = latestUserMessage?.content.find(
    (part) => part.type === "text",
  );

  return textPart?.type === "text"
    ? textPart.text
    : "";
}


function getLatestImage(
  messages: Parameters<ChatModelAdapter["run"]>[0]["messages"],
): File | undefined {
  const latestUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");

  if (latestUserMessage?.role !== "user") {
    return undefined;
  }

  return latestUserMessage.attachments.find(
    (attachment) => attachment.type === "image",
  )?.file;
}


function serializeMessages(
  messages: Parameters<ChatModelAdapter["run"]>[0]["messages"],
): StoredChatMessage[] {
  return messages.flatMap((message) => {
    if (
      message.role !== "user"
      && message.role !== "assistant"
    ) {
      return [];
    }

    const textContent = message.content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n")
      .trim();

    const hasImage = (
      message.role === "user"
      && message.attachments.some(
        (attachment) => attachment.type === "image",
      )
    );

    const content = textContent
      || (hasImage ? "[已上传图片]" : "");

    if (!content) {
      return [];
    }

    return [
      {
        id: message.id,
        role: message.role,
        content,
        createdAt: message.createdAt.toISOString(),
      },
    ];
  });
}


export const chatAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal, unstable_assistantMessageId }) {
    const question = getLatestQuestion(messages);
    const image = getLatestImage(messages);
    const sessionId = getOrCreateSessionId();

    if (abortSignal.aborted) {
      return;
    }

    try {
      const user = [...messages].reverse().find((message) => message.role === "user");
      const stored = getConversations().find((conversation) => conversation.id === sessionId)?.messages || [];
      const previousIndex = stored.findIndex((message) => message.id === user?.id);
      const previousAnswer = previousIndex >= 0 && stored[previousIndex + 1]?.role === "assistant"
        ? stored[previousIndex + 1].content : undefined;
      const regenerating = previousAnswer !== undefined;
      const previousAnswerHash = regenerating
        ? Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(previousAnswer))))
          .map((byte) => byte.toString(16).padStart(2, "0")).join("")
        : undefined;
      const request = {
        message: question || (regenerating ? "[已上传图片]" : ""),
        session_id: sessionId,
        turn_id: user?.id,
        regenerate: regenerating,
        previous_answer_hash: previousAnswerHash,
      };
      let response: ChatResponse | null = null;
      let streamedAnswer = "";

      if (image && !regenerating) {
        response = await sendChatMessage(
          { ...request, message: question },
          image,
          abortSignal,
        );
      } else {
        for await (const event of streamChatMessage(
          request,
          abortSignal,
        )) {
          if (abortSignal.aborted) return;
          if (event.type === "error") {
            throw new Error(event.message);
          }
          if (event.type === "status") {
            yield {
              content: [{ type: "text", text: event.message }],
            };
            continue;
          }
          if (event.type === "token") {
            streamedAnswer += event.delta;
            yield {
              content: [{ type: "text", text: streamedAnswer }],
            };
            continue;
          }

          response = event;
          yield {
            content: [{ type: "text", text: event.answer }],
          };
        }
      }

      if (abortSignal.aborted) {
        return;
      }

      if (!response) {
        throw new Error("后端未返回最终客服回答");
      }

      const storedMessages = serializeMessages(messages);

      storedMessages.push({
        id: unstable_assistantMessageId,
        role: "assistant",
        content: response.answer,
        createdAt: new Date().toISOString(),
      });

      saveStoredChatMessages(sessionId, storedMessages);

      if (image && !regenerating) {
        yield {
          content: [{ type: "text", text: response.answer }],
        };
      }
    } catch (error) {
      if (abortSignal.aborted) {
        return;
      }

      const rawError = error instanceof Error ? error.message : String(error);
      const isConnectionFailed =
        rawError.includes("Failed to fetch")
        || rawError.includes("NetworkError")
        || rawError.includes("ECONNREFUSED");

      const errorText = isConnectionFailed
        ? "⚠️ **未能连接到后端 FastAPI 服务**\n\n请确认：\n1. 后端服务已在终端启动（如在根目录执行 `uvicorn back.api:app --reload` 或 `python -m back.main`）。\n2. 接口地址为 `http://127.0.0.1:8000`。\n\n启动后您可以点击下方「重新生成」或刷新页面重试。"
        : `⚠️ **请求处理失败**\n\n> 详情：${rawError}\n\n建议重新提问或检查输入内容。`;

      yield {
        content: [
          {
            type: "text",
            text: errorText,
          },
        ],
      };
    }
  },
};
