import type { ChatModelAdapter } from "@assistant-ui/react";

import { sendChatMessage } from "../api/chat";
import {
  getOrCreateSessionId,
  saveStoredChatMessages,
  type StoredChatMessage,
} from "./chat-storage";


const SESSION_ID = getOrCreateSessionId();


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
    : "你的问题";
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

    const content = message.content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n")
      .trim();

    if (!content) {
      return [];
    }

    return [
      {
        role: message.role,
        content,
        createdAt: message.createdAt.toISOString(),
      },
    ];
  });
}


export const demoChatAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const question = getLatestQuestion(messages);

    if (abortSignal.aborted) {
      return;
    }

    const response = await sendChatMessage({
      message: question,
      session_id: SESSION_ID,
    });

    if (abortSignal.aborted) {
      return;
    }

    const storedMessages = serializeMessages(messages);

    storedMessages.push({
      role: "assistant",
      content: response.answer,
      createdAt: new Date().toISOString(),
    });

    saveStoredChatMessages(storedMessages);

    yield {
      content: [
        {
          type: "text",
          text: response.answer,
        },
      ],
    };
  },
};
