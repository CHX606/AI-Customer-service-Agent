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
    const image = getLatestImage(messages);

    if (abortSignal.aborted) {
      return;
    }

    const response = await sendChatMessage({
      message: question,
      session_id: SESSION_ID,
    }, image);

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
