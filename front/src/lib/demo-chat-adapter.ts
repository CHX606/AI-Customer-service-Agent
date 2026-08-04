import type { ChatModelAdapter } from "@assistant-ui/react";

import { sendChatMessage } from "../api/chat";


const SESSION_ID = crypto.randomUUID();


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