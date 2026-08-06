import type { ThreadMessageLike } from "@assistant-ui/react";


const SESSION_ID_KEY = "customer-service-session-id";
const CHAT_MESSAGES_KEY = "customer-service-chat-messages";


interface StoredChatMessage {
  role: "user" | "assistant";
  content: string;
  createdAt: string;
}


export function getOrCreateSessionId(): string {
  const existingSessionId = localStorage.getItem(
    SESSION_ID_KEY,
  );

  if (existingSessionId) {
    return existingSessionId;
  }

  const newSessionId = crypto.randomUUID();

  localStorage.setItem(
    SESSION_ID_KEY,
    newSessionId,
  );

  return newSessionId;
}


export function loadStoredChatMessages(): ThreadMessageLike[] {
  const storedValue = localStorage.getItem(
    CHAT_MESSAGES_KEY,
  );

  if (!storedValue) {
    return [];
  }

  try {
    const messages = JSON.parse(
      storedValue,
    ) as StoredChatMessage[];

    return messages.map((message) => ({
      role: message.role,
      content: message.content,
      createdAt: new Date(message.createdAt),
    }));
  } catch {
    localStorage.removeItem(CHAT_MESSAGES_KEY);
    return [];
  }
}


export function saveStoredChatMessages(
  messages: StoredChatMessage[],
): void {
  localStorage.setItem(
    CHAT_MESSAGES_KEY,
    JSON.stringify(messages),
  );
}


export function resetStoredChatSession(): void {
  localStorage.removeItem(SESSION_ID_KEY);
  localStorage.removeItem(CHAT_MESSAGES_KEY);
}


export type { StoredChatMessage };
