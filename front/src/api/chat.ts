const API_BASE_URL = "http://127.0.0.1:8000";


export interface ChatRequest {
  message: string;
  session_id: string;
}


export interface ChatResponse {
  answer: string;
  session_id: string;
}


export async function sendChatMessage(
  request: ChatRequest,
): Promise<ChatResponse> {
  const response = await fetch(
    `${API_BASE_URL}/chat`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(request),
    },
  );

  if (!response.ok) {
    throw new Error(
      `聊天请求失败：${response.status}`,
    );
  }

  const data: ChatResponse = await response.json();

  return data;
}