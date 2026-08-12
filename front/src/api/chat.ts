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
  image?: File,
): Promise<ChatResponse> {
  const endpoint = image
    ? "/chat/image"
    : "/chat";

  const requestInit: RequestInit = image
    ? {
        method: "POST",
        body: buildImageFormData(request, image),
      }
    : {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      };

  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    requestInit,
  );

  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => null) as {
        detail?: string;
      } | null;

    throw new Error(
      errorData?.detail
      ?? `聊天请求失败：${response.status}`,
    );
  }

  const data: ChatResponse = await response.json();

  return data;
}


function buildImageFormData(
  request: ChatRequest,
  image: File,
): FormData {
  const formData = new FormData();

  formData.append("message", request.message);
  formData.append("session_id", request.session_id);
  formData.append("image", image, image.name);

  return formData;
}
