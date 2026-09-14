const configuredBaseUrl = (
  import.meta.env.VITE_API_BASE_URL as string | undefined
)?.trim();

export const API_BASE_URL = (configuredBaseUrl || "http://127.0.0.1:8000")
  .replace(/\/$/, "");

interface ApiErrorBody {
  detail?: string;
}

export async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  fallbackMessage = "请求失败",
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiErrorBody | null;
    throw new Error(body?.detail || `${fallbackMessage}：HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getErrorMessage(error: unknown, fallback: string): string {
  if (!(error instanceof Error) || !error.message) return fallback;

  if (
    error instanceof TypeError ||
    /failed to fetch|networkerror|load failed/i.test(error.message)
  ) {
    return "暂时无法连接后端服务，请确认 FastAPI 已启动";
  }

  return error.message;
}
