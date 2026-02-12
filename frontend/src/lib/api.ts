const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || "Request failed");
  }
  return res.json();
}

export async function startResearch(query: string) {
  return request<{ session_id: string }>("/api/research", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export async function getSessions() {
  return request<
    Array<{
      id: string;
      title: string | null;
      model: string;
      created_at: string;
      updated_at: string;
    }>
  >("/api/sessions");
}

export async function getSession(sessionId: string) {
  return request<{
    session: { id: string; title: string | null; model: string; created_at: string; updated_at: string };
    messages: Array<{ id: string; session_id: string; role: string; content: string; metadata: Record<string, unknown>; created_at: string }>;
  }>(`/api/sessions/${sessionId}`);
}

export async function deleteSession(sessionId: string) {
  return request<{ status: string }>(`/api/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export function getStreamUrl(sessionId: string): string {
  return `${API_URL}/api/research/${sessionId}/stream`;
}
