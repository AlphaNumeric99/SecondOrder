import type { EventType, SSEEvent } from "@/types";

type SSECallback = (event: SSEEvent) => void;

export function connectSSE(
  url: string,
  onEvent: SSECallback,
  onError?: (error: Event) => void,
  onOpen?: () => void,
): () => void {
  const eventSource = new EventSource(url);

  if (onOpen) {
    eventSource.onopen = onOpen;
  }

  const eventTypes: EventType[] = [
    "plan_created",
    "agent_started",
    "agent_progress",
    "agent_completed",
    "search_result",
    "scrape_result",
    "synthesis_started",
    "synthesis_progress",
    "research_complete",
    "error",
  ];

  for (const eventType of eventTypes) {
    eventSource.addEventListener(eventType, (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        onEvent({ event: eventType, data });
      } catch {
        console.error(`Failed to parse SSE event: ${eventType}`, e.data);
      }
    });
  }

  eventSource.onerror = (e) => {
    if (onError) onError(e);
    eventSource.close();
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}
