// --- SSE Events ---

export type EventType =
  | "plan_created"
  | "agent_started"
  | "agent_progress"
  | "agent_completed"
  | "search_result"
  | "scrape_result"
  | "synthesis_started"
  | "synthesis_progress"
  | "research_complete"
  | "error";

export interface SSEEvent {
  event: EventType;
  data: Record<string, unknown>;
}

export interface PlanCreatedData {
  steps: string[];
}

export interface AgentStartedData {
  agent: string;
  step?: number;
  query?: string;
  url?: string;
}

export interface SearchResultData {
  step: number;
  results: SearchResult[];
}

export interface SearchResult {
  title: string;
  url: string;
  content: string;
  score: number;
}

export interface ScrapeResultData {
  url: string;
  content_preview: string;
}

export interface SynthesisProgressData {
  chunk: string;
}

export interface ResearchCompleteData {
  report: string;
  sources: Source[];
  tokens_used: number;
}

export interface Source {
  title: string;
  url: string;
  domain: string;
}

export interface ErrorData {
  message: string;
  agent?: string;
}

// --- API Types ---

export interface Session {
  id: string;
  title: string | null;
  model: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

// --- UI State ---

export type ResearchStatus =
  | "idle"
  | "planning"
  | "searching"
  | "scraping"
  | "synthesizing"
  | "complete"
  | "error";

export interface AgentStep {
  id: string;
  agent: string;
  status: "running" | "completed" | "error";
  label: string;
  detail?: string;
  results?: SearchResult[];
}

export interface ResearchState {
  status: ResearchStatus;
  sessionId: string | null;
  plan: string[];
  steps: AgentStep[];
  report: string;
  sources: Source[];
  error: string | null;
}
