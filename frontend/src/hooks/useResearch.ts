"use client";

import { useCallback, useRef, useState } from "react";
import { startResearch, getStreamUrl } from "@/lib/api";
import { connectSSE } from "@/lib/sse";
import type {
  AgentStep,
  ResearchSnapshot,
  ResearchState,
  SSEEvent,
  Source,
  SearchResult,
} from "@/types";

const initialState: ResearchState = {
  status: "idle",
  sessionId: null,
  plan: [],
  steps: [],
  report: "",
  sources: [],
  error: null,
};

export function useResearch() {
  const [state, setState] = useState<ResearchState>(initialState);
  const cleanupRef = useRef<(() => void) | null>(null);
  const stepCounterRef = useRef(0);

  const handleEvent = useCallback((event: SSEEvent) => {
    setState((prev) => {
      switch (event.event) {
        case "plan_created": {
          const steps = event.data.steps as string[];
          return { ...prev, status: "planning", plan: steps };
        }

        case "agent_started": {
          const agent = event.data.agent as string;
          const query = event.data.query as string | undefined;
          const url = event.data.url as string | undefined;
          const newStep: AgentStep = {
            id: `step-${stepCounterRef.current++}`,
            agent,
            status: "running",
            label: query || url || `${agent} agent`,
            detail: query ? `Searching: ${query}` : url ? `Scraping: ${url}` : undefined,
          };
          const newStatus = agent === "search" ? "searching" : agent === "scraper" ? "scraping" : prev.status;
          return { ...prev, status: newStatus, steps: [...prev.steps, newStep] };
        }

        case "agent_progress": {
          // Update the latest step for this agent
          const agent = event.data.agent as string;
          const updatedSteps = [...prev.steps];
          const lastIdx = updatedSteps.findLastIndex((s) => s.agent === agent && s.status === "running");
          if (lastIdx !== -1) {
            updatedSteps[lastIdx] = {
              ...updatedSteps[lastIdx],
              detail: (event.data.query as string) || (event.data.url as string) || updatedSteps[lastIdx].detail,
            };
          }
          return { ...prev, steps: updatedSteps };
        }

        case "agent_completed": {
          const agent = event.data.agent as string;
          const updatedSteps = prev.steps.map((s) =>
            s.agent === agent && s.status === "running" ? { ...s, status: "completed" as const } : s
          );
          return { ...prev, steps: updatedSteps };
        }

        case "search_result": {
          const results = event.data.results as SearchResult[];
          const step = event.data.step as number;
          const updatedSteps = [...prev.steps];
          // Find the search step and attach results
          const searchSteps = updatedSteps.filter((s) => s.agent === "search");
          if (searchSteps[step]) {
            searchSteps[step].results = results;
            searchSteps[step].detail = `Found ${results.length} results`;
          }
          return { ...prev, steps: updatedSteps };
        }

        case "scrape_result": {
          const url = event.data.url as string;
          const updatedSteps = prev.steps.map((s) =>
            s.agent === "scraper" && s.status === "running" && s.label === url
              ? { ...s, detail: `Scraped: ${(event.data.content_preview as string).slice(0, 100)}...` }
              : s
          );
          return { ...prev, steps: updatedSteps };
        }

        case "synthesis_started": {
          return { ...prev, status: "synthesizing" };
        }

        case "synthesis_progress": {
          const chunk = event.data.chunk as string;
          return { ...prev, report: prev.report + chunk };
        }

        case "research_complete": {
          const report = event.data.report as string;
          const sources = event.data.sources as Source[];
          return { ...prev, status: "complete", report, sources };
        }

        case "error": {
          const message = event.data.message as string;
          return { ...prev, status: "error", error: message };
        }

        default:
          return prev;
      }
    });
  }, []);

  const startNewResearch = useCallback(
    async (query: string) => {
      // Clean up any existing connection
      if (cleanupRef.current) {
        cleanupRef.current();
      }

      // Reset state
      stepCounterRef.current = 0;
      setState({ ...initialState, status: "planning" });

      try {
        // Start research on backend
        const { session_id } = await startResearch(query);
        setState((prev) => ({ ...prev, sessionId: session_id }));

        // Connect to SSE stream
        const streamUrl = getStreamUrl(session_id);
        cleanupRef.current = connectSSE(
          streamUrl,
          handleEvent,
          () => {
            setState((prev) =>
              prev.status !== "complete"
                ? { ...prev, status: "error", error: "Connection lost" }
                : prev
            );
          },
        );
      } catch (err) {
        setState((prev) => ({
          ...prev,
          status: "error",
          error: err instanceof Error ? err.message : "Failed to start research",
        }));
      }
    },
    [handleEvent],
  );

  const reset = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    stepCounterRef.current = 0;
    setState(initialState);
  }, []);

  const hydrateFromSession = useCallback(
    (sessionId: string, snapshot: ResearchSnapshot | null | undefined) => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      stepCounterRef.current = 0;

      if (!snapshot) {
        setState({ ...initialState, sessionId });
        return;
      }

      const normalizedSteps = (snapshot.steps || []).map((step) => ({
        ...step,
        id: step.id || `history-step-${stepCounterRef.current++}`,
      }));

      setState({
        status: snapshot.status || (snapshot.report ? "complete" : "idle"),
        sessionId,
        plan: snapshot.plan || [],
        steps: normalizedSteps,
        report: snapshot.report || "",
        sources: snapshot.sources || [],
        error: snapshot.error || null,
      });
    },
    [],
  );

  return { state, startNewResearch, reset, hydrateFromSession };
}
