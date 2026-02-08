"use client";

import { Loader2 } from "lucide-react";
import { ResearchPlan } from "./ResearchPlan";
import { AgentStepList } from "./AgentStepList";
import { SourceCard } from "./SourceCard";
import type { ResearchState } from "@/types";
import { cn } from "@/lib/utils";

interface ResearchPanelProps {
  state: ResearchState;
}

const statusLabels: Record<string, string> = {
  idle: "Ready",
  planning: "Creating research plan...",
  searching: "Searching the web...",
  scraping: "Extracting content...",
  synthesizing: "Writing research report...",
  complete: "Research complete",
  error: "Error",
};

export function ResearchPanel({ state }: ResearchPanelProps) {
  if (state.status === "idle") return null;

  const completedSearches = state.steps.filter(
    (s) => s.agent === "search" && s.status === "completed",
  ).length;

  const isWorking = !["idle", "complete", "error"].includes(state.status);

  return (
    <div className="flex h-full w-80 flex-col border-l border-zinc-800 bg-zinc-900">
      {/* Status Header */}
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
        {isWorking && <Loader2 size={16} className="animate-spin text-blue-400" />}
        <span
          className={cn(
            "text-sm font-medium",
            state.status === "error" ? "text-red-400" : "text-zinc-300",
          )}
        >
          {statusLabels[state.status] || state.status}
        </span>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {/* Error */}
        {state.error && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {state.error}
          </div>
        )}

        {/* Research Plan */}
        <ResearchPlan
          steps={state.plan}
          status={state.status}
          completedSearches={completedSearches}
        />

        {/* Agent Steps */}
        <AgentStepList steps={state.steps} />

        {/* Sources */}
        {state.sources.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Sources ({state.sources.length})
            </h3>
            <div className="space-y-1.5">
              {state.sources.map((source, i) => (
                <SourceCard key={i} source={source} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
