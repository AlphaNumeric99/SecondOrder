"use client";

import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import type { ResearchStatus } from "@/types";

interface ResearchPlanProps {
  steps: string[];
  status: ResearchStatus;
  completedSearches: number;
}

export function ResearchPlan({ steps, status, completedSearches }: ResearchPlanProps) {
  if (steps.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Research Plan
      </h3>
      <div className="space-y-1.5">
        {steps.map((step, i) => {
          const isComplete = i < completedSearches;
          const isActive = i === completedSearches && (status === "searching" || status === "planning");

          return (
            <div key={i} className="rounded-md border border-zinc-800/80 bg-zinc-900/70 px-2.5 py-2">
              <div className="flex items-center gap-2 text-sm">
              {isComplete ? (
                <CheckCircle2 size={14} className="shrink-0 self-center text-green-400" />
              ) : isActive ? (
                <Loader2 size={14} className="shrink-0 self-center animate-spin text-blue-400" />
              ) : (
                <Circle size={14} className="shrink-0 self-center text-zinc-600" />
              )}
                <span
                  className={`break-words ${isComplete ? "text-zinc-400" : isActive ? "text-zinc-100" : "text-zinc-500"}`}
                >
                  {step}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
