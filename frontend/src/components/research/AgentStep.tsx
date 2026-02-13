"use client";

import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentStep as AgentStepType } from "@/types";

interface AgentStepProps {
  step: AgentStepType;
}

const agentLabels: Record<string, string> = {
  search: "SEARCH",
  scraper: "SCRAPE",
};

const statusLabel: Record<AgentStepType["status"], string> = {
  running: "Running",
  completed: "Done",
  error: "Error",
};

export function AgentStep({ step }: AgentStepProps) {
  return (
    <div
      className={cn(
        "rounded-lg border px-3 py-2.5 text-sm transition-colors",
        step.status === "running"
          ? "border-blue-500/30 bg-blue-500/5"
          : step.status === "completed"
            ? "border-zinc-700/50 bg-zinc-800/50"
            : "border-red-500/30 bg-red-500/5",
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-3">
        <span
          className={cn(
            "rounded-md px-1.5 py-0.5 text-[10px] font-semibold tracking-wide",
            step.agent === "search" ? "bg-sky-500/20 text-sky-300" : "bg-emerald-500/20 text-emerald-300",
          )}
        >
          {agentLabels[step.agent] || step.agent.toUpperCase()}
        </span>
        <div className="flex items-center gap-1.5 text-xs text-zinc-400">
          {step.status === "running" ? (
            <Loader2 size={14} className="animate-spin text-blue-400" />
          ) : step.status === "completed" ? (
            <CheckCircle2 size={14} className="text-green-400" />
          ) : (
            <AlertCircle size={14} className="text-red-400" />
          )}
          <span>{statusLabel[step.status]}</span>
        </div>
      </div>

      <p className="break-words text-base leading-snug font-semibold text-zinc-100">{step.label}</p>

      {step.detail && <p className="mt-1 break-words text-xs leading-relaxed text-zinc-400">{step.detail}</p>}

      {step.results && step.results.length > 0 && (
        <p className="mt-2 text-xs text-zinc-500">{step.results.length} sources found</p>
      )}

      {step.agent === "scraper" && /^https?:\/\//.test(step.label) && (
        <a
          href={step.label}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 inline-flex max-w-full break-all text-xs text-blue-400 hover:text-blue-300 hover:underline"
        >
          Open source
        </a>
      )}

      {step.agent === "search" && step.results && step.results.length > 0 && (
        <div className="mt-2 space-y-1 border-t border-zinc-700/60 pt-2">
          {step.results.slice(0, 2).map((result, idx) => (
            <a
              key={`${step.id}-result-${idx}`}
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block break-words text-xs text-zinc-300 hover:text-blue-300 hover:underline"
            >
              {result.title || result.url}
            </a>
          ))}
        </div>
      )}

      {step.agent === "search" && step.results && step.results.length > 2 && (
        <p className="mt-1 text-xs text-zinc-500">
          +{step.results.length - 2} more result{step.results.length - 2 === 1 ? "" : "s"}
        </p>
      )}

      {step.agent === "search" && !step.results && <p className="mt-1 text-xs text-zinc-500">Searching...</p>}
      {step.agent === "scraper" && step.status === "running" && (
        <p className="mt-1 text-xs text-zinc-500">Fetching page content...</p>
      )}
    </div>
  );
}
