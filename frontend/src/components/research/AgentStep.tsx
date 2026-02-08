"use client";

import { CheckCircle2, Loader2, Search, Globe, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentStep as AgentStepType } from "@/types";

interface AgentStepProps {
  step: AgentStepType;
}

const agentIcons: Record<string, typeof Search> = {
  search: Search,
  scraper: Globe,
};

export function AgentStep({ step }: AgentStepProps) {
  const Icon = agentIcons[step.agent] || Search;

  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border px-3 py-2.5 text-sm transition-colors",
        step.status === "running"
          ? "border-blue-500/30 bg-blue-500/5"
          : step.status === "completed"
            ? "border-zinc-700/50 bg-zinc-800/50"
            : "border-red-500/30 bg-red-500/5",
      )}
    >
      {/* Status icon */}
      <div className="mt-0.5">
        {step.status === "running" ? (
          <Loader2 size={16} className="animate-spin text-blue-400" />
        ) : step.status === "completed" ? (
          <CheckCircle2 size={16} className="text-green-400" />
        ) : (
          <AlertCircle size={16} className="text-red-400" />
        )}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Icon size={14} className="text-zinc-400" />
          <span className="font-medium text-zinc-300">{step.label}</span>
        </div>
        {step.detail && (
          <p className="mt-0.5 truncate text-xs text-zinc-500">{step.detail}</p>
        )}
        {step.results && step.results.length > 0 && (
          <p className="mt-0.5 text-xs text-zinc-500">
            {step.results.length} sources found
          </p>
        )}
      </div>
    </div>
  );
}
