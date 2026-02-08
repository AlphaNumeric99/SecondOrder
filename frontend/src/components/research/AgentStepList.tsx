"use client";

import { AgentStep } from "./AgentStep";
import type { AgentStep as AgentStepType } from "@/types";

interface AgentStepListProps {
  steps: AgentStepType[];
}

export function AgentStepList({ steps }: AgentStepListProps) {
  if (steps.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Agent Activity
      </h3>
      <div className="space-y-1.5">
        {steps.map((step) => (
          <AgentStep key={step.id} step={step} />
        ))}
      </div>
    </div>
  );
}
