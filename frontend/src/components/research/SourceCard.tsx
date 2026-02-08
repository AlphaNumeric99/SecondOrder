"use client";

import { ExternalLink } from "lucide-react";
import type { Source } from "@/types";

interface SourceCardProps {
  source: Source;
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-center gap-2 rounded-lg border border-zinc-700/50 bg-zinc-800/50 px-3 py-2 text-sm transition-colors hover:border-blue-500/30 hover:bg-zinc-800"
    >
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium text-zinc-300 group-hover:text-blue-400">
          {source.title || source.domain}
        </p>
        <p className="text-xs text-zinc-500">{source.domain}</p>
      </div>
      <ExternalLink size={14} className="shrink-0 text-zinc-500 group-hover:text-blue-400" />
    </a>
  );
}
