"use client";

import { Trash2 } from "lucide-react";
import type { Session } from "@/types";
import { cn, formatDate, truncate } from "@/lib/utils";

interface SessionItemProps {
  session: Session;
  isActive: boolean;
  onClick: () => void;
  onDelete: () => void;
}

export function SessionItem({ session, isActive, onClick, onDelete }: SessionItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "group flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition-colors",
        isActive
          ? "bg-zinc-700 text-white"
          : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200",
      )}
    >
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium">
          {truncate(session.title || "Untitled Research", 40)}
        </p>
        <p className="text-xs text-zinc-500">{formatDate(session.created_at)}</p>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="ml-2 hidden rounded p-1 text-zinc-500 hover:bg-zinc-600 hover:text-zinc-300 group-hover:block"
      >
        <Trash2 size={14} />
      </button>
    </button>
  );
}
