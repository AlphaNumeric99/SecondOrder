"use client";

import { Plus, Search } from "lucide-react";
import { SessionItem } from "./SessionItem";
import type { Session } from "@/types";

interface SidebarProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onNewResearch: () => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  onNewResearch,
}: SidebarProps) {
  return (
    <aside className="flex h-full w-72 flex-col border-r border-zinc-800 bg-zinc-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 p-4">
        <div className="flex items-center gap-2">
          <Search size={20} className="text-blue-400" />
          <h1 className="text-lg font-bold text-white">SecondOrder</h1>
        </div>
        <button
          onClick={onNewResearch}
          className="rounded-lg bg-blue-600 p-2 text-white transition-colors hover:bg-blue-500"
          title="New Research"
        >
          <Plus size={18} />
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto p-2">
        {sessions.length === 0 ? (
          <p className="px-3 py-8 text-center text-sm text-zinc-500">
            No research sessions yet.
            <br />
            Start a new one!
          </p>
        ) : (
          <div className="space-y-1">
            {sessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={session.id === activeSessionId}
                onClick={() => onSelectSession(session.id)}
                onDelete={() => onDeleteSession(session.id)}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
