"use client";

import { useCallback, useEffect, useState } from "react";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessages } from "@/components/chat/ChatMessages";
import { ResearchPanel } from "@/components/research/ResearchPanel";
import { useResearch } from "@/hooks/useResearch";
import { useSessions } from "@/hooks/useSessions";
import { getSession } from "@/lib/api";
import type { Message } from "@/types";

export default function Home() {
  const { state: researchState, startNewResearch, reset } = useResearch();
  const { sessions, refresh: refreshSessions, deleteSession } = useSessions();

  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  // Refresh sessions when research completes
  useEffect(() => {
    if (researchState.status === "complete") {
      refreshSessions();
    }
  }, [researchState.status, refreshSessions]);

  const activeSessionId = researchState.sessionId ?? selectedSessionId;

  const handleSubmit = useCallback(
    async (query: string) => {
      // Add user message to local state immediately
      const userMsg: Message = {
        id: `temp-${Date.now()}`,
        session_id: "",
        role: "user",
        content: query,
        metadata: {},
        created_at: new Date().toISOString(),
      };
      setMessages([userMsg]);
      await startNewResearch(query);
    },
    [startNewResearch],
  );

  const handleSelectSession = useCallback(async (sessionId: string) => {
    try {
      reset();
      const data = await getSession(sessionId);
      setSelectedSessionId(sessionId);
      setMessages(
        data.messages.map((m) => ({
          ...m,
          role: m.role as "user" | "assistant",
        })),
      );
    } catch (err) {
      console.error("Failed to load session:", err);
    }
  }, [reset]);

  const handleNewResearch = useCallback(() => {
    reset();
    setMessages([]);
    setSelectedSessionId(null);
  }, [reset]);

  const handleDeleteSession = useCallback(
    async (sessionId: string) => {
      await deleteSession(sessionId);
      if (activeSessionId === sessionId) {
        handleNewResearch();
      }
    },
    [activeSessionId, deleteSession, handleNewResearch],
  );

  const isResearching = !["idle", "complete", "error"].includes(researchState.status);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onNewResearch={handleNewResearch}
      />

      {/* Main Content */}
      <div className="relative flex flex-1 flex-col">
        <ChatMessages
          messages={messages}
          streamingContent={
            researchState.status === "synthesizing" || researchState.status === "complete"
              ? researchState.report
              : undefined
          }
        />
        <ChatInput
          onSubmit={handleSubmit}
          disabled={isResearching}
        />
      </div>

      {/* Research Panel (right side) */}
      {researchState.status !== "idle" && <ResearchPanel state={researchState} />}
    </div>
  );
}
