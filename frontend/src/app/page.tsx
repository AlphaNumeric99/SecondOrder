"use client";

import { useCallback, useEffect, useState } from "react";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessages } from "@/components/chat/ChatMessages";
import { ResearchPanel } from "@/components/research/ResearchPanel";
import { useResearch } from "@/hooks/useResearch";
import { useSessions } from "@/hooks/useSessions";
import { getSession, getModels } from "@/lib/api";
import type { Message, ModelInfo } from "@/types";

const DEFAULT_MODELS: ModelInfo[] = [
  {
    id: "claude-sonnet-4-5-20250929",
    name: "Claude Sonnet 4.5",
    description: "Fast and capable. Good balance of speed and quality.",
  },
  {
    id: "claude-opus-4-6",
    name: "Claude Opus 4.6",
    description: "Most capable model. Best for complex research.",
  },
  {
    id: "claude-haiku-4-5-20251001",
    name: "Claude Haiku 4.5",
    description: "Fastest model. Quick lookups and lightweight research.",
  },
];

export default function Home() {
  const { state: researchState, startNewResearch, reset } = useResearch();
  const { sessions, refresh: refreshSessions, deleteSession } = useSessions();

  const [models, setModels] = useState<ModelInfo[]>(DEFAULT_MODELS);
  const [selectedModel, setSelectedModel] = useState("claude-sonnet-4-5-20250929");
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Load models from backend
  useEffect(() => {
    getModels()
      .then((res) => setModels(res.models))
      .catch(() => {
        /* Use defaults */
      });
  }, []);

  // Refresh sessions when research completes
  useEffect(() => {
    if (researchState.status === "complete") {
      refreshSessions();
    }
  }, [researchState.status, refreshSessions]);

  // Track active session from research
  useEffect(() => {
    if (researchState.sessionId) {
      setActiveSessionId(researchState.sessionId);
    }
  }, [researchState.sessionId]);

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
      await startNewResearch(query, selectedModel);
    },
    [selectedModel, startNewResearch],
  );

  const handleSelectSession = useCallback(async (sessionId: string) => {
    try {
      reset();
      const data = await getSession(sessionId);
      setActiveSessionId(sessionId);
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
    setActiveSessionId(null);
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
          models={models}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          onSubmit={handleSubmit}
          disabled={isResearching}
        />
      </div>

      {/* Research Panel (right side) */}
      {researchState.status !== "idle" && <ResearchPanel state={researchState} />}
    </div>
  );
}
