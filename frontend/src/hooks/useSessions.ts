"use client";

import { useCallback, useEffect, useState } from "react";
import { getSessions, deleteSession as apiDeleteSession } from "@/lib/api";
import type { Session } from "@/types";

export function useSessions() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getSessions();
      setSessions(data);
    } catch (err) {
      console.error("Failed to fetch sessions:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        await apiDeleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      } catch (err) {
        console.error("Failed to delete session:", err);
      }
    },
    [],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { sessions, loading, refresh, deleteSession };
}
