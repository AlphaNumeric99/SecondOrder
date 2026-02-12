"use client";

import { KeyboardEvent, useState } from "react";
import { ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSubmit: (query: string) => void;
  disabled?: boolean;
}

export function ChatInput({
  onSubmit,
  disabled = false,
}: ChatInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = () => {
    const trimmed = query.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setQuery("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="sticky bottom-0 z-10 bg-gradient-to-t from-zinc-950 via-zinc-950/95 to-transparent px-4 pb-4 pt-6">
      <div className="mx-auto max-w-3xl">
        {/* Textarea */}
        <div className="relative rounded-xl border border-zinc-700 bg-zinc-800 focus-within:border-blue-500">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What would you like to research?"
            disabled={disabled}
            rows={3}
            className="w-full resize-none rounded-xl bg-transparent px-4 py-3 pr-12 text-white placeholder-zinc-500 focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={handleSubmit}
            disabled={disabled || !query.trim()}
            className={cn(
              "absolute bottom-3 right-3 rounded-lg p-2 transition-colors",
              query.trim() && !disabled
                ? "bg-blue-600 text-white hover:bg-blue-500"
                : "bg-zinc-700 text-zinc-500",
            )}
          >
            <ArrowUp size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
