"use client";

import { KeyboardEvent, useState } from "react";
import { ArrowUp, ChevronDown } from "lucide-react";
import type { ModelInfo } from "@/types";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  models: ModelInfo[];
  selectedModel: string;
  onModelChange: (model: string) => void;
  onSubmit: (query: string) => void;
  disabled?: boolean;
}

export function ChatInput({
  models,
  selectedModel,
  onModelChange,
  onSubmit,
  disabled = false,
}: ChatInputProps) {
  const [query, setQuery] = useState("");
  const [showModelPicker, setShowModelPicker] = useState(false);

  const selectedModelInfo = models.find((m) => m.id === selectedModel);

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

        {/* Model Selector */}
        <div className="relative mt-2 flex items-center justify-between">
          <button
            onClick={() => setShowModelPicker(!showModelPicker)}
            className="flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
          >
            {selectedModelInfo?.name || selectedModel}
            <ChevronDown size={14} />
          </button>

          {showModelPicker && (
            <div className="absolute bottom-full left-0 mb-1 rounded-lg border border-zinc-700 bg-zinc-800 p-1 shadow-xl">
              {models.map((model) => (
                <button
                  key={model.id}
                  onClick={() => {
                    onModelChange(model.id);
                    setShowModelPicker(false);
                  }}
                  className={cn(
                    "block w-full rounded-md px-3 py-2 text-left text-sm transition-colors",
                    model.id === selectedModel
                      ? "bg-blue-600/20 text-blue-400"
                      : "text-zinc-300 hover:bg-zinc-700",
                  )}
                >
                  <p className="font-medium">{model.name}</p>
                  <p className="text-xs text-zinc-500">{model.description}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
