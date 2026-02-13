"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { User, Bot } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <div className={cn("flex gap-3 py-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600">
          <Bot size={16} className="text-white" />
        </div>
      )}

      <div
        className={cn(
          "max-w-3xl rounded-2xl px-4 py-3 text-sm",
          isUser
            ? "bg-blue-600 text-white"
            : "bg-zinc-800/70 text-zinc-300",
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="markdown-body text-sm">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                a: ({ ...props }) => (
                  <a
                    {...props}
                    target="_blank"
                    rel="noopener noreferrer"
                  />
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-zinc-700">
          <User size={16} className="text-zinc-300" />
        </div>
      )}
    </div>
  );
}
