"use client";

import { useEffect, useRef } from "react";
import { ChatMessage } from "./ChatMessage";
import type { Message } from "@/types";

interface ChatMessagesProps {
  messages: Message[];
  streamingContent?: string;
}

export function ChatMessages({ messages, streamingContent }: ChatMessagesProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  return (
    <div className="flex-1 overflow-y-auto px-4">
      <div className="mx-auto max-w-3xl pb-40 pt-8 text-sm">
        {messages.length === 0 && !streamingContent ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <h2 className="mb-2 text-2xl font-bold text-white">SecondOrder</h2>
            <p className="text-zinc-400">
              The world&apos;s most advanced deep research tool.
              <br />
              Ask anything — I&apos;ll research it thoroughly.
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <ChatMessage key={msg.id} role={msg.role} content={msg.content} />
            ))}
            {streamingContent && (
              <ChatMessage role="assistant" content={streamingContent} />
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
