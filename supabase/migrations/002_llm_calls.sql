-- Track all LLM API calls for cost monitoring and debugging

CREATE TABLE llm_calls (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid REFERENCES sessions(id) ON DELETE SET NULL,
    model           text NOT NULL,
    caller          text NOT NULL DEFAULT 'unknown',
    input_tokens    int NOT NULL DEFAULT 0,
    output_tokens   int NOT NULL DEFAULT 0,
    total_tokens    int NOT NULL DEFAULT 0,
    duration_ms     int,
    status          text NOT NULL DEFAULT 'success' CHECK (status IN ('success', 'error')),
    error           text,
    metadata        jsonb NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_llm_calls_session_id ON llm_calls(session_id);
CREATE INDEX idx_llm_calls_created_at ON llm_calls(created_at);
CREATE INDEX idx_llm_calls_model ON llm_calls(model);
