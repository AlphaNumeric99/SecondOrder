from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter (required)
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str
    openrouter_model: str
    planner_model: str = ""  # optional override for plan generation only
    benchmark_judge_model: str

    # Tavily
    tavily_api_key: str

    # Search provider
    search_provider: str = "brave"  # brave | tavily
    brave_api_key: str = ""
    search_fallback_to_tavily: bool = True
    search_executor_mode: str = "deterministic"  # deterministic | agent_loop
    search_max_queries_per_step: int = 3
    search_max_results_per_query: int = 8
    search_max_parallel_requests: int = 8

    # Hasdata
    hasdata_api_key: str
    scrape_cache_enabled: bool = False
    scrape_cache_dir: str = ".cache/scrape"
    scrape_cache_ttl_hours: int = 168
    scrape_output_format: str = "html"  # html | text | markdown
    scrape_max_parallel_requests: int = 8
    extractor_max_page_chars: int = 120000
    extractor_fallback: str = "markitdown"  # markitdown | none
    extractor_retry_enabled: bool = True
    extractor_retry_max: int = 1
    extract_in_thread: bool = True
    scrape_headless_default: bool = True
    scrape_quality_threshold: float = 0.55
    scrape_pipeline_max_parallel: int = 4
    scrape_retry_max: int = 1
    scrape_provider: str = "firecrawl"  # firecrawl | jina_reader | playwright | auto
    firecrawl_base_url: str = "http://localhost:3002"  # self-hosted Firecrawl
    firecrawl_api_key: str = ""  # optional for self-hosted deployments
    jina_reader_base_url: str = ""  # optional fallback, keep empty for local-only

    # Staged + bounded mesh execution
    shadow_mode: bool = True
    max_parallel_search: int = 4
    max_parallel_extract: int = 6
    max_parallel_verify: int = 4

    # Memory / embeddings
    memory_backend: str = "chromadb"  # chromadb
    chroma_persist_dir: str = ".cache/chroma"
    chroma_session_ttl_hours: int = 168
    embedding_backend: str = "local"  # local
    local_embed_model: str = "bge-small-en-v1.5"
    local_embed_batch_size: int = 32

    # Orchestration quality/speed controls
    synthesis_context_char_budget: int = 45000
    prompt_bookend_enabled: bool = False
    prompt_bookend_pattern: str = "query_context_query"  # query_context_query | context_query_query | query_context | context_query
    review_mode: str = "conditional"  # conditional | always | off
    review_min_supported_ratio: float = 0.6
    review_min_citations: int = 2
    max_follow_up_rounds: int = 2
    max_follow_up_queries_per_round: int = 3
    max_post_synthesis_review_rounds: int = 1
    max_post_synthesis_follow_up_queries: int = 2

    # PostgreSQL database
    database_url: str = ""  # postgresql://user:pass@localhost:5432/dbname
    supabase_url: str = ""  # deprecated, kept for backwards compatibility
    supabase_anon_key: str = ""

    # App
    cors_origins: str = "http://localhost:3000"
    app_log_level: str = "INFO"
    noisy_log_level: str = "WARNING"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
