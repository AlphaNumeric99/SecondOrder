from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter (required)
    openrouter_api_key: str
    default_model: str = "openai/gpt-4o-mini"
    openrouter_model: str = ""  # Optional override for default_model
    benchmark_judge_model: str = "openai/gpt-4o-mini"

    # Tavily
    tavily_api_key: str

    # Search provider
    search_provider: str = "brave"  # brave | tavily
    brave_api_key: str = ""
    search_fallback_to_tavily: bool = True

    # Hasdata
    hasdata_api_key: str

    # Supabase
    supabase_url: str
    supabase_anon_key: str

    # App
    cors_origins: str = "http://localhost:3000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
