from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    default_model: str = "claude-sonnet-4-5-20250929"

    # Tavily
    tavily_api_key: str

    # Hasdata
    hasdata_api_key: str

    # Supabase
    supabase_url: str
    supabase_anon_key: str

    # App
    cors_origins: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
