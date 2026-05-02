from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PP_", env_file=".env", extra="ignore")

    anthropic_base_url: str = "https://api.anthropic.com"

    lm_studio_url: str = "http://localhost:1234"
    local_model: str = "qwen3.5-9b-mlx"
    local_model_context_length: int = 32768

    host: str = "127.0.0.1"
    port: int = 8082
