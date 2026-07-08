from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PP_", env_file=".env", extra="ignore")

    anthropic_base_url: str = "https://api.anthropic.com"

    # Local backend is optional. If either lm_studio_url or local_model is unset,
    # requests targeting the local alias fall back to Anthropic + `fallback_model`.
    lm_studio_url: str | None = None
    local_model: str | None = None
    local_model_context_length: int = 32768

    # Alias sent by clients to request the local backend.
    local_model_alias: str = "local"

    # Model used when a request targets the local alias but no local backend is configured.
    fallback_model: str = "claude-haiku-4-5"

    # Path to the lms CLI binary.
    lms_path: str = "lms"

    host: str = "127.0.0.1"
    port: int = 8082

    @property
    def local_configured(self) -> bool:
        return bool(self.lm_studio_url and self.local_model)
