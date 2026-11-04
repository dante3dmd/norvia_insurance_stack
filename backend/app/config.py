from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://norvia:norvia_secure_pass@localhost:5432/norvia"
    REDIS_URL: str = "redis://localhost:6379/0"
    LLM_PROVIDER: str = "fake"
    EMBEDDING_PROVIDER: str = "fake"
    USER_RATE_LIMIT_PER_MINUTE: int = 5
    USER_RATE_BURST: int = 2
    GLOBAL_RATE_LIMIT_PER_MINUTE: int = 20
    CONTEXT_WINDOW_TOKENS: int = 32768
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SERVICE_NAME: str = "norvia-backend"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
