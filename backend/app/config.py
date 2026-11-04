from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://norvia:norvia_secure_pass@localhost:5432/norvia"
    LLM_PROVIDER: str = "fake"
    EMBEDDING_PROVIDER: str = "fake"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
