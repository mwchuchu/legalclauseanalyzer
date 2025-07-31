from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_HOST: str = "redis"  # ← change this
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    VECTOR_DB_PATH: str = "/app/RAGembeddings"      # ← optional, centralize path

    class Config:
        env_file = ".env"

settings = Settings()
