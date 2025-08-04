from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    # Use relative path or create in current directory
    VECTOR_DB_PATH: str = "./RAGembeddings"  # FIXED: relative path

settings = Settings()
