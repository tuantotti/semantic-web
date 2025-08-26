from typing import Optional

import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    NEO4J_URL: str
    NEO4J_USER: str
    NEO4J_PWD: str

    LANGSMITH_TRACING: Optional[bool]
    LANGSMITH_ENDPOINT: Optional[str]
    LANGSMITH_API_KEY: Optional[str]
    LANGSMITH_PROJECT: Optional[str]

    APPLICATION_API_PORT: int

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


dotenv.load_dotenv(override=True)
settings = Settings()
