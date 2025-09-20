from typing import Optional

import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    NEO4J_URL: str
    NEO4J_USER: str
    NEO4J_PWD: str
    
    MILVUS_URI: str
    MILVUS_COLLECTION_NAME: str
    MILVUS_TOKEN: str
    MILVUS_TOPK: str
    MILVUS_SEARCH_TYPE: str

    LANGSMITH_TRACING: Optional[bool]
    LANGSMITH_ENDPOINT: Optional[str]
    LANGSMITH_API_KEY: Optional[str]
    LANGSMITH_PROJECT: Optional[str]
    
    EMBEDDING_DEPLOYMENT_NAME: Optional[str]
    EMBEDDING_MODEL_NAME: Optional[str]
    EMBEDDING_AZURE_ENDPOINT: Optional[str]
    EMBEDDING_AZURE_OPENAI_API_KEY: Optional[str]
    EMBEDDING_API_VERSION: Optional[str]

    AZURE_ENDPOINT: Optional[str]
    AZURE_OPENAI_API_KEY: Optional[str]
    API_VERSION: Optional[str]
    DEPLOYMENT_NAME: Optional[str]

    APPLICATION_API_PORT: int

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


dotenv.load_dotenv(override=True)
settings = Settings()
