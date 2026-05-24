import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "local-faiss-index")
    VECTOR_DIMENSION = 1024

    # Pinecone Configuration
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-index")

    HUGGINGFACE_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-zh-v1.5/pipeline/feature-extraction"
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "blissful_ishizaka_626/gemma4-cloud")

    RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-large")

    DEFAULT_TOP_K = 5
    PER_RETRIEVER_K = 10
    RRF_K = 60

    MIN_MULTI_QUERIES = 5
    MAX_QUERY_LENGTH = 500
    COMPLEX_QUERY_THRESHOLD = 50

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    @classmethod
    def validate(cls) -> bool:
        return True

    @classmethod
    def get_vector_store_config(cls) -> dict:
        return {
            "index_name": cls.VECTOR_INDEX_NAME,
            "dimension": cls.VECTOR_DIMENSION,
            "backend": "faiss",
        }

    @classmethod
    def get_ollama_config(cls) -> dict:
        return {
            "base_url": cls.OLLAMA_BASE_URL,
            "api_key": cls.OLLAMA_API_KEY,
            "default_model": cls.DEFAULT_MODEL,
        }
