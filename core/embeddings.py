
# Embedding management using HuggingFace API.

import requests
from typing import List, Optional
from .config import Config


class EmbeddingManager:
    # Manages text embeddings using HuggingFace Inference API.
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or Config.HUGGINGFACE_API_KEY
        self.api_url = api_url or Config.HUGGINGFACE_API_URL
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self._cache = {}  # Simple cache for embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        # Get embedding for a single text.
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Embedding API error: {response.status_code} - {response.text}")
        
        embedding = response.json()
        
        # Cache the result
        self._cache[cache_key] = embedding
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Embed a list of documents.
        return [self.get_embedding(text) for text in texts]
    
    def embed_query(self, query: str) -> List[float]:
        # Embed a single query.
        return self.get_embedding(query)
    
    def clear_cache(self):
        # Clear the embedding cache.
        self._cache.clear()
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        # Calculate cosine similarity between two embeddings.
        import math
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
