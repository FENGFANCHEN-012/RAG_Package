from typing import List
from langchain_core.embeddings import Embeddings
import requests
import os
import sys
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
import jieba  # Assuming jieba for Chinese tokenization
from langchain_core.documents import Document
import numpy as np
import faiss

try:
    from langchain_pinecone import Pinecone as PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

class MyHuggingFaceEmbeddings(Embeddings):

    def __init__(self, index=None):
        self.API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-zh-v1.5/pipeline/feature-extraction"
        token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or ""
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

        self.index = index
        # Use FAISS for local vector storage fallback
        self.index_store = None  # Will be initialized when documents are added
        self.document_store = []  # Store documents with their embeddings
        self.embedding_dim = 1024  # bge-large-zh-v1.5 dimension
        self._local_embedder = None
        self.bm25 = None
        self.documents = []
        self._vector_store = None

        # If a Pinecone index is provided, wrap it with PineconeVectorStore
        if index is not None and PineconeVectorStore is not None:
            try:
                self._vector_store = PineconeVectorStore(
                    index=index, embedding=self, text_key="text"
                )
                print("[Vector Store] Using PineconeVectorStore for retrieval", file=sys.stderr)
            except Exception as e:
                print(f"[Vector Store] PineconeVectorStore init failed: {e}", file=sys.stderr)

    @property
    def vector_store(self):
        """Return PineconeVectorStore when available, otherwise self (FAISS)."""
        if self._vector_store is not None:
            return self._vector_store
        return self


    
    def _get_local_embedder(self):
        if self._local_embedder is None:
            from sentence_transformers import SentenceTransformer

            self._local_embedder = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        return self._local_embedder



    def _embed_local(self, text: str) -> List[float]:
        model = self._get_local_embedder()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def build_bm25(self, documents: List[Document]):
        """Build BM25 index from LangChain Documents."""
        self.documents = documents
        # simple tokenization for Chinese using jieba, fallback to list(text) if not available
        tokenized_corpus = []
        for doc in documents:
            try:
                tokenized_corpus.append(list(jieba.cut(doc.page_content)))
            except:
                tokenized_corpus.append(list(doc.page_content))
        self.bm25 = BM25Okapi(tokenized_corpus)

    def build_faiss_index(self, documents: List[Document]):
        """Build FAISS index from LangChain Documents for vector search."""
        self.documents = documents
        if not documents:
            print("[FAISS] No documents to build index", file=sys.stderr)
            return

        try:
            # Get embeddings for all documents
            embeddings = []
            for i, doc in enumerate(documents):
                embedding = self.get_embedding(doc.page_content)
                embeddings.append(embedding)
                if i == 0:
                    print(f"[FAISS] First document embedding dimension: {len(embedding)}", file=sys.stderr)

            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            print(f"[FAISS] Building index with {len(documents)} documents, embedding dim: {embeddings_array.shape}", file=sys.stderr)

            # Initialize FAISS index
            self.embedding_dim = len(embeddings[0])
            self.index_store = faiss.IndexFlatL2(self.embedding_dim)
            self.index_store.add(embeddings_array)
            self.document_store = documents
            print(f"[FAISS] Index built successfully, total vectors: {self.index_store.ntotal}", file=sys.stderr)
        except Exception as e:
            print(f"[FAISS] Failed to build index: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    def _matches_filter(self, doc: Document, filter: dict = None) -> bool:
        if not filter:
            return True
        metadata = getattr(doc, "metadata", {}) or {}
        for key, expected in filter.items():
            actual = metadata.get(key)
            if isinstance(expected, dict):
                for operator, value in expected.items():
                    if operator == "$gte" and not (actual is not None and actual >= value):
                        return False
                    if operator == "$lte" and not (actual is not None and actual <= value):
                        return False
                    if operator == "$gt" and not (actual is not None and actual > value):
                        return False
                    if operator == "$lt" and not (actual is not None and actual < value):
                        return False
            elif actual != expected:
                return False
        return True

    def similarity_search(self, query: str, k: int = 10, filter: dict = None) -> List[Document]:
        """Search using FAISS vector similarity."""
        if self.index_store is None or not self.document_store:
            print(f"[FAISS] Search skipped - index_store={self.index_store is not None}, document_store={len(self.document_store) if self.document_store else 0}", file=sys.stderr)
            return []

        try:
            # Get query embedding
            query_embedding = np.array([self.get_embedding(query)]).astype('float32')
            print(f"[FAISS] Searching for query '{query[:30]}...', k={k}", file=sys.stderr)

            # Search
            search_k = min(max(k * 5, k), len(self.document_store))
            distances, indices = self.index_store.search(query_embedding, search_k)
            print(f"[FAISS] Search returned {len(indices[0])} results", file=sys.stderr)

            # Return documents
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.document_store):
                    doc = self.document_store[idx]
                    if self._matches_filter(doc, filter):
                        results.append(doc)
                    if len(results) >= k:
                        break

            print(f"[FAISS] Returning {len(results)} documents", file=sys.stderr)
            return results
        except Exception as e:
            print(f"[FAISS] Search failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return []

    def bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        try:
            tokenized_query = list(jieba.cut(query))
        except:
            tokenized_query = list(query)
        
        scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_n]

    def add_texts(self, texts: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(len(self.documents) + idx) for idx in range(len(texts))]
        new_docs = [
            Document(
                page_content=text,
                metadata={**(metadatas[idx] or {}), "id": ids[idx]},
            )
            for idx, text in enumerate(texts)
        ]
        self.documents = (self.documents or []) + new_docs
        self.build_bm25(self.documents)
        self.build_faiss_index(self.documents)
        return ids

    def get_embedding(self, text):
        if not self.headers:
            return self._embed_local(text)

        response = requests.post(self.API_URL, headers=self.headers, json={"inputs": text}, timeout=60)

        try:
            payload = response.json()
        except ValueError:
            snippet = (response.text or "").strip().replace("\n", " ")[:240]
            raise RuntimeError(
                f"Embedding API returned non-JSON (status={response.status_code}): {snippet}"
            )

        if response.status_code >= 400:
            message = payload.get("error") if isinstance(payload, dict) else str(payload)
            if response.status_code in (401, 403):
                return self._embed_local(text)
            raise RuntimeError(f"Embedding API error (status={response.status_code}): {message}")

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Embedding API error: {payload['error']}")

        # HF feature-extraction may return nested arrays: [[...]].
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            return payload[0]

        if isinstance(payload, list):
            return payload

        raise RuntimeError(f"Unexpected embedding payload type: {type(payload).__name__}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self.get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self.get_embedding(text)
    

    def upload_to_vector_store(self):
        raise NotImplementedError("Use add_texts() to index documents into the active vector store.")


class CrossEncoderReranker:
    """Cross-Encoder Reranker for improving retrieval accuracy."""
    
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-reranker-large"):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """Initialize the reranker with a cross-encoder model."""
        self.model_name = model_name
        self._model = None
        self._device = None
    
    def _load_model(self):
        """Lazy loading of the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                import torch
                
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading CrossEncoder model {self.model_name} on {self._device}...")
                
                self._model = CrossEncoder(
                    self.model_name,
                    device=self._device,
                    max_length=512
                )
                print(f"CrossEncoder model loaded successfully")
            except Exception as e:
                print(f"Failed to load CrossEncoder: {e}")
                raise
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        batch_size: int = 8
    ) -> List[tuple]:
        """
        Re-rank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of Document objects to re-rank
            top_k: Number of top documents to return
            batch_size: Batch size for inference
            
        Returns:
            List of (Document, score) tuples sorted by relevance score
        """
        if not documents:
            return []
        
        model = self._load_model()
        
        # Prepare sentence pairs (query, document)
        sentence_pairs = [
            [query, doc.page_content[:1000]]  # Truncate long docs
            for doc in documents
        ]
        
        # Compute relevance scores in batches
        try:
            scores = model.predict(
                sentence_pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"Reranking failed: {e}, returning documents without reranking")
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        # Pair documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def rerank_with_fusion(
        self,
        query: str,
        topic: str,
        documents: List[Document],
        vector_scores: dict = None,
        bm25_scores: dict = None,
        top_k: int = 5,
        rerank_weight: float = 0.6,
        topic_weight: float = 0.2
    ) -> List[Document]:
        """
        Advanced re-ranking with multi-signal fusion:
        - Cross-encoder relevance (query-doc)
        - Topic matching score (topic-doc)
        - Original vector/BM25 scores
        
        Args:
            query: Original user query
            topic: Extracted topic/keywords
            documents: Documents to rerank
            vector_scores: Optional dict of doc_id -> vector similarity score
            bm25_scores: Optional dict of doc_id -> BM25 score
            top_k: Number of results to return
            rerank_weight: Weight for cross-encoder score (0-1)
            topic_weight: Weight for topic matching score (0-1)
            
        Returns:
            List of re-ranked Document objects
        """
        if not documents:
            return []
        
        model = self._load_model()
        
        # Prepare pairs for both query and topic
        query_pairs = [[query, doc.page_content[:1000]] for doc in documents]
        topic_pairs = [[topic, doc.page_content[:1000]] for doc in documents]
        
        try:
            # Get relevance scores
            query_scores = model.predict(query_pairs, batch_size=8, show_progress_bar=False)
            topic_scores = model.predict(topic_pairs, batch_size=8, show_progress_bar=False)
        except Exception as e:
            print(f"Multi-signal reranking failed: {e}, falling back to simple reranking")
            ranked = self.rerank(query, documents, top_k=top_k)
            return [doc for doc, _ in ranked]
        
        # Normalize scores to 0-1 range
        def normalize(scores):
            if len(scores) == 0:
                return scores
            min_s, max_s = np.min(scores), np.max(scores)
            if max_s - min_s < 1e-6:
                return np.ones_like(scores) * 0.5
            return (scores - min_s) / (max_s - min_s)
        
        query_scores_norm = normalize(query_scores)
        topic_scores_norm = normalize(topic_scores)
        
        # Compute combined scores
        combined_scores = []
        for i, doc in enumerate(documents):
            score = 0.0
            
            # Cross-encoder query relevance
            score += rerank_weight * query_scores_norm[i]
            
            # Topic matching
            score += topic_weight * topic_scores_norm[i]
            
            # Original retrieval scores (if available)
            remaining_weight = 1.0 - rerank_weight - topic_weight
            if remaining_weight > 0:
                orig_score = 0.0
                if vector_scores and doc.metadata.get("id") in vector_scores:
                    orig_score += 0.5 * vector_scores[doc.metadata.get("id")]
                if bm25_scores and doc.metadata.get("id") in bm25_scores:
                    orig_score += 0.5 * bm25_scores[doc.metadata.get("id")]
                score += remaining_weight * orig_score
            
            combined_scores.append((doc, score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in combined_scores[:top_k]]