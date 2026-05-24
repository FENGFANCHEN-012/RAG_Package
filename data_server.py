import os
import time
import uuid
import sys
from typing import List, Dict, Any, Optional, Iterable
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import ChatOpenAI
from Embedding import MyHuggingFaceEmbeddings, CrossEncoderReranker

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

load_dotenv()

_INDEX_NAME = "medical-index"
_MEDICAL_TEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "train_pure_text",
    "medical_txt",
)

embeddings = None
_init_error = None
llm = None  # 用于生成 summary 的 LLM


try:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if Pinecone is not None and pinecone_api_key:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(_INDEX_NAME)
        embeddings = MyHuggingFaceEmbeddings(index=index)
        print(f"[Init] Using Pinecone index: {_INDEX_NAME}", file=sys.stderr)
    else:
        embeddings = MyHuggingFaceEmbeddings(index=_INDEX_NAME)
        print("[Init] Using local FAISS fallback (Pinecone unavailable)", file=sys.stderr)

    # 初始化 LLM 用于生成 summary
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="blissful_ishizaka_626/gemma4-cloud",
        temperature=0
    )

    if os.path.exists(_MEDICAL_TEXT_PATH):
        with open(_MEDICAL_TEXT_PATH, "r", encoding="utf-8") as handle:
            medical_content = handle.read()
        docs_for_bm25 = [
            Document(page_content=chunk.strip(), metadata={"id": i, "source": "medical_txt"})
            for i, chunk in enumerate(medical_content.split("\n\n"))
            if chunk.strip()
        ]
        if docs_for_bm25:
            embeddings.build_bm25(docs_for_bm25)
            embeddings.build_faiss_index(docs_for_bm25)
except Exception as exc:
    _init_error = str(exc)


def generate_summary(text: str, max_length: int = 150) -> str:
    """
    使用 LLM 生成文本摘要
    
    Args:
        text: 原始文本
        max_length: 摘要最大长度
        
    Returns:
        str: 生成的摘要
    """
    if not llm:
        # 如果 LLM 不可用，返回截断的文本作为摘要
        return text[:max_length] + "..." if len(text) > max_length else text
    
    try:
        prompt = f"""请为以下文本生成一个简洁的摘要（不超过{max_length}字）：

{text}

摘要："""
        
        response = llm.invoke(prompt)
        summary = response.content.strip()
        
        # 确保摘要不超过最大长度
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    except Exception as e:
        print(f"[Summary Generation] Failed: {e}, using truncated text", file=sys.stderr)
        return text[:max_length] + "..." if len(text) > max_length else text


def _doc_key(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    if metadata.get("id") is not None:
        return f"id:{metadata['id']}"
    source = metadata.get("source", "")
    return f"src:{source}|txt:{hash(doc.page_content)}"


def _to_source_dict(doc: Document) -> dict:
    metadata = getattr(doc, "metadata", {}) or {}
    return {
        "content": doc.page_content,
        "title": str(metadata.get("title", "")),
        "url": str(metadata.get("url", "")),
        "source": str(metadata.get("source", "")),
        "id": metadata.get("id"),
        "metadata": metadata,
    }


def _find_local_doc_by_id(doc_id: str) -> Optional[Document]:
    if not doc_id or not embeddings:
        return None
    for doc in getattr(embeddings, "documents", []) or []:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("id") == doc_id:
            return doc
    for doc in getattr(embeddings, "document_store", []) or []:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("id") == doc_id:
            return doc
    return None


def _fetch_doc_by_id(doc_id: str) -> Optional[Document]:
    """Try local store first, then Pinecone fetch."""
    doc = _find_local_doc_by_id(doc_id)
    if doc:
        return doc
    # Try Pinecone fetch
    if embeddings and getattr(embeddings, "index", None):
        try:
            result = embeddings.index.fetch(ids=[doc_id])
            vectors = getattr(result, "vectors", {}) or {}
            if doc_id in vectors:
                vector = vectors[doc_id]
                metadata = getattr(vector, "metadata", {}) or {}
                text = metadata.get("text", "")
                if text:
                    return Document(page_content=text, metadata=dict(metadata))
        except Exception as e:
            print(f"[Fetch] Pinecone fetch failed for {doc_id}: {e}", file=sys.stderr)
    return None


def reciprocal_rank_fusion(rank_lists: Iterable[List[Document]], k: int = 60) -> List[Document]:
    fused_scores = defaultdict(float)
    doc_lookup = {}

    for ranked_docs in rank_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _doc_key(doc)
            # every doc has its own score
            fused_scores[key] += 1.0 / (k + rank)
            if key not in doc_lookup:
                doc_lookup[key] = doc

    return sorted(
        doc_lookup.values(),
        key=lambda d: fused_scores[_doc_key(d)],
        reverse=True,
    )


def get_docs_for_queries(
    queries: List[str],
    topic: str = None,
    top_k: int = 5,
    per_retriever_k: int = 12,
    rrf_k: int = 60,
    enable_rerank: bool = True,
    rerank_weight: float = 0.5,
    topic_weight: float = 0.3,
    bm25_weight: float = 0.5,
    use_summary_search: bool = False,
    return_scores: bool = False,
) -> List[dict] | tuple:
    """
    Advanced hybrid retrieval with topic-based search and CrossEncoder reranking.
    
    Args:
        queries: List of search queries (expanded from original question)
        topic: Extracted topic/keywords for focused BM25 search
        top_k: Number of final results to return
        per_retriever_k: Number of candidates per retriever
        rrf_k: RRF fusion parameter
        enable_rerank: Whether to use CrossEncoder reranking
        rerank_weight: Weight for cross-encoder query-doc relevance (0-1)
        topic_weight: Weight for topic-doc matching (0-1)
        bm25_weight: Weight for BM25 keyword search vs vector search (0-1). 
                     Higher values favor keyword search (e.g., 0.8 for definition queries).
        use_summary_search: Whether to use summary-based retrieval (search summaries, return full docs)
        return_scores: Whether to return (docs, avg_score) tuple instead of just docs
    """
    if not embeddings:
        if _init_error:
            print(f"[WARN] Retrieval is unavailable: {_init_error}", file=sys.stderr)
        if return_scores:
            return [], 0.0
        return []

    # 如果启用 summary 搜索
    if use_summary_search:
        result = get_docs_by_summary_search(
            queries=queries,
            topic=topic,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            rrf_k=rrf_k,
            enable_rerank=enable_rerank,
            rerank_weight=rerank_weight,
            topic_weight=topic_weight,
        )
        if return_scores:
            # Summary search doesn't have scores, return 0.5 as default
            return result, 0.5
        return result

    # 原有的检索逻辑
    # Collect all candidate documents with their scores
    all_docs = {}
    vector_scores = {}
    bm25_scores = {}
    
    # Phase 1: Multi-query retrieval (original queries)
    for raw_query in queries:
        query = str(raw_query).strip()
        if not query:
            continue

        # Vector search with original query
        if embeddings.vector_store:
            try:
                vector_results = embeddings.vector_store.similarity_search(query, k=per_retriever_k)
                print(f"[DEBUG] Vector search returned {len(vector_results)} results for query '{query[:30]}...'", file=sys.stderr)
                for rank, doc in enumerate(vector_results):
                    key = _doc_key(doc)
                    if key not in all_docs:
                        all_docs[key] = doc
                    # Use RRF scoring for vector results
                    vector_scores[key] = vector_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
                print(f"[DEBUG] vector_scores now has {len(vector_scores)} entries", file=sys.stderr)
            except Exception as e:
                print(f"Vector search failed for query '{query[:30]}...': {e}", file=sys.stderr)

        # Keyword search with original query
        try:
            keyword_results = embeddings.bm25_search(query, k=per_retriever_k)
            for rank, doc in enumerate(keyword_results):
                key = _doc_key(doc)
                if key not in all_docs:
                    all_docs[key] = doc
                # Use RRF scoring for BM25 results, scaled by bm25_weight
                bm25_scores[key] = bm25_scores.get(key, 0) + bm25_weight / (rrf_k + rank + 1)
        except Exception as e:
            print(f"BM25 search failed for query '{query[:30]}...': {e}")

    # Phase 2: Topic-focused BM25 search (if topic provided)
    if topic and topic.strip():
        try:
            topic_results = embeddings.bm25_search(topic.strip(), k=per_retriever_k)
            for rank, doc in enumerate(topic_results):
                key = _doc_key(doc)
                if key not in all_docs:
                    all_docs[key] = doc
                # Boost BM25 scores for topic matches
                topic_boost = 1.5  # Higher weight for topic matches
                bm25_scores[key] = bm25_scores.get(key, 0) + topic_boost / (rrf_k + rank + 1)
        except Exception as e:
            print(f"Topic search failed for '{topic[:30]}...': {e}")

    if not all_docs:
        return []

    # Convert to list for reranking
    candidate_docs = list(all_docs.values())
    
    # Phase 3: CrossEncoder reranking with multi-signal fusion
    if enable_rerank and candidate_docs:
        try:
            # Get the primary query (first one) for reranking
            primary_query = queries[0] if queries else ""
            
            # Initialize reranker (singleton)
            reranker = CrossEncoderReranker.get_instance()
            
            # Use advanced reranking with topic fusion
            reranked_docs = reranker.rerank_with_fusion(
                query=primary_query,
                topic=topic or primary_query,
                documents=candidate_docs,
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                top_k=top_k,
                rerank_weight=rerank_weight,
                topic_weight=topic_weight,
            )
            
            final_docs = reranked_docs[:top_k]
            # Extract relevance scores from reranked documents
            avg_score = 0.0
            if final_docs and hasattr(final_docs[0], 'metadata'):
                scores = [doc.metadata.get('_relevance_score', 0.5) for doc in final_docs if hasattr(doc, 'metadata')]
                avg_score = sum(scores) / len(scores) if scores else 0.5
            print(f"[INFO] Hybrid search: {len(candidate_docs)} candidates, returned {len(final_docs)} results (reranked), avg_score: {avg_score:.3f}", file=sys.stderr)
            
        except Exception as e:
            print(f"Reranking failed: {e}, falling back to RRF fusion", file=sys.stderr)
            # Fallback to RRF fusion
            fused_results = reciprocal_rank_fusion(
                [[all_docs[k] for k in vector_scores.keys()], 
                 [all_docs[k] for k in bm25_scores.keys()]],
                k=rrf_k
            )[:top_k]
            final_docs = fused_results
            avg_score = 0.5  # Default score for fallback
            print(f"[INFO] Hybrid search: {len(candidate_docs)} candidates, returned {len(final_docs)} results (RRF fallback), avg_score: {avg_score:.3f}", file=sys.stderr)
    else:
        print(f"[DEBUG] Before RRF fusion: vector_scores={len(vector_scores)}, bm25_scores={len(bm25_scores)}", file=sys.stderr)
        rank_lists = []
        if vector_scores:
            rank_lists.append([all_docs[k] for k in vector_scores.keys()])
        if bm25_scores:
            rank_lists.append([all_docs[k] for k in bm25_scores.keys()])
        print(f"[DEBUG] RRF fusion with {len(rank_lists)} rank lists", file=sys.stderr)
        fused_results = reciprocal_rank_fusion(rank_lists, k=rrf_k)[:top_k]
        final_docs = fused_results
        avg_score = 0.5  # Default score for non-reranked results
        print(f"[INFO] Hybrid search: {len(candidate_docs)} candidates, returned {len(final_docs)} results (RRF fusion), avg_score: {avg_score:.3f}", file=sys.stderr)

    source_dicts = [_to_source_dict(doc) for doc in final_docs]
    if return_scores:
        return source_dicts, avg_score
    return source_dicts


def get_docs_by_summary_search(
    queries: List[str],
    topic: str = None,
    top_k: int = 5,
    per_retriever_k: int = 12,
    rrf_k: int = 60,
    enable_rerank: bool = True,
    rerank_weight: float = 0.5,
    topic_weight: float = 0.3,
    summary_weight: float = 0.8,  # Increased to favor summary search for better relevance
) -> List[dict]:
    """
    摘要优先检索：只搜索摘要，再根据 full_doc_id 返回完整文档
    
    策略：
    1. 对 is_summary=True 的摘要进行向量搜索
    2. 使用摘要 metadata.full_doc_id 定位完整文档
    3. 按摘要命中分数排序，返回完整文档
    
    Args:
        queries: List of search queries
        topic: Extracted topic/keywords
        top_k: Number of final results to return
        per_retriever_k: Number of candidates per retriever
        rrf_k: RRF fusion parameter
        enable_rerank: Whether to use CrossEncoder reranking
        rerank_weight: Weight for cross-encoder query-doc relevance
        topic_weight: Weight for topic-doc matching
        summary_weight: Weight for summary search vs full doc search (0-1). 
                       Higher values favor summary search (e.g., 0.6).
        
    Returns:
        List[dict]: 完整文档列表
    """
    if not embeddings:
        if _init_error:
            print(f"[WARN] Retrieval is unavailable: {_init_error}", file=sys.stderr)
        return []

    # 收集摘要命中的评分
    all_docs = {}
    summary_scores = {}
    
    # Phase 1: 对摘要进行向量搜索
    for raw_query in queries:
        query = str(raw_query).strip()

        # 如果是空的就跳过
        if not query:
            continue

        if embeddings.vector_store:
            try:
                # 搜索摘要 (is_summary=True)
                #通过similarity seach
                summary_results = embeddings.vector_store.similarity_search(
                    query,
                    k=per_retriever_k,
                    filter={"is_summary": True}
                )
                # 每个document 返回一个rank
                for rank, doc in enumerate(summary_results):
                    metadata = getattr(doc, "metadata", {}) or {}
                    full_doc_id = metadata.get("full_doc_id")

                    if full_doc_id:
                        # 使用 full_doc_id 作为 key，以便后续融合
                        key = f"doc:{full_doc_id}"
                        if key not in all_docs:
                            # 暂时存储摘要文档，后续会替换为完整文档
                            all_docs[key] = doc
                        # 使用 RRF 评分，乘以 summary_weight
                        summary_scores[key] = summary_scores.get(key, 0) + summary_weight / (rrf_k + rank + 1)
            except Exception as e:
                print(f"[Hybrid Search] Summary search failed for query '{query[:30]}...': {e}")

    if not all_docs:
        print("[Hybrid Search] No documents found, falling back to regular search")
        return get_docs_for_queries(
            queries=queries,
            topic=topic,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            rrf_k=rrf_k,
            enable_rerank=enable_rerank,
            use_summary_search=False
        )

    print(f"[Summary Search] Found {len(all_docs)} summary matches")

    # Phase 2: 使用摘要命中分数排序
    fused_scores = {}
    for key in all_docs:
        fused_scores[key] = summary_scores.get(key, 0)

    # Phase 3: 根据摘要的 full_doc_id 获取完整文档
    final_docs = {}
    for key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        doc = all_docs[key]
        metadata = getattr(doc, "metadata", {}) or {}

        full_doc_id = metadata.get("full_doc_id") or metadata.get("id")
        full_doc = _fetch_doc_by_id(full_doc_id)
        final_docs[key] = full_doc or doc

    # Phase 4: 重排序
    candidate_docs = list(final_docs.values())
    
    if enable_rerank and candidate_docs:
        try:
            primary_query = queries[0] if queries else ""
            reranker = CrossEncoderReranker.get_instance()
            
            reranked_docs = reranker.rerank_with_fusion(
                query=primary_query,
                topic=topic or primary_query,
                documents=candidate_docs,
                vector_scores=fused_scores,
                bm25_scores={},
                top_k=top_k,
                rerank_weight=rerank_weight,
                topic_weight=topic_weight,
            )
            
            final_docs = reranked_docs[:top_k]
            print(f"[Summary Search] Reranked {len(candidate_docs)} docs, returned {len(final_docs)}")
        except Exception as e:
            print(f"[Summary Search] Reranking failed: {e}")
            # 根据融合分数排序
            final_docs = sorted(candidate_docs, key=lambda d: fused_scores.get(f"doc:{_doc_key(d)}", 0), reverse=True)[:top_k]
    else:
        # 根据融合分数排序
        final_docs = sorted(candidate_docs, key=lambda d: fused_scores.get(f"doc:{_doc_key(d)}", 0), reverse=True)[:top_k]
    
    return [_to_source_dict(doc) for doc in final_docs]


def get_docs_for_query(
    query: str,
    topic: str = None,
    top_k: int = 5,
    per_retriever_k: int = 12,
    rrf_k: int = 60,
    enable_rerank: bool = True,
    bm25_weight: float = 0.5,
) -> List[dict]:
    return get_docs_for_queries(
        [query],
        topic=topic,
        top_k=top_k,
        per_retriever_k=per_retriever_k,
        rrf_k=rrf_k,
        enable_rerank=enable_rerank,
        bm25_weight=bm25_weight,
    )


def _chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Simple character-based chunking."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _chunk_by_markdown_headers(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split markdown text respecting headers:
    - # and ## headers are preserved as chunk boundaries (don't split in the middle)
    - If a section is larger than chunk_size, it will be further split
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    # Split by # and ## headers (preserve these boundaries)
    import re
    # Pattern matches # or ## at the start of a line
    header_pattern = r'^(#{1,2}\s+.+)$'
    
    lines = cleaned.split('\n')
    sections = []
    current_section = []
    
    for line in lines:
        if re.match(header_pattern, line.strip()):
            # Found a # or ## header, save current section and start new one
            if current_section:
                sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Now process each section: if it's too large, split it further
    final_chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if len(section) <= chunk_size:
            # Section is small enough, keep as is
            final_chunks.append(section)
        else:
            # Section is too large, need to split it
            # But we try to find good break points (### headers or paragraph boundaries)
            sub_chunks = _split_large_section(section, chunk_size, chunk_overlap)
            final_chunks.extend(sub_chunks)
    
    return final_chunks if final_chunks else _chunk_text(cleaned, chunk_size, chunk_overlap)


def _split_large_section(section: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split a large section into smaller chunks.
    Try to find natural break points like ### headers or paragraph boundaries.
    """
    import re
    
    # If section fits in one chunk, return it
    if len(section) <= chunk_size:
        return [section]
    
    # Try to split by ### headers first (these are not preserved like #/##)
    header3_pattern = r'^###\s+.+$'
    lines = section.split('\n')
    
    sub_sections = []
    current_sub = []
    
    for line in lines:
        if re.match(header3_pattern, line.strip()):
            if current_sub:
                sub_sections.append('\n'.join(current_sub))
            current_sub = [line]
        else:
            current_sub.append(line)
    
    if current_sub:
        sub_sections.append('\n'.join(current_sub))
    
    # Now check each sub-section
    final_chunks = []
    for sub in sub_sections:
        sub = sub.strip()
        if not sub:
            continue
            
        if len(sub) <= chunk_size:
            final_chunks.append(sub)
        else:
            # Still too large, use character-based chunking
            char_chunks = _chunk_text(sub, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            final_chunks.extend(char_chunks)
    
    return final_chunks if final_chunks else _chunk_text(section, chunk_size, chunk_overlap)


def _chunk_by_headers(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text by headers (look for markdown-style headers)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    lines = cleaned.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        is_header = line.strip().startswith(('#', '##', '###', '####', '=', '-')) and line.strip()
        
        # If adding this line would exceed chunk_size and we have content
        if current_size + line_size > chunk_size and current_chunk:
            # Check if this line is a header
            if is_header and current_chunk:
                # Save current chunk and start new one with header
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    # Add remaining chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks if chunks else _chunk_text(cleaned, chunk_size, chunk_overlap)


def upload_text_to_vector_store(
    text: str,
    source: str = "manual_upload",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    split_method: str = "character",
    generate_summaries: bool = True,
) -> dict:
    """
    上传文本到本地 FAISS 向量索引
    
    Args:
        text: 要上传的文本
        source: 来源标识
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        split_method: 分块方法
        generate_summaries: 是否为每个 chunk 生成摘要
        
    Returns:
        dict: 上传结果
    """
    if not embeddings:
        reason = _init_error or "Embeddings are not initialized"
        raise RuntimeError(f"Vector upload is unavailable: {reason}")

    # Select chunking method
    if split_method == "markdown_headers":
        chunks = _chunk_by_markdown_headers(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif split_method == "headers":
        chunks = _chunk_by_headers(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:  # default to character-based
        chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Text is empty after trimming")

    upload_ts = int(time.time())
    document_id = f"{source}-{uuid.uuid4().hex[:12]}"  # 文档的唯一 ID
    
    # 为每个 chunk 生成摘要（如果启用）
    summaries = []
    if generate_summaries:
        print(f"[Upload] Generating summaries for {len(chunks)} chunks...", file=sys.stderr)
        for idx, chunk in enumerate(chunks):
            summary = generate_summary(chunk, max_length=150)
            summaries.append(summary)
            print(f"[Upload] Summary {idx + 1}/{len(chunks)}: {summary[:50]}...", file=sys.stderr)
    
    # 准备完整文档的数据
    full_doc_ids = [f"{document_id}-full-{idx}" for idx in range(len(chunks))]
    full_doc_metadatas = [
        {
            "source": source,
            "chunk_index": idx,
            "uploaded_at": upload_ts,
            "document_id": document_id,
            "is_summary": False,
            "summary_id": f"{document_id}-summary-{idx}" if generate_summaries else None,
        }
        for idx in range(len(chunks))
    ]
    
    # 准备摘要的数据（如果生成）
    summary_ids = []
    summary_metadatas = []
    if generate_summaries:
        summary_ids = [f"{document_id}-summary-{idx}" for idx in range(len(summaries))]
        summary_metadatas = [
            {
                "source": source,
                "chunk_index": idx,
                "uploaded_at": upload_ts,
                "document_id": document_id,
                "is_summary": True,
                "full_doc_id": f"{document_id}-full-{idx}",
            }
            for idx in range(len(summaries))
        ]
    
    embeddings.vector_store.add_texts(texts=chunks, metadatas=full_doc_metadatas, ids=full_doc_ids)

    if generate_summaries and summaries:
        embeddings.vector_store.add_texts(texts=summaries, metadatas=summary_metadatas, ids=summary_ids)

    return {
        "success": True,
        "chunks_uploaded": len(chunks),
        "summaries_generated": len(summaries) if generate_summaries else 0,
        "document_id": document_id,
        "source": source,
        "split_method": split_method,
        "full_doc_ids": full_doc_ids,
        "summary_ids": summary_ids,
    }
