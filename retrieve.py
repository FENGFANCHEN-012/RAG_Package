from Embedding import MyHuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from collections import defaultdict

import call_AI
import query_processor

# Load environment variables
load_dotenv()

from langchain_core.documents import Document

# ... (other imports)

index_name = "local-faiss-index"
embeddings = MyHuggingFaceEmbeddings(index=index_name)

with open("c:/Users/billychen/Desktop/Transformer/train_pure_text/medical_txt", "r", encoding="utf-8") as f:
    medical_content = f.read()

# Simple chunking for demo
docs = [Document(page_content=chunk.strip(), metadata={"id": i}) 
        for i, chunk in enumerate(medical_content.split("\n\n")) if chunk.strip()]
embeddings.build_bm25(docs)
embeddings.build_faiss_index(docs)

#用户问题
question = "糖尿病，应该注意什么?"

# ... (rest of the file)

#问题向量化
question_embedding = embeddings.embed_query(question)
print(f"Question embedding length: {len(question_embedding)}")
print(f"First 5 values: {question_embedding[:5]}")



# 使用多查询 + RRF 融合向量检索与 BM25 检索
def _doc_key(doc):
    """Create a stable key so the same document from different retrievers can be fused."""
    metadata = getattr(doc, "metadata", {}) or {}
    if metadata.get("id") is not None:
        return f"id:{metadata['id']}"
    source = metadata.get("source", "")
    return f"src:{source}|txt:{hash(doc.page_content)}"




def reciprocal_rank_fusion(rank_lists, k=60):
    """RRF: score(doc) = sum(1 / (k + rank_i))."""
    fused_scores = defaultdict(float)
    doc_lookup = {}

    for ranked_docs in rank_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _doc_key(doc)
            fused_scores[key] += 1.0 / (k + rank)
            if key not in doc_lookup:
                doc_lookup[key] = doc

    final_docs = sorted(
        doc_lookup.values(),
        key=lambda d: fused_scores[_doc_key(d)],
        reverse=True,
    )
    return final_docs


def hybrid_search_with_multi_query_rrf(query, top_k=3, per_retriever_k=10, rrf_k=60, enable_complex_detection=True):
    """
    1) Optionally detect complex query and generate a composed query
    2) Expand the (processed) query into multiple sub-queries
    3) Retrieve from vector store + BM25 for each sub-query
    4) Fuse all ranked lists with Reciprocal Rank Fusion
    """
    # Step 1: Complexity detection and composed query generation (if enabled)
    processed_query = query
    if enable_complex_detection:
        processed_query = query_processor.process_user_input(query, enable_complex_detection=True)
        # If the query was complex, we may want to log or use the composed query
        if processed_query != query:
            print(f"[INFO] Complex query detected, using composed query: {processed_query}")
    
    # Step 2: Multi-query expansion
    expanded_queries = generate_multi_queries(processed_query)
    all_rank_lists = []

    # Step 3: Retrieve for each expanded query
    for q in expanded_queries:
        if embeddings.vector_store:
            try:
                vector_results = embeddings.vector_store.similarity_search(q, k=per_retriever_k)
            except Exception as e:
                print(f"Vector search failed for query '{q}...': {e}", file=sys.stderr)
                vector_results = []
        else:
            vector_results = []
        keyword_results = embeddings.bm25_search(q, k=per_retriever_k)
        if vector_results:
            all_rank_lists.append(vector_results)
        if keyword_results:
            all_rank_lists.append(keyword_results)

    if not all_rank_lists:
        return []

    # Step 4: Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(all_rank_lists, k=rrf_k)
    return fused_results[:top_k]





#向量检索（找最相似的3个）
# results = embeddings.vector_store.similarity_search_by_vector(
#     embedding=question_embedding,
#     k=3  # 返回top3
# )

results = hybrid_search_with_multi_query_rrf(question, top_k=3, per_retriever_k=10, rrf_k=60)
print(f"Retrieved docs: {len(results)}")




# 重新排序

from sentence_transformers import CrossEncoder

#加载重排序模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#对检索结果重新打分

candidate_docs = ["文档1内容", "文档2内容", "文档3内容"]

scores = reranker.predict([(question, doc) for doc in candidate_docs])

#按分数排序
ranked_docs = [doc for _, doc in sorted(zip(scores, candidate_docs), reverse=True)]
from call_AI import CallAI

if not results:
    print("No related documents found. Try a different question or lower retrieval threshold.")
else:
    call_ai = CallAI(question, [doc.page_content for doc in results])
    print(call_ai.call_ollama())
