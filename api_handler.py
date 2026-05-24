"""
 ============================================================================
 Python API 处理器 (api_handler.py)
 ============================================================================
 
 功能描述：
 - Python 后端主入口，处理来自 Node.js 的所有 API 请求
 - 路由请求到不同的处理器（RAG、数据库查询、意图分类等）
 - 集成向量检索、AI 调用、数据库查询等功能
 - 支持多阶段回退机制（文档检索 -> AI 生成 -> 网络搜索）
 - 支持混合搜索（元数据过滤 + 语义检索）
 
 主要处理器：
 - handle_rag(): RAG 检索增强生成，支持多阶段回退
 - handle_query(): 普通 LLM 查询
 - handle_deepthink(): 深度思考模式，暂未推出
 - handle_upload(): 上传文本到向量数据库
 - handle_database_query(): 数据库查询结果处理
 - handle_classify_intent(): 意图分类
 - handle_generate_sql(): SQL 生成
 - handle_hybrid_search(): 混合搜索（元数据过滤 + 语义检索）
 - handle_smart_query(): 智能查询（自动路由到 SQL/语义/混合搜索）
 
 架构：
 Node.js -> api_handler.py -> 各个处理器 -> 外部服务（Ollama、Vector Store、MySQL）
 
 混合搜索流程：
 1. 用户查询 -> 查询路由器（LLM + 意图分类器）
 2. 判断查询类型：sql_search / semantic_search / hybrid
 3. 根据类型路由到对应处理器
 4. 返回结果
 
 依赖：
 - call_AI.py: AI 调用封装
 - query_processor.py: 查询处理和扩展
 - query_router.py: 查询路由器（LLM + 意图分类器）
 - hybrid_search.py: 混合搜索处理器
 - sql_generator.py: SQL 生成器（含元数据过滤器）
 - data_server.py: 向量检索服务
 - local FAISS vector store: 向量数据库
 - openai: OpenAI 兼容客户端（用于 Ollama）
 
 ============================================================================
"""

import json
import os
import re
import sys
from typing import Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI


import query_processor
from call_AI import CallAI
from data_server import get_docs_for_queries, get_docs_by_summary_search, upload_text_to_vector_store
from query_router import QueryRouter
from hybrid_search import HybridSearchProcessor
from sql_generator import SQLGenerator


# change the model based on the scenario
DEFAULT_MODEL = "blissful_ishizaka_626/gemma4-cloud"

# use small model for query processing to save resources
# CRAG_EVALUATOR_MODEL = "google/t5-large-crag-evaluator"  # Model not found on HuggingFace
# Alternative: use FLAN-T5 for relevance evaluation
# CRAG_EVALUATOR_MODEL = "google/flan-t5-large"  # Disabled - using LLM prompt classification instead
CRAG_EVALUATOR_MODEL = None

# Self-RAG preference weights
SELF_RAG_PREFERENCE_WEIGHTS = {
    'relevance': 1.0,
    'support': 1.5,
    'utility': 2.0
}

# Self-RAG score threshold for retry
SELF_RAG_SCORE_THRESHOLD = 3.0

# Default retrieval parameters
DEFAULT_TOP_K = 8  # 增加检索文档数量
DEFAULT_PER_RETRIEVER_K = 16

_crag_tokenizer = None
_crag_model = None
_crag_device = None


def evaluate_relevance(query: str, docs: List, model: str = DEFAULT_MODEL) -> float:
    """
    评估检索文档与查询的相关性分数（0-1）
    使用 LLM 评估文档相关性
    """
    if not docs:
        return 0.0
    
    try:
        # 取前3个文档进行评估
        sample_docs = docs[:3]
        docs_text = "\n\n".join([_doc_to_text(doc) for doc in sample_docs])
        
        prompt = f"""请评估以下文档与查询的相关性，返回0到1之间的分数。

查询: {query}

文档:
{docs_text}

只返回一个数字（0-1之间的小数），不要其他文字："""
        
        result = call_direct_llm(prompt, model=model, max_tokens=10)
        score = float(result.strip())
        return max(0.0, min(1.0, score))  # 确保在0-1范围内
    except Exception as e:
        print(f"[Self-RAG] Relevance evaluation failed: {e}", file=sys.stderr)
        return 0.5  # 默认中等分数


def extract_support_level(docs: List) -> float:
    """
    评估文档的支持度（0-1）
    检查文档是否包含足够的事实依据
    """
    if not docs:
        return 0.0
    
    try:
        # 简单的启发式：检查文档长度和结构
        total_chars = sum(len(_doc_to_text(doc)) for doc in docs)
        avg_chars = total_chars / len(docs)
        
        # 基于文档长度的简单评分
        if avg_chars > 500:
            return 1.0
        elif avg_chars > 200:
            return 0.7
        elif avg_chars > 100:
            return 0.5
        else:
            return 0.3
    except Exception as e:
        print(f"[Self-RAG] Support level extraction failed: {e}", file=sys.stderr)
        return 0.5


def extract_utility_score(query: str, docs: List, model: str = DEFAULT_MODEL) -> float:
    """
    评估文档的有用性分数（0-1）
    使用 LLM 评估文档对回答查询的有用性
    """
    if not docs:
        return 0.0
    
    try:
        sample_docs = docs[:3]
        docs_text = "\n\n".join([_doc_to_text(doc) for doc in sample_docs])
        
        prompt = f"""请评估以下文档对回答用户查询的有用性，返回0到1之间的分数。

查询: {query}

文档:
{docs_text}

只返回一个数字（0-1之间的小数），不要其他文字："""
        
        result = call_direct_llm(prompt, model=model, max_tokens=10)
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[Self-RAG] Utility score extraction failed: {e}", file=sys.stderr)
        return 0.5


def evaluate_retrieval_quality_with_reasoning_tokens(
    query: str,
    docs: List,
    preference_weights: dict = None,
    model: str = DEFAULT_MODEL
) -> dict:
    """
    使用推理令牌评估检索质量
    
    让 LLM 输出推理令牌 [Retrieve=Yes/No]、[ISSUP=High/Med/Low]、[ISUSE=High/Med/Low]
    然后基于令牌计算综合分数
    
    Returns:
        dict: {
            'total_score': float,
            'relevance_score': float,
            'support_score': float,
            'utility_score': float,
            'is_satisfactory': bool,
            'reasoning_tokens': dict
        }
    """
    if preference_weights is None:
        preference_weights = SELF_RAG_PREFERENCE_WEIGHTS
    
    if not docs:
        return {
            'total_score': 0.0,
            'relevance_score': 0.0,
            'support_score': 0.0,
            'utility_score': 0.0,
            'is_satisfactory': False,
            'reasoning_tokens': {}
        }
    
    try:
        # 取前3个文档进行评估
        sample_docs = docs[:3]
        docs_text = "\n\n".join([f"文档{i+1}: {_doc_to_text(doc)}" for i, doc in enumerate(sample_docs)])
        
        prompt = f"""请评估以下检索文档的质量，使用推理令牌标记。

查询: {query}

检索文档:
{docs_text}

请输出推理令牌：
1. [Retrieve=Yes/No] - 是否需要重新检索（Yes表示文档质量差，需要重新检索）
2. [ISSUP=High/Med/Low] - 文档支持度（High=高支持度，Med=中等，Low=低支持度）
3. [ISUSE=High/Med/Low] - 文档有用性（High=高有用性，Med=中等，Low=低有用性）

只输出三个推理令牌，格式如下：
[Retrieve=Yes/No] [ISSUP=High/Med/Low] [ISUSE=High/Med/Low]"""
        
        result = call_direct_llm(prompt, model=model, max_tokens=50)
        
        # 解析推理令牌
        reasoning_tokens = {}
        if '[Retrieve=Yes]' in result:
            reasoning_tokens['retrieve'] = 'Yes'
        elif '[Retrieve=No]' in result:
            reasoning_tokens['retrieve'] = 'No'
        
        if '[ISSUP=High]' in result:
            reasoning_tokens['support'] = 'High'
        elif '[ISSUP=Med]' in result:
            reasoning_tokens['support'] = 'Med'
        elif '[ISSUP=Low]' in result:
            reasoning_tokens['support'] = 'Low'
        
        if '[ISUSE=High]' in result:
            reasoning_tokens['utility'] = 'High'
        elif '[ISUSE=Med]' in result:
            reasoning_tokens['utility'] = 'Med'
        elif '[ISUSE=Low]' in result:
            reasoning_tokens['utility'] = 'Low'
        
        # 将令牌转换为分数
        relevance_score = 0.0 if reasoning_tokens.get('retrieve') == 'Yes' else 0.8
        
        support_score_map = {'High': 1.0, 'Med': 0.6, 'Low': 0.3}
        support_score = support_score_map.get(reasoning_tokens.get('support', 'Med'), 0.6)
        
        utility_score_map = {'High': 1.0, 'Med': 0.6, 'Low': 0.3}
        utility_score = utility_score_map.get(reasoning_tokens.get('utility', 'Med'), 0.6)
        
        # 计算综合分数
        total_score = (
            preference_weights['relevance'] * relevance_score +
            preference_weights['support'] * support_score +
            preference_weights['utility'] * utility_score
        )
        
        print(f"[Self-RAG] 推理令牌: {reasoning_tokens} - Relevance: {relevance_score:.2f}, Support: {support_score:.2f}, Utility: {utility_score:.2f}, Total: {total_score:.2f}", file=sys.stderr)
        
        return {
            'total_score': total_score,
            'relevance_score': relevance_score,
            'support_score': support_score,
            'utility_score': utility_score,
            'is_satisfactory': total_score >= SELF_RAG_SCORE_THRESHOLD,
            'reasoning_tokens': reasoning_tokens
        }
    except Exception as e:
        print(f"[Self-RAG] 推理令牌评估失败: {e}", file=sys.stderr)
        # 回退到简单评估
        return {
            'total_score': 2.0,
            'relevance_score': 0.5,
            'support_score': 0.5,
            'utility_score': 0.5,
            'is_satisfactory': False,
            'reasoning_tokens': {}
        }


def self_rag_retrieval_with_retry(
    query: str,
    retrieve_func,
    max_retries: int = 3,
    preference_weights: dict = None,
    model: str = DEFAULT_MODEL,
    **retrieve_kwargs
) -> tuple:
    """
    Self-RAG 检索机制，基于推理令牌评估检索质量并自动重试
    
    Args:
        query: 用户查询
        retrieve_func: 检索函数（如 retrieve_semantic_docs）
        max_retries: 最大重试次数
        preference_weights: 偏好权重
        model: LLM 模型
        **retrieve_kwargs: 传递给检索函数的参数
    
    Returns:
        tuple: (docs, quality_evaluation, retry_count)
    """
    max_retries = max(1, min(int(max_retries), 3))
    retry_count = 0
    best_docs = None
    best_score = 0.0
    
    while retry_count < max_retries:
        print(f"[Self-RAG] 检索尝试 {retry_count + 1}/{max_retries}", file=sys.stderr)
        
        try:
            # 调用检索函数
            retrieval_result = retrieve_func(query, model=model, **retrieve_kwargs)
            if isinstance(retrieval_result, tuple):
                docs, avg_score = retrieval_result
            else:
                docs = retrieval_result
                avg_score = 0.0
            
            if not docs:
                print(f"[Self-RAG] 未检索到文档", file=sys.stderr)
                retry_count += 1
                continue
            
            # 使用推理令牌评估检索质量
            quality = evaluate_retrieval_quality_with_reasoning_tokens(query, docs, preference_weights, model)
            
            retrieve_decision = quality.get('reasoning_tokens', {}).get('retrieve')

            # 如果推理令牌显示 [Retrieve=No] 且分数满意，返回结果
            if quality['is_satisfactory'] and retrieve_decision == 'No':
                print(f"[Self-RAG] 检索质量满意，推理令牌: {quality['reasoning_tokens']}, 分数: {quality['total_score']:.2f}", file=sys.stderr)
                return docs, quality, retry_count
            
            # 保存最佳结果
            if quality['total_score'] > best_score:
                best_docs = docs
                best_score = quality['total_score']
            
            if quality['is_satisfactory'] and retrieve_decision != 'Yes':
                print(f"[Self-RAG] 检索分数已达标，返回当前最佳文档，推理令牌: {quality.get('reasoning_tokens', {})}, 分数: {quality['total_score']:.2f}", file=sys.stderr)
                return docs, quality, retry_count

            if retry_count >= max_retries - 1:
                break

            # 如果推理令牌显示 [Retrieve=Yes] 或分数太低，调整检索参数重试
            if quality.get('reasoning_tokens', {}).get('retrieve') == 'Yes':
                print(f"[Self-RAG] 推理令牌建议重新检索 [Retrieve=Yes]，准备重试", file=sys.stderr)
            elif not quality['is_satisfactory']:
                print(f"[Self-RAG] 检索质量不满意（{quality['total_score']:.2f} < {SELF_RAG_SCORE_THRESHOLD}），准备重试", file=sys.stderr)
            else:
                # 质量满意但推理令牌建议重试，也重试
                print(f"[Self-RAG] 推理令牌建议重新检索，准备重试", file=sys.stderr)
            
            # 调整检索参数（增加 top_k）
            if 'top_k' in retrieve_kwargs:
                retrieve_kwargs['top_k'] = min(retrieve_kwargs['top_k'] + 3, 20)
            if 'per_retriever_k' in retrieve_kwargs:
                retrieve_kwargs['per_retriever_k'] = min(retrieve_kwargs['per_retriever_k'] + 5, 30)
            
            retry_count += 1
            
        except Exception as e:
            print(f"[Self-RAG] 检索失败: {e}", file=sys.stderr)
            retry_count += 1
    
    # 达到最大重试次数，返回最佳结果
    print(f"[Self-RAG] 达到最大重试次数，返回最佳结果（分数: {best_score:.2f}）", file=sys.stderr)
    return best_docs or [], {'total_score': best_score, 'is_satisfactory': False}, retry_count



def call_direct_llm(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 2048, conversation_history=None) -> str:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
   
    messages = [
        {"role": "system", "content": "你是一个有帮助的问答助手。可以使用通用知识回答。直接输出最终答案，不要输出思考过程,可以结合用户的历史输入回答。"}
    ]

    for msg in (conversation_history or [])[-10:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_cli_json_args(raw_args: list[str]) -> dict:
    args_json = " ".join(raw_args).strip()
    if not args_json:
        return {"type": "rag", "question": "什么是机器学习"}

    try:
        return json.loads(args_json)
    except json.JSONDecodeError:
        repaired = args_json
        repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
        repaired = re.sub(r":\s*([^{}\[\],\"']+)(\s*[,}])", lambda m: ': "' + m.group(1).strip() + '"' + m.group(2), repaired)
        repaired = repaired.replace("'", '"')
        return json.loads(repaired)


# web search for fallback
def search_web(query: str) -> str:
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")

        # required api key
        if not api_key:
            print("[WEB] TAVILY_API_KEY is not configured", file=sys.stderr)
            return ""

        print(f"[WEB] Searching with Tavily for: {query}", file=sys.stderr)
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_answer=True,
        )

        results = []
        answer = response.get("answer")
        if answer:
            results.append(f"网络搜索答案: {answer}")

        for index, item in enumerate(response.get("results", [])[:3], start=1):
            title = item.get("title") or f"搜索结果 {index}"
            url = item.get("url") or ""
            content = item.get("content") or ""
            if content:
                results.append(f"[{index}] {title}\nURL: {url}\n内容: {content}")

        return "\n\n".join(results).strip()
    except Exception as e:
        print(f"[WEB] Tavily search failed: {e}", file=sys.stderr)
        return ""




def _ensure_utf8_stdio() -> None:
    # Windows console encodings can crash on Chinese output; force UTF-8 streams.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# 将相应返回为字符串
def _coerce_content(raw_content) -> str:
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = getattr(item, "text", "") or getattr(item, "content", "")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(raw_content)


def _get_crag_evaluator():
    """Lazily load the CRAG evaluator model for relevance checks."""

    global _crag_tokenizer, _crag_model, _crag_device

    # Skip loading if CRAG evaluator is disabled
    if CRAG_EVALUATOR_MODEL is None:
        print("[CRAG Eval] CRAG evaluator disabled", file=sys.stderr)
        return None, None, None

    if _crag_tokenizer is not None and _crag_model is not None:
        return _crag_tokenizer, _crag_model, _crag_device

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification


        # use GPU if available for faster evaluation
        _crag_device = "cuda" if torch.cuda.is_available() else "cpu"

        _crag_tokenizer = AutoTokenizer.from_pretrained(CRAG_EVALUATOR_MODEL)
        _crag_model = AutoModelForSequenceClassification.from_pretrained(
            CRAG_EVALUATOR_MODEL,
            num_labels=3,
        )
        _crag_model.to(_crag_device)
        _crag_model.eval()
        return _crag_tokenizer, _crag_model, _crag_device
    except Exception as exc:
        print(f"[CRAG Eval] Failed to load evaluator: {exc}", file=sys.stderr)
        return None, None, None


def _doc_to_text(doc) -> str:
    if isinstance(doc, dict):
        return str(doc.get("content") or doc.get("text") or doc.get("title") or "")
    return str(getattr(doc, "page_content", doc))



# 接入llm尾端如果没有结果
def _answer_has_no_info(answer: str) -> bool:
    if not answer:
        return True

    no_info_phrases = [
        "根据现有文档无法找到相关信息",
        "根据现有文档无法找到",
        "文档中没有相关信息",
        "无法找到相关信息",
        "没有找到相关信息",
        "文档中未提及",
        "文档中没有提到",
        "无法回答",
        "不知道",
    ]
    return any(phrase in answer for phrase in no_info_phrases)


def answer_from_web_results(question: str, web_results: str, model: str, max_tokens: int, conversation_history=None) -> str:
    prompt = f"""请根据以下网络搜索结果回答用户问题。

用户问题：{question}

网络搜索结果：
{web_results}

请直接给出清晰、准确的答案。"""
    return call_direct_llm(
        prompt,
        model=model,
        max_tokens=max_tokens,
        conversation_history=conversation_history,
    )


def answer_from_docs(question: str, docs: List[Any], model: str, max_tokens: int, conversation_history=None) -> str:
    ai = CallAI(
        question=question,
        retrieved_docs=docs,
        model=model,
        max_tokens=max_tokens,
        include_citations=False,
        conversation_history=conversation_history,
    )
    return ai.call_ollama()


# 如果没有答案自己回答
def fallback_answer(question: str, model: str, max_tokens: int, conversation_history=None, research_note: bool = False) -> str:
    web_results = search_web(question)
    if web_results:
        return answer_from_web_results(question, web_results, model, max_tokens, conversation_history)

    if research_note:
        prompt = f"""基于本地知识库和网络搜索都没有找到可靠答案。

用户问题：{question}

请用你自己的分析回答，并明确说明：基于目前检索结果无法确认权威答案，以下是你的分析。"""
        return call_direct_llm(prompt, model=model, max_tokens=max_tokens, conversation_history=conversation_history)

    return call_direct_llm(
        question,
        model=model,
        max_tokens=max_tokens,
        conversation_history=conversation_history,
    )


def retrieve_semantic_docs(
    question: str,
    model: str,
    top_k: int,
    per_retriever_k: int,
    enable_rerank: bool,
) -> tuple[List[Any], float]:
    is_definition = query_processor.is_definition_query(question)
    is_complex = classify_question_complexity(question)

    if is_definition:
        print("[Semantic Search] Simple definition query", file=sys.stderr)
       
       
        retrieval_queries = generate_multi_queries(question, model=model)
        return get_docs_for_queries(
            retrieval_queries,
            topic=None,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            rrf_k=60,
            enable_rerank=False,
            bm25_weight=0.8,
            return_scores=True,
        )

    if is_complex:
        print("[Semantic Search] Complex query: step-back + topic + summary-first retrieval", file=sys.stderr)
        topic = extract_topic(question, model=model)
        retrieval_queries = build_retrieval_queries(question, model=model)
        return get_docs_for_queries(
            retrieval_queries,
            topic=topic,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            rrf_k=60,
            enable_rerank=enable_rerank,
            rerank_weight=0.5,
            topic_weight=0.3,
            bm25_weight=0.7,
            use_summary_search=True,
            return_scores=True,
        )

    print("[Semantic Search] Simple query", file=sys.stderr)
    retrieval_queries = generate_multi_queries(question, model=model)
    return get_docs_for_queries(
        retrieval_queries,
        topic=None,
        top_k=top_k,
        per_retriever_k=per_retriever_k,
        rrf_k=60,
        enable_rerank=False,
        bm25_weight=0.7,
        return_scores=True,
    )


# if it is relevant then return LLM evaluation
def answer_with_relevance_or_fallback(
    question: str,
    docs: List[Any],
    model: str,
    max_tokens: int,
    top_k: int,
    per_retriever_k: int,
    enable_rerank: bool,
    conversation_history=None,
) -> str:
    docs, relevance_status = apply_crag_relevance_filter(question, docs, top_k=top_k)
    print(f"[Relevance] Status: {relevance_status}", file=sys.stderr)

    if relevance_status in ("correct", "ambiguous"):
        answer = answer_from_docs(question, docs, model, max_tokens, conversation_history)
        if not _answer_has_no_info(answer):
            return answer

    print("[Fallback] Retrieved data is irrelevant or insufficient; trying AI-generated semantic query", file=sys.stderr)
    ai_query = call_direct_llm(
        f"请根据用户问题生成一段用于语义检索的核心答案或关键词，不要调用工具，不要解释。\n\n用户问题：{question}",
        model=model,
        max_tokens=2048,
        conversation_history=conversation_history,
    )

    
    if ai_query and not _answer_has_no_info(ai_query):
        docs2, _ = get_docs_for_queries(
            [ai_query],
            topic=None,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            rrf_k=60,
            enable_rerank=enable_rerank,
            bm25_weight=0.7,
            use_summary_search=True,
            return_scores=True,
        )
        docs2, status2 = apply_crag_relevance_filter(question, docs2, top_k=top_k)
        print(f"[Fallback] AI-query relevance status: {status2}", file=sys.stderr)
        if status2 in ("correct", "ambiguous"):
            answer = answer_from_docs(question, docs2, model, max_tokens, conversation_history)
            if not _answer_has_no_info(answer):
                return answer

    return fallback_answer(question, model, max_tokens, conversation_history, research_note=True)


def semantic_fallback_answer(
    question: str,
    model: str,
    max_tokens: int,
    top_k: int = 6,
    per_retriever_k: int = 12,
    enable_rerank: bool = True,
    conversation_history=None,
) -> str:
    print("[Fallback] Trying semantic search before web/direct LLM fallback", file=sys.stderr)
    docs, avg_score = retrieve_semantic_docs(
        question=question,
        model=model,
        top_k=top_k,
        per_retriever_k=per_retriever_k,
        enable_rerank=enable_rerank,
    )
    print(f"[Fallback] Semantic search retrieved {len(docs)} docs, avg_score: {avg_score:.3f}", file=sys.stderr)
    return answer_with_relevance_or_fallback(
        question=question,
        docs=docs,
        model=model,
        max_tokens=max_tokens,
        top_k=top_k,
        per_retriever_k=per_retriever_k,
        enable_rerank=enable_rerank,
        conversation_history=conversation_history,
    )


def _attach_relevance(doc, evaluation: dict):
    if isinstance(doc, dict):
        merged = dict(doc)
        merged["relevance"] = evaluation
        return merged
    return doc

# after get the retrieved documents, use CRAG evaluator to judge relevance and filter out irrelevant docs
def evaluate_document_relevance(query: str, document: str) -> Optional[dict]:
    # Use main LLM for relevance evaluation instead of CRAG model
    try:
        relevance_prompt = f"""请判断以下文档是否与查询相关。

查询: {query}

文档: {document}

请回答以下三个选项之一:
- correct: 文档包含查询的直接答案或高度相关信息
- incorrect: 文档与查询无关或包含错误信息
- ambiguous: 文档包含部分相关信息但不够充分

只回答一个词(correct/incorrect/ambiguous):"""

        result = call_direct_llm(relevance_prompt, max_tokens=10)

        # Parse the response
        result_lower = result.lower().strip()
        if "correct" in result_lower:
            return {"class": "correct", "confidence": 0.9}
        elif "incorrect" in result_lower:
            return {"class": "incorrect", "confidence": 0.9}
        elif "ambiguous" in result_lower:
            return {"class": "ambiguous", "confidence": 0.7}
        else:
            # Default to ambiguous if unclear
            return {"class": "ambiguous", "confidence": 0.5}
    except Exception as exc:
        print(f"[CRAG Eval] LLM evaluation failed: {exc}", file=sys.stderr)
        return None


def apply_crag_relevance_filter(query: str, docs: List, top_k: int = 5) -> tuple[List, str]:
    if not docs:
        return docs, "irrelevant"

    evaluations = []
    for doc in docs:
        evaluation = evaluate_document_relevance(query, _doc_to_text(doc))
        evaluations.append(evaluation)

    correct_docs = []
    ambiguous_docs = []
    incorrect_count = 0

    for doc, evaluation in zip(docs, evaluations):
        if not evaluation:
            correct_docs.append(doc)
            continue

        if evaluation["class"] == "correct":
            correct_docs.append(_attach_relevance(doc, evaluation))
        elif evaluation["class"] == "ambiguous":
            ambiguous_docs.append(_attach_relevance(doc, evaluation))
        else:
            incorrect_count += 1

    if correct_docs:
        return correct_docs[:top_k] if top_k else correct_docs, "correct"

    if ambiguous_docs:
        final_docs = list(ambiguous_docs)
        try:
            web_answer = search_web(query)
            if web_answer:
                final_docs.append(
                    {
                        "content": web_answer,
                        "title": "网络搜索结果",
                        "url": "",
                        "source": "web_search",
                        "id": "web_fallback",
                        "relevance": {"class": "ambiguous", "confidence": 1.0},
                    }
                )
        except Exception as exc:
            print(f"[CRAG Eval] Web search failed: {exc}", file=sys.stderr)
        return final_docs[:top_k] if top_k else final_docs, "ambiguous"

    return [], "irrelevant"



# ---------------------------   处理query ------------------------
# 解析JSON数组，去除重复项并保持顺序
def _parse_json_array(content: str) -> List[str]:
    if not content:
        return []

    cleaned = re.sub(r"^```(?:json)?\\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return _dedupe_keep_order([str(item).strip() for item in parsed])




# 去除出现两遍的字符串
def _dedupe_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out



def classify_question_complexity(question: str) -> bool:
    return query_processor.is_complex_query(question)



def generate_step_back_question(original_question: str, model: str = DEFAULT_MODEL) -> str:
    ai = CallAI(question=original_question, retrieved_docs=[], model=model, max_tokens=256)
    response = ai.client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Rewrite specific user questions into a broader background-retrieval query. Return one line only.",
            },
            {
                "role": "user",
                "content": (
                    "Original question:\n"
                    f"{original_question}\n\n"
                    "Output only one broader step-back query."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    # return result as string
    content = _coerce_content(getattr(response.choices[0].message, "content", None)).strip()
    return content or original_question




def handle_query_decomposition(query: str, model: str = DEFAULT_MODEL) -> List[str]:
    ai = CallAI(question=query, retrieved_docs=[], model=model, max_tokens=512)
    response = ai.client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You split complex questions into 3-5 atomic sub-questions. Return JSON array only.",
            },
            {
                "role": "user",
                "content": (
                    "Decompose the question into 3-5 sub-questions for retrieval.\n"
                    "Return JSON array only.\n"
                    f"Question: {query}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    content = _coerce_content(getattr(response.choices[0].message, "content", None)).strip()
    return _parse_json_array(content)[:5]


def generate_multi_queries(query: str, model: str = DEFAULT_MODEL) -> List[str]:
    candidates = query_processor.generate_multi_queries_ai(query, model=model)
    return _dedupe_keep_order([query] + candidates)


def extract_topic(question: str, model: str = DEFAULT_MODEL) -> str:
    """
    Extract core topic/keywords from user question using AI.
    Returns a concise topic string for hybrid search.
    """
    ai = CallAI(question=question, retrieved_docs=[], model=model, max_tokens=128)
    response = ai.client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze the user's question and extract the core topic or key concepts. "
                    "Return ONLY 3-5 essential keywords or a short phrase that captures the main subject. "
                    "No explanation, no punctuation, just the core topic."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nCore topic/keywords:",
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )
    content = _coerce_content(getattr(response.choices[0].message, "content", None)).strip()
    # Clean up the response
    content = re.sub(r'[，。,\.\n]+', ' ', content).strip()
    return content or question

# 根据query复杂度处理query
def build_retrieval_queries(question: str, model: str = DEFAULT_MODEL) -> List[str]:
    if not classify_question_complexity(question):
        return generate_multi_queries(question, model=model)

    composed = query_processor.process_user_input(question, enable_complex_detection=True)
    step_back = generate_step_back_question(composed, model=model)
    decomposed = handle_query_decomposition(step_back, model=model)
    return _dedupe_keep_order([question, composed, step_back] + decomposed)


# based on query and docs, call ollama
def handle_query(args):
    question = args.get("question", "")
    model = args.get("model", DEFAULT_MODEL)
    max_tokens = args.get("max_tokens", 1024)
    docs = args.get("docs", [])

    ai = CallAI(
        question=question,
        retrieved_docs=docs,
        model=model,
        max_tokens=max_tokens,
    )
    return ai.call_ollama()


def handle_rag(args):
    question = args.get("question", "")
    model = args.get("model", DEFAULT_MODEL)
    max_tokens = int(args.get("max_tokens", 4096))  # 增加到4096让回答更长
    top_k = int(args.get("top_k", DEFAULT_TOP_K))  # 使用默认值8
    per_retriever_k = int(args.get("per_retriever_k", DEFAULT_PER_RETRIEVER_K))  # 使用默认值16
    enable_rerank = args.get("enable_rerank", True)
    conversation_history = args.get("conversation_history", [])
    enable_self_rag = args.get("enable_self_rag", True)  # 新增参数，默认启用 Self-RAG

    # 使用 Self-RAG 机制进行检索（如果启用）
    if enable_self_rag:
        print("[Self-RAG] 启用 Self-RAG 检索机制", file=sys.stderr)
        docs, quality, retry_count = self_rag_retrieval_with_retry(
            query=question,
            retrieve_func=retrieve_semantic_docs,
            max_retries=int(args.get("self_rag_max_retries", 1)),
            preference_weights=SELF_RAG_PREFERENCE_WEIGHTS,
            model=model,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            enable_rerank=enable_rerank,
        )
        print(f"[Self-RAG] 检索完成，重试次数: {retry_count}, 最终分数: {quality['total_score']:.2f}", file=sys.stderr)
    else:
        # 原有的检索逻辑
        docs, avg_score = retrieve_semantic_docs(
            question=question,
            model=model,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            enable_rerank=enable_rerank,
        )
        print(f"[Semantic Search] Retrieved {len(docs)} docs, avg_score: {avg_score:.3f}", file=sys.stderr)

    return answer_with_relevance_or_fallback(
        question=question,
        docs=docs,
        model=model,
        max_tokens=max_tokens,
        top_k=top_k,
        per_retriever_k=per_retriever_k,
        enable_rerank=enable_rerank,
        conversation_history=conversation_history,
    )



def handle_deepthink(_args):
    return "DeepThink mode is not implemented yet."


def handle_database_query(args):
    """
    Handle database query results with AI-generated natural language response.
    """
    query = args.get("query", "")
    db_results = args.get("db_results", "[]")
    model = args.get("model", DEFAULT_MODEL)
    
    try:
        # Parse database results
        import json
        results = json.loads(db_results)
        
        # Format results for the prompt
        results_text = ""
        if isinstance(results, list) and len(results) > 0:
            for i, row in enumerate(results, 1):
                results_text += f"\n记录 {i}:\n"
                for key, value in row.items():
                    results_text += f"  {key}: {value}\n"
        else:
            results_text = "无数据"
        
        # Generate AI response
        prompt = f"""
你是一个数据查询助手。用户提出了一个问题，系统从数据库中查询到了以下结果。

用户问题：{query}

数据库查询结果：
{results_text}

请根据用户的问题和数据库查询结果，用自然语言给出一个清晰、友好的回答。
如果查询结果为空，请明确告知用户未找到相关数据。
不要编造数据，只基于提供的查询结果回答。
"""
        
        ai = CallAI(
            question=prompt,
            retrieved_docs=[],
            model=model,
            max_tokens=2048,
            include_citations=False,
        )
        
        answer = ai.call_ollama()
        
        return json.dumps({"success": True, "answer": answer}, ensure_ascii=False)
        
    except Exception as e:
        print(f"[DB Query Handler] Error: {e}", file=sys.stderr)
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)




# 通过意图模型进行意图分类

def handle_classify_intent(args):
    """
    使用训练好的意图分类器判断用户查询意图
    """
    query = args.get("query", "")
    
    try:
        # 尝试导入意图分类器
        from intent_classifier import IntentClassifier
        
        # 加载训练好的模型
        classifier = IntentClassifier(model_path="./intent_model")
        
        # 预测意图
        intent = classifier.predict(query)
        
        return json.dumps({"success": True, "intent": intent}, ensure_ascii=False)
        
    except ImportError:
        print("[Intent Classifier] Module not found, using rule-based fallback", file=sys.stderr)
        # 回退到规则匹配
        db_keywords = ['查询', 'select', '数据库', 'database', '员工', 'employee', 
                      '部门', 'department', '显示', 'show', '列出', 'list',
                      '统计', 'count', '查找', 'find', '搜索', 'search']
        lower_query = query.lower()
        if any(keyword in lower_query for keyword in db_keywords):
            return json.dumps({"success": True, "intent": "database_query"}, ensure_ascii=False)
        else:
            return json.dumps({"success": True, "intent": "rag_query"}, ensure_ascii=False)
    except Exception as e:
        print(f"[Intent Classifier] Error: {e}", file=sys.stderr)
        # 出错时回退到 rag_query
        return json.dumps({"success": True, "intent": "rag_query"}, ensure_ascii=False)


def _answer_from_sql_result(question, result, model, max_tokens, conversation_history=None):
    result_data = result.get("result")
    sql = result.get("sql", "")
    result_text = json.dumps(result_data, ensure_ascii=False, indent=2)
    prompt = f"""你是一个数据库查询结果分析助手。请根据数据库查询结果回答用户问题。

用户问题：
{question}

执行的 SQL：
{sql}

数据库查询结果：
{result_text}

要求：
1. 只根据数据库查询结果回答，不要编造数据。
2. 如果是数量问题，请直接给出数量。
3. 用自然、简洁的中文回答。
"""
    ai = CallAI(
        question=prompt,
        retrieved_docs=[],
        model=model,
        max_tokens=max_tokens,
        include_citations=False,
        conversation_history=conversation_history,
    )
    return ai.call_ollama()


def handle_generate_sql(args):
    """
    使用 LangChain SQL 生成器将自然语言转换为 SQL 并执行
    如果 SQL 查询失败或找不到结果，回退到 RAG 检索
    """
    query = args.get("query") or args.get("question", "")
    model = args.get("model", DEFAULT_MODEL)
    max_tokens = int(args.get("max_tokens", 4096))
    top_k = int(args.get("top_k", DEFAULT_TOP_K))
    per_retriever_k = int(args.get("per_retriever_k", 12))
    enable_rerank = args.get("enable_rerank", True)
    conversation_history = args.get("conversation_history", [])
    
    try:
        # 尝试导入 SQL 生成器
        from sql_generator import SQLGenerator
        
        # 创建 SQL 生成器实例
        generator = SQLGenerator()
        
        # 生成 SQL 并执行
        result = generator.generate_sql(query)
        
        # 检查 SQL 查询是否成功且有结果
        success = result.get("success")
        result_data = result.get("result")
        answer = result.get("answer", "")
        error = result.get("error")
        
        # 判断是否需要回退到 RAG：
        # 1. success 为 False
        # 2. result 为空或 None
        # 3. answer 包含错误信息（如"无法理解"、"未能获取"等）
        # 4. 存在 error
        error_phrases = ["无法理解", "未能理解", "未能获取", "查询失败", "找不到", "not found", "无法找到", "未能查询到", "未能正确理解", "无法提供"]
        has_error_in_answer = any(phrase in answer for phrase in error_phrases)
        
        # 检查 result_data 是否为空列表或 None
        is_result_empty = result_data is None or (isinstance(result_data, list) and len(result_data) == 0)
        
        if not success or is_result_empty or has_error_in_answer or error:
            print(f"[SQL Generator] SQL query failed or no results, falling back to semantic search", file=sys.stderr)
            print(f"[SQL Generator] Debug: success={success}, result={result_data}, answer={answer[:50] if answer else 'N/A'}, error={error}", file=sys.stderr)
            answer = semantic_fallback_answer(query, model, max_tokens, top_k, per_retriever_k, enable_rerank, conversation_history)
            return json.dumps({"success": True, "answer": answer}, ensure_ascii=False)
        
        sql_answer = _answer_from_sql_result(query, result, model, max_tokens, conversation_history)
        return json.dumps({"success": True, "answer": sql_answer}, ensure_ascii=False)
        
    except ImportError:
        print("[SQL Generator] Module not found, falling back to semantic search", file=sys.stderr)
        answer = semantic_fallback_answer(query, model, max_tokens, top_k, per_retriever_k, enable_rerank, conversation_history)
        return json.dumps({"success": True, "answer": answer}, ensure_ascii=False)
    except Exception as e:
        print(f"[SQL Generator] Error: {e}, falling back to semantic search", file=sys.stderr)
        answer = semantic_fallback_answer(query, model, max_tokens, top_k, per_retriever_k, enable_rerank, conversation_history)
        return json.dumps({"success": True, "answer": answer, "error": str(e)}, ensure_ascii=False)





# 处理上传文件
def handle_upload(args):
    """
    处理文本上传到向量数据库
    
    Args:
        args: 包含 text, source, chunk_size, chunk_overlap, split_method 的字典
        
    Returns:
        str: JSON 格式的上传结果
    """
    text = args.get("text", "")
    source = args.get("source", "manual_upload")
    chunk_size = int(args.get("chunk_size", 500))
    chunk_overlap = int(args.get("chunk_overlap", 50))
    split_method = args.get("split_method", "character")

    if not str(text).strip():
        return json.dumps({"success": False, "error": "Text is required"}, ensure_ascii=False)

    try:
        result = upload_text_to_vector_store(
            text=str(text),
            source=str(source),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_method=split_method,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def classify_fallback_strategy(question: str, model: str = DEFAULT_MODEL) -> str:
    """
    判断 RAG 检索失败时的回退策略
    
    Args:
        question: 用户查询
        model: LLM 模型
        
    Returns:
        str: "web_search" 或 "direct_chat"
    """
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model,
            temperature=0
        )
        
        prompt = f"""请判断以下查询适合哪种回退策略：

查询：{question}

策略选项：
1. web_search - 需要网络搜索获取最新信息（如新闻、天气、时事、具体数据等）
2. direct_chat - 可以直接由 AI 回答（如闲聊、问候、常识性问题、创意问题等）

只返回策略名称（web_search 或 direct_chat），不要其他文字。"""
        
        response = llm.invoke(prompt)
        strategy = response.content.strip().lower()
        
        if "web" in strategy:
            return "web_search"
        else:
            return "direct_chat"
            
    except Exception as e:
        print(f"[Fallback Classification] Error: {e}, defaulting to direct_chat", file=sys.stderr)
        return "direct_chat"


def handle_smart_query(args):
    """
    智能查询处理器 - 自动路由到 SQL/语义/混合搜索
    
    工作流程：
    1. 使用查询路由器（LLM + 意图分类器）判断查询类型
    2. 根据查询类型路由到对应处理器：
       - sql_search: 使用 SQL 生成器（元数据过滤器）
       - semantic_search: 使用 RAG 检索
       - hybrid: 使用混合搜索（元数据过滤 + 语义检索）
    3. 返回结果
    
    Args:
        args: 包含 question, model, top_k 等参数的字典
        
    Returns:
        str: JSON 格式的查询结果
    """
    question = args.get("question", "")
    model = args.get("model", DEFAULT_MODEL)
    top_k = int(args.get("top_k", DEFAULT_TOP_K))
    
    if not question:
        return json.dumps({"success": False, "error": "Question is required"}, ensure_ascii=False)
    
    try:
        import sys
        print(f"[Smart Query] 处理查询: {question}", file=sys.stderr)
        
        # 初始化查询路由器
        router = QueryRouter()
        
        # 路由查询
        routing_info = router.route_query(question)
        query_type = routing_info["query_type"]
        confidence = routing_info["confidence"]
        intent = routing_info.get("intent", "")
        handler = routing_info.get("handler", "RAGRetriever")
        
        print(f"[Smart Query] 意图: {intent}, 查询类型: {query_type}, 置信度: {confidence}, 处理器: {handler}", file=sys.stderr)
        
        # 根据 QueryRouter 最终处理器路由
        if handler == "SQLGenerator":
            print(f"[Smart Query] 路由到 SQL 搜索", file=sys.stderr)
            return handle_generate_sql(args)

        elif handler == "HybridSearchProcessor":
            print(f"[Smart Query] 路由到混合搜索", file=sys.stderr)
            return handle_hybrid_search(args)

        elif handler == "RAGRetriever":
            print(f"[Smart Query] 路由到 RAG/语义搜索", file=sys.stderr)
            return handle_rag(args)

        else:
            print(f"[Smart Query] 未知处理器 {handler}，路由到 RAG 搜索", file=sys.stderr)
            return handle_rag(args)
            
    except Exception as e:
        print(f"[Smart Query] Error: {e}", file=sys.stderr)
        # 回退到 RAG 检索
        return handle_rag(args)


def handle_hybrid_search(args):
    question = args.get("question", "")
    model = args.get("model", DEFAULT_MODEL)
    top_k = int(args.get("top_k", DEFAULT_TOP_K))
    max_tokens = int(args.get("max_tokens", 4096))
    per_retriever_k = int(args.get("per_retriever_k", 12))
    enable_rerank = args.get("enable_rerank", True)
    conversation_history = args.get("conversation_history", [])

    if not question:
        return json.dumps({"success": False, "error": "Question is required"}, ensure_ascii=False)

    try:
        print(f"[Hybrid Search] Processing query: {question}", file=sys.stderr)
        processor = HybridSearchProcessor()

        # Extract structured SQL parameters from query using template
        from db_metadata_filter import DatabaseMetadataFilter
        sql_filter = DatabaseMetadataFilter()
        sql_template = sql_filter.parse_query_template(question)
        print(f"[Hybrid Search] Extracted SQL template: table={sql_template.table_name}, conditions={sql_template.conditions}", file=sys.stderr)

        # Phase 1: Summary-first semantic search on Pinecone
        semantic_docs = get_docs_by_summary_search(
            queries=[question],
            topic=None,
            top_k=top_k,
            per_retriever_k=per_retriever_k,
            enable_rerank=enable_rerank,
        )
        print(f"[Hybrid Search] Summary-first semantic search returned {len(semantic_docs)} docs", file=sys.stderr)

        # Phase 2: Database search via structured SQL template
        db_results = []
        try:
            from sql_generator import SQLGenerator
            generator = SQLGenerator()
            db_result = generator.generate_sql(question)
            if db_result.get("success") and db_result.get("result"):
                db_results = db_result.get("result", [])
                print(f"[Hybrid Search] Database search returned {len(db_results)} rows", file=sys.stderr)
        except Exception as db_e:
            print(f"[Hybrid Search] Database search failed: {db_e}", file=sys.stderr)

        # Combine semantic docs and db results
        combined_docs = semantic_docs[:top_k]
        if not combined_docs and db_results:
            combined_docs = [{"content": str(row), "source": "database", "metadata": row} for row in db_results[:top_k]]

        if combined_docs:
            answer = answer_with_relevance_or_fallback(
                question=question,
                docs=combined_docs,
                model=model,
                max_tokens=max_tokens,
                top_k=top_k,
                per_retriever_k=per_retriever_k,
                enable_rerank=enable_rerank,
                conversation_history=conversation_history,
            )
        elif sql_template.conditions:
            template_reference = json.dumps({
                "table": sql_template.table_name,
                "conditions": sql_template.conditions,
                "date_range": sql_template.date_range,
            }, ensure_ascii=False)
            prompt = f"""Local documents and database search did not return usable results, but the system extracted these SQL query parameters.

User question:
{question}

SQL query parameters:
{template_reference}

Answer using these parameters as reference. If the answer cannot be confirmed from retrieved results, clearly say it cannot be confirmed."""
            answer = call_direct_llm(prompt, model=model, max_tokens=max_tokens, conversation_history=conversation_history)
            if _answer_has_no_info(answer):
                answer = semantic_fallback_answer(question, model, max_tokens, top_k, per_retriever_k, enable_rerank, conversation_history)
        else:
            answer = semantic_fallback_answer(question, model, max_tokens, top_k, per_retriever_k, enable_rerank, conversation_history)

        return json.dumps({
            "success": True,
            "answer": answer,
        }, ensure_ascii=False)
    except Exception as e:
        print(f"[Hybrid Search] Error: {e}, falling back to semantic search", file=sys.stderr)
        answer = semantic_fallback_answer(question, model, max_tokens, top_k, per_retriever_k, enable_rerank, conversation_history)
        return json.dumps({"success": True, "answer": answer, "error": str(e)}, ensure_ascii=False)


def main():
    """
    主函数：处理来自 Node.js 的 API 请求
    
    工作流程：
    1. 解析命令行参数（JSON 格式）
    2. 根据 request_type 路由到对应的处理器
    3. 返回处理结果
    
    支持的请求类型：
    - query: 普通 LLM 查询
    - rag: RAG 检索增强生成
    - deepthink: 深度思考模式
    - upload: 上传文本到向量数据库
    - database_query: 数据库查询
    - classify_intent: 意图分类
    - generate_sql: SQL 生成
    - smart_query: 智能查询（自动路由到 SQL/语义/混合搜索）
    - hybrid_search: 混合搜索（元数据过滤 + 语义检索）
    """
    _ensure_utf8_stdio()

    try:
        if len(sys.argv) < 2:
            print("[DEBUG] Using hardcoded test args", file=sys.stderr)
        args = _parse_cli_json_args(sys.argv[1:])

        request_type = args.get("type", "query")

        # 路由到对应的处理器
        if request_type == "rag":
            result = handle_rag(args)
        elif request_type == "deepthink":
            result = handle_deepthink(args)
        elif request_type == "upload":
            result = handle_upload(args)
        elif request_type == "database_query":
            result = handle_database_query(args)
        elif request_type == "classify_intent":
            result = handle_classify_intent(args)
        elif request_type == "generate_sql":
            result = handle_generate_sql(args)
        elif request_type == "smart_query":
            result = handle_smart_query(args)
        elif request_type == "hybrid_search":
            result = handle_hybrid_search(args)
        else:
            result = handle_query(args)

        print(result)

    except json.JSONDecodeError as exc:
        print(f"Error parsing arguments: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
