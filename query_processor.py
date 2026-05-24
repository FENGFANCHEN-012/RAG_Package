"""
 ============================================================================
 查询处理器 (query_processor.py)
 ============================================================================
 
 功能描述：
 - 处理和扩展用户查询
 - 判断查询类型（定义查询、复杂查询等）
 - 使用 AI 生成多个相关查询
 - 支持查询组合和规范化
 
 主要函数：
 - is_definition_query(): 判断是否为定义类查询
 - is_complex_query(): 判断是否为复杂查询
 - generate_composed_query(): 生成组合查询
 - generate_multi_queries_ai(): 使用 AI 生成多个相关查询
 - _normalize_query_list(): 规范化查询列表
 - _extract_queries_from_text(): 从文本提取查询
 - _ensure_min_queries(): 确保最少查询数量
 
 用途：
 - RAG 检索时的查询扩展
 - 提高检索召回率
 - 支持多角度查询
 
 依赖：
 - openai: OpenAI 兼容客户端（用于 Ollama）
 
 ============================================================================
"""
import re
import json
from typing import List, Optional
from openai import OpenAI

# ========== 默认配置 ==========
# 使用与 call_AI.py 相同的默认配置
DEFAULT_MODEL = "blissful_ishizaka_626/gemma4-cloud"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"


# ========== 辅助函数 ==========

def _coerce_message_content_to_text(raw_content) -> str:
    """
    将聊天消息内容转换为纯文本，处理多种可能的内容格式。
    支持 None、字符串、列表（包含字典或对象）等输入格式。
    """
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", "") or getattr(item, "content", "")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(raw_content)


def _normalize_query_list(items) -> List[str]:
    """
    将模型输出标准化为唯一的、非空的查询列表。
    去除重复项，限制列表长度为 5 个元素。
    """
    cleaned = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
        if len(cleaned) >= 5:
            break
    return cleaned


def _extract_queries_from_text(content: str) -> List[str]:
    """
    从常见的模型输出格式中解析查询（JSON 数组/对象或纯文本行）。
    支持 JSON 解析、嵌入式 JSON 数组提取和纯文本行解析。
    """
    if not content:
        return []

    # 处理常见的 markdown 包装
    text = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # 1) 尝试直接 JSON 解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _normalize_query_list(parsed)
        if isinstance(parsed, dict):
            for key in ("queries", "query", "items", "data", "results"):
                value = parsed.get(key)
                if isinstance(value, list):
                    normalized = []
                    for v in value:
                        if isinstance(v, dict):
                            normalized.append(v.get("query") or v.get("text") or v.get("content") or "")
                        else:
                            normalized.append(v)
                    return _normalize_query_list(normalized)
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
    except json.JSONDecodeError:
        pass

    # 2) 尝试提取文本中嵌入的第一个 JSON 数组
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            embedded = json.loads(match.group(0))
            if isinstance(embedded, list):
                return _normalize_query_list(embedded)
        except json.JSONDecodeError:
            pass

    # 3) Fallback: parse numbered or bullet lines.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+[\.、]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip("\"' ，。,.")
        if line:
            cleaned.append(line)
    return _normalize_query_list(cleaned)


def _ensure_min_queries(base_query: str, queries: List[str], min_count: int = 5) -> List[str]:
    """Ensure at least min_count similar queries are returned."""
    unique = _normalize_query_list(queries)

    if base_query.strip() and base_query.strip() not in unique:
        unique.insert(0, base_query.strip())

    templates = [
        "{q} 的定义和核心概念",
        "{q} 的常见原因和风险因素",
        "{q} 的典型症状与识别方法",
        "{q} 的检查诊断与评估方式",
        "{q} 的治疗方案与用药选择",
        "{q} 的日常管理与注意事项",
        "如何预防和长期管理 {q}",
        "{q} 常见误区与正确做法",
    ]

    seed = base_query.strip() or "该问题"
    for template in templates:
        candidate = template.format(q=seed).strip()
        if candidate and candidate not in unique:
            unique.append(candidate)
        if len(unique) >= min_count:
            break

    return unique[:max(min_count, len(unique))]

def get_ai_client():
    """返回 OpenAI 兼容的客户端实例"""
    return OpenAI(
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
    )


def is_definition_query(query: str) -> bool:
    """
    判断查询是否是"名词 + 是什么"类型的定义查询。
    这类查询应该侧重关键词检索（BM25），因为用户在寻找明确的定义或解释。

    规则：
    1. 包含"是什么"、"是什么意思"、"是什么的"、"是什么叫"等模式
    2. 查询以名词开头，后接"是什么"类短语
    3. 查询长度相对较短（通常不超过30字符）
    """
    if not query:
        return False

    query = query.strip()

    # 定义相关的模式
    definition_patterns = [
        r'是什么$',
        r'是什么意思$',
        r'是什么的$',
        r'是什么叫$',
        r'什么是',
        r'什么叫',
        r'何为',
        r'即',
        r'指的是',
    ]

    for pattern in definition_patterns:
        if re.search(pattern, query):
            return True

    # 检查是否是简短的名词+定义模式
    # 例如："糖尿病是什么"、"胰岛素是什么"
    if len(query) <= 30 and ('是什么' in query or '是什么意思' in query):
        return True

    return False


def is_complex_query(query: str) -> bool:
    """
    基于规则判断查询是否为复杂查询。
    规则：
    1. 包含多个问号（>1）
    2. 包含连接词（和、或、同时、以及、还有、并且）
    3. 长度超过 50 个字符
    4. 包含多个子句（逗号、分号分隔）
    5. 包含多个句子（句号、感叹号、问号分隔）
    """
    if not query:
        return False

    # 规则1：多个问号
    if query.count('？') > 1 or query.count('?') > 1:
        return True

    # 规则2：连接词
    connectives = ['和', '或', '同时', '以及', '还有', '并且', '而且', '或者', '还是']
    for conn in connectives:
        if conn in query:
            return True

    # 规则3：长度阈值
    if len(query) > 50:
        # 长度长但可能是简单描述，结合其他规则
        pass

    # 规则4：多个子句（逗号、分号）
    if query.count('，') > 1 or query.count(',') > 1 or query.count('；') > 0 or query.count(';') > 0:
        return True

    # 规则5：多个句子（句号、感叹号、问号）
    sentence_delimiters = r'[。！？!?]'
    sentences = re.split(sentence_delimiters, query)
    # 过滤空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        return True

    # 如果以上规则都不满足，但长度超过100字符，也视为复杂
    if len(query) > 100:
        return True

    return False


def generate_composed_query(query: str, model: str = DEFAULT_MODEL) -> str:
    """
    将复杂查询重写为一个更全面、更清晰的组合查询。
    使用 AI 模型进行重写。
    """
    prompt = f"""请将以下复杂问题重写为一个更全面、更清晰的查询，以便进行信息检索。
    保持原意，但将其整合成一个连贯的查询。不要添加额外解释，直接输出重写后的查询。



原问题：{query}

重写后的查询："""
    
    client = get_ai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个查询重写助手，专注于将复杂问题重写为清晰的检索查询。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=256,
    )
    
    raw_content = getattr(response.choices[0].message, "content", None)
    composed = _coerce_message_content_to_text(raw_content).strip()

    if composed:
        composed = re.sub(r"^```(?:json)?\s*", "", composed, flags=re.IGNORECASE)
        composed = re.sub(r"\s*```$", "", composed)

    if not composed:
        return query

    # 清理可能的引用标记
    composed = re.sub(r'^["\']|["\']$', '', composed)
    return composed if composed else query




def generate_multi_queries_ai(query: str, model: str = DEFAULT_MODEL) -> List[str]:
    """
    使用 AI 生成多个相关查询（类似于 call_AI.get_muti_query，但修复了格式）。
    返回字符串列表。
    """
    prompt = f"请基于以下问题生成至少5个类似查询，输出纯JSON数组字符串（例如 [\"q1\",\"q2\"]），不要任何额外文字：{query}"
    client = get_ai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个查询扩展助手，生成与原始问题相关的变体查询。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512,
    )
    
    raw_content = getattr(response.choices[0].message, "content", None)
    content = _coerce_message_content_to_text(raw_content).strip()
    queries = _extract_queries_from_text(content) if content else []

    # Retry once with a stricter prompt if the first parse returns too few queries.
    if len(queries) < 5:
        retry_prompt = (
            "请仅返回JSON数组，必须包含至少5个不同查询，不要markdown，不要解释。"
            f"\n原始问题：{query}"
        )
        retry_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是查询改写助手，只输出JSON数组。"},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        retry_raw_content = getattr(retry_response.choices[0].message, "content", None)
        retry_content = _coerce_message_content_to_text(retry_raw_content).strip()
        retry_queries = _extract_queries_from_text(retry_content) if retry_content else []
        if len(retry_queries) > len(queries):
            queries = retry_queries

    return _ensure_min_queries(query, queries, min_count=5)




def process_user_input(query: str, enable_complex_detection: bool = True) -> str:
    """
    处理用户输入：如果启用复杂度检测且判断为复杂查询，则生成组合查询。
    否则返回原始查询。
    """
    if not enable_complex_detection:
        return query
    
    if is_complex_query(query):
        print(f"检测到复杂查询，正在生成组合查询...")
        composed = generate_composed_query(query)
        if not composed.strip():
            return query
        print(f"组合查询: {composed}")
        return composed
    else:
        return query

if __name__ == "__main__":
    # 测试代码
    test_queries = [
        "糖尿病注意事项",
        "我奶奶得了糖尿病，应该注意什么？",
        "阿司匹林和布洛芬有什么区别？哪个更适合头痛？",
        "请问高血压的治疗方法有哪些？以及饮食上需要注意什么？同时，运动有什么建议？"
        
    ]
    for q in test_queries:
        print(f"原始查询: {q}")
        print(f"是否复杂: {is_complex_query(q)}")
        if is_complex_query(q):
            composed = generate_composed_query(q)
            print(f"组合查询: {composed}")
        print("-" * 40)