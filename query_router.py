"""
 ============================================================================
 查询路由器 (query_router.py)
 ============================================================================
 
 功能描述：
- 使用 LLM 对用户查询进行分类
- 集成意图分类模型（TF-IDF + SVM）进行初步意图判断
- 判断查询类型：sql_search（SQL 搜索）、semantic_search（语义搜索）、hybrid（混合搜索）
- 根据查询类型路由到不同的处理器
- 支持自定义路由规则和提示词
 
 主要类：
 - QueryRouter: 查询路由器类
   - __init__: 初始化路由器、LLM 和意图分类器
   - classify_query: 分类查询类型（结合意图分类器和 LLM）
   - route_query: 根据查询类型路由到对应处理器
   - _build_classification_prompt: 构建分类提示词
   - _map_intent_to_query_type: 将意图分类映射到查询类型
 
 工作流程：
 1. 接收用户自然语言查询
 2. 使用意图分类器进行初步意图判断（database_query/rag_query）
 3. LLM 分析查询意图和特征，细化为查询类型
 4. 分类为 sql_search、semantic_search 或 hybrid
 5. 返回查询类型和置信度
 
 路由规则：
 - sql_search: 涉及数据库查询、数据统计、结构化数据
 - semantic_search: 涉及知识检索、文档搜索、概念理解
 - hybrid: 同时需要结构化数据和语义检索
 
 依赖：
 - langchain: LangChain 框架
 - langchain-openai: OpenAI 兼容客户端
 - intent_classifier: 意图分类器（TF-IDF + SVM）
 
 ============================================================================
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
import sys
import re
from intent_classifier import IntentClassifier


class QueryType(BaseModel):
    """查询类型分类结果"""
    query_type: str = Field(description="查询类型: sql_search, semantic_search, hybrid")
    confidence: float = Field(description="置信度 0-1")
    reasoning: str = Field(description="分类理由")
    intent: str = Field(default="", description="意图分类结果")


class QueryRouter:
    """
    查询路由器
    使用 LLM 和意图分类器对用户查询进行分类和路由
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        intent_classifier: Optional[IntentClassifier] = None
    ):
        """
        初始化查询路由器

        Args:
            llm: LLM 实例，如果为 None 则使用默认配置
            intent_classifier: 意图分类器实例，如果为 None 则创建新实例
        """
        # 初始化 LLM
        if llm is None:
            self.llm = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="blissful_ishizaka_626/gemma4-cloud",
                temperature=0
            )
        else:
            self.llm = llm

        # 初始化意图分类器
        if intent_classifier is None:
            self.intent_classifier = IntentClassifier()
        else:
            self.intent_classifier = intent_classifier

        # 检查 SQL 数据库是否可用
        self.sql_available = self._check_sql_available()

    def _check_sql_available(self) -> bool:
        """
        检查 SQL 数据库是否可用

        Returns:
            bool: SQL 是否可用
        """
        try:
            from sql_generator import SQLGenerator
            generator = SQLGenerator()
            return generator.engine is not None
        except Exception as e:
            print(f"[Query Router] SQL database not available: {e}", file=sys.stderr)
            return False
    
    def _build_classification_prompt(self, query: str, intent_result: Optional[str] = None) -> str:
        """
        构建查询分类提示词
        
        Args:
            query: 用户查询
            intent_result: 意图分类器的结果（可选）
            
        Returns:
            str: 分类提示词
        """
        # 添加意图分类器的结果到提示词
        intent_hint = ""
        if intent_result:
            intent_hint = f"\n意图分类器的初步判断: {intent_result}（此结果权重很高，请优先参考）\n"

        prompt = f"""你是一个智能查询分类器。分析用户的查询，判断其类型。

查询类型定义：
1. sql_search（SQL 搜索）：
   - 涉及数据库查询
   - 需要统计、计数、聚合
   - 查询结构化数据（员工、部门、订单等）
   - 示例："有多少个员工"、"查询2024年入职的员工"

2. semantic_search（语义搜索）：
   - 涉及知识检索
   - 查询文档、文章、概念
   - 需要语义理解和相似度匹配
   - 示例："什么是机器学习"、"查找关于深度学习的文档"

3. hybrid（混合搜索）：
   - 同时涉及结构化数据和语义检索
   - 需要先过滤再进行语义搜索
   - 只有当查询同时包含明确的时间/类别过滤条件和语义内容时才选择
   - 示例："查找2024年发布的关于机器学习的文章"
   - 谨慎选择
{intent_hint}
用户查询: {query}

重要：如果意图分类器提供了初步判断，请优先参考其结果，除非有非常明确的证据表明其判断错误。

请返回 JSON 格式的分类结果：
{{
  "query_type": "sql_search/semantic_search/hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "分类理由"
}}

只返回 JSON，不要有其他文字：
"""
        return prompt
    
    def _map_intent_to_query_type(self, intent: str) -> str:
        """
        将意图分类映射到查询类型
        
        Args:
            intent: 意图分类结果 (database_query/rag_query)
            
        Returns:
            str: 查询类型 (sql_search/semantic_search/hybrid)
        """
        # 意图映射规则
        intent_mapping = {
            "database_query": "sql_search",
            "rag_query": "semantic_search"  # 修复：rag_query 应该映射到 semantic_search
        }
        return intent_mapping.get(intent, "semantic_search")
    
    def _self_rag_verification(self, query: str, classification_result: dict, intent_result: Optional[str] = None) -> dict:
        """
        Self-RAG: 让 LLM 自我验证分类结果是否合理
        
        Args:
            query: 用户查询
            classification_result: LLM 的分类结果
            intent_result: 意图分类器的结果（可选）
            
        Returns:
            dict: {
                "is_reasonable": bool,
                "suggested_type": str,
                "reasoning": str
            }
        """
        query_type = classification_result.get("query_type", "semantic_search")
        reasoning = classification_result.get("reasoning", "")
        confidence = classification_result.get("confidence", 0.8)
        
        intent_hint = ""
        if intent_result:
            intent_hint = f"\n意图分类器的判断: {intent_result}"
        
        verification_prompt = f"""你是一个查询分类验证专家。请评估以下分类结果是否合理。

用户查询: {query}

分类结果:
- 类型: {query_type}
- 理由: {reasoning}
- 置信度: {confidence}
{intent_hint}

查询类型定义:
1. sql_search: 涉及数据库查询、统计、计数、结构化数据（员工、部门、订单等）
2. semantic_search: 涉及知识检索、文档搜索、概念理解
3. hybrid: 同时需要结构化过滤和语义检索（如"查找2024年发布的关于机器学习的文章"）

评估标准:
1. 分类是否与查询内容匹配
2. 理由是否充分且有说服力
3. 如果意图分类器有判断，是否与其一致（除非有明确证据表明意图分类器错误）
4. 是否有更合适的分类类型

请返回 JSON 格式的评估结果:
{{
  "is_reasonable": true/false,
  "suggested_type": "建议的分类类型（如果不合理）",
  "reasoning": "评估理由"
}}

只返回 JSON，不要有其他文字：
"""
        
        try:
            response = self.llm.invoke(verification_prompt)
            content = response.content.strip()
            
            # 清理可能的 markdown 代码块
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            # 解析 JSON
            result = json.loads(content)
            
            print(f"[Query Router] Self-RAG 验证结果: is_reasonable={result.get('is_reasonable')}, suggested_type={result.get('suggested_type')}", file=sys.stderr)
            
            return {
                "is_reasonable": result.get("is_reasonable", True),
                "suggested_type": result.get("suggested_type", query_type),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            print(f"[Query Router] Self-RAG 验证失败: {e}", file=sys.stderr)
            # 验证失败时，默认认为分类合理
            return {
                "is_reasonable": True,
                "suggested_type": query_type,
                "reasoning": f"验证失败，使用原分类: {str(e)}"
            }

    def _looks_like_sql_query(self, query: str) -> bool:
        """
        判断查询是否真的属于当前数据库 schema 的结构化查询。
        只有明确涉及数据库表/实体时才允许路由到 SQL。
        """
        query_lower = query.lower()
        db_entity_keywords = [
            "员工", "employee", "employees",
            "部门", "department", "departments",
            "发票", "invoice", "invoices",
            "停车场", "carpark", "carparks",
            "username", "email", "position", "hire_date",
        ]
        aggregate_keywords = [
            "多少", "几个", "数量", "统计", "count", "sum", "avg", "max", "min",
            "最大", "最小", "平均", "列出", "查询", "显示", "找出",
        ]
        return (
            any(keyword in query_lower for keyword in db_entity_keywords)
            and any(keyword in query_lower for keyword in aggregate_keywords)
        )
    
    def classify_query(self, query: str) -> QueryType:
        """
        分类查询类型（结合意图分类器和 LLM）
        
        Args:
            query: 用户查询
            
        Returns:
            QueryType: 查询类型分类结果
        """
        # 步骤 1: 使用意图分类器进行初步判断
        intent_result = None
        if self.intent_classifier and self.intent_classifier.is_trained:
            try:
                intent_result = self.intent_classifier.predict(query)
                print(f"[Query Router] 意图分类器结果: {intent_result}", file=sys.stderr)
            except Exception as e:
                print(f"[Query Router] 意图分类失败: {e}", file=sys.stderr)
        else:
            print(f"[Query Router] 意图分类器未初始化或未训练", file=sys.stderr)

        sql_guard = self._looks_like_sql_query(query)
        print(f"[Query Router] SQL Guard 检查: {sql_guard}", file=sys.stderr)
        if not sql_guard and intent_result == "database_query":
            print("[Query Router] 数据库意图被 schema guard 拦截，改为 rag_query", file=sys.stderr)
            intent_result = "rag_query"
        
        # 步骤 2: 使用 LLM 进行细化和分类
        try:
            # 构建提示词，包含意图分类器的结果
            prompt = self._build_classification_prompt(query, intent_result)
            
            # 调用 LLM 进行分类
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 清理可能的 markdown 代码块
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            # 解析 JSON
            result_dict = json.loads(content)
            print(f"[Query Router] LLM 分类结果: {result_dict.get('query_type')}, 理由: {result_dict.get('reasoning')}", file=sys.stderr)
            
            # 步骤 2.5: Self-RAG - 让 LLM 自我验证分类是否合理
            self_rag_result = self._self_rag_verification(query, result_dict, intent_result)
            
            # 如果自我验证认为分类不合理，使用验证后的结果
            if self_rag_result.get("is_reasonable") == False:
                print(f"[Query Router] Self-RAG 认为分类不合理，重新分类", file=sys.stderr)
                result_dict["query_type"] = self_rag_result.get("suggested_type", result_dict.get("query_type"))
                result_dict["reasoning"] = f"Self-RAG 修正: {self_rag_result.get('reasoning', '')}。原分类: {result_dict.get('reasoning', '')}"
            
            # 创建 QueryType 对象
            query_type = QueryType(
                query_type=result_dict.get("query_type", "semantic_search"),
                confidence=result_dict.get("confidence", 0.8),
                reasoning=result_dict.get("reasoning", ""),
                intent=intent_result or ""
            )

            # 如果意图分类器有结果且与 LLM 结果冲突，优先使用意图分类器的结果
            if intent_result:
                mapped_type = self._map_intent_to_query_type(intent_result)
                if mapped_type != query_type.query_type:
                    print(f"[Query Router] 意图分类器结果 ({mapped_type}) 与 LLM 结果 ({query_type.query_type}) 冲突，优先使用意图分类器结果", file=sys.stderr)
                    query_type.query_type = mapped_type
                    query_type.reasoning = f"优先使用意图分类器结果: {intent_result} (LLM 原判断: {query_type.reasoning})"

            if query_type.query_type == "sql_search" and not sql_guard:
                print("[Query Router] 查询不匹配当前数据库 schema，将 sql_search 改为 semantic_search", file=sys.stderr)
                query_type.query_type = "semantic_search"
                query_type.reasoning = f"查询不涉及当前数据库结构化实体，使用语义搜索: {query_type.reasoning}"

            # 如果 SQL 数据库不可用，将 sql_search 改为 semantic_search
            if query_type.query_type == "sql_search" and not self.sql_available:
                print(f"[Query Router] SQL 数据库不可用，将 sql_search 改为 semantic_search", file=sys.stderr)
                query_type.query_type = "semantic_search"
                query_type.reasoning = f"SQL 数据库不可用，使用语义搜索替代: {query_type.reasoning}"

            print(f"[Query Router] 最终分类结果: {query_type.query_type} (置信度: {query_type.confidence})", file=sys.stderr)
            print(f"[Query Router] 理由: {query_type.reasoning}", file=sys.stderr)

            return query_type
            
        except Exception as e:
            print(f"[Query Router] LLM 分类失败: {e}", file=sys.stderr)
            # 回退到意图分类器的结果
            if intent_result:
                mapped_type = self._map_intent_to_query_type(intent_result)
                return QueryType(
                    query_type=mapped_type,
                    confidence=0.6,
                    reasoning=f"LLM 失败，使用意图分类器结果: {intent_result}",
                    intent=intent_result
                )
            # 返回默认分类
            return QueryType(
                query_type="semantic_search",
                confidence=0.5,
                reasoning=f"分类失败，使用默认值: {str(e)}",
                intent=""
            )
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        路由查询到对应的处理器
        
        Args:
            query: 用户查询
            
        Returns:
            dict: 包含查询类型和处理建议的字典
        """
        # 分类查询
        query_type = self.classify_query(query)
        
        # 根据查询类型生成处理建议
        routing_info = {
            "query_type": query_type.query_type,
            "confidence": query_type.confidence,
            "reasoning": query_type.reasoning,
            "intent": query_type.intent,
            "handler": self._get_handler_name(query_type.query_type),
            "suggestions": self._get_suggestions(query_type.query_type, query)
        }
        
        return routing_info
    
    def _get_handler_name(self, query_type: str) -> str:
        """
        根据查询类型获取处理器名称
        
        Args:
            query_type: 查询类型
            
        Returns:
            str: 处理器名称
        """
        handler_map = {
            "sql_search": "SQLGenerator",
            "rag_search": "RAGRetriever",
            "semantic_search": "RAGRetriever",
            "hybrid": "HybridSearchProcessor"
        }
        return handler_map.get(query_type, "RAGRetriever")
    
    def _get_suggestions(self, query_type: str, query: str) -> list:
        """
        根据查询类型生成处理建议
        
        Args:
            query_type: 查询类型
            query: 用户查询
            
        Returns:
            list: 处理建议列表
        """
        suggestions = []
        
        if query_type == "sql_search":
            suggestions = [
                "使用元数据过滤器生成 SQL",
                "执行 SQL 查询",
                "返回结构化数据"
            ]
        elif query_type == "semantic_search":
            suggestions = [
                "使用向量检索查找相关文档",
                "应用重排序提升相关性",
                "生成自然语言回答"
            ]
        elif query_type == "hybrid":
            suggestions = [
                "使用元数据过滤器进行初步筛选",
                "对筛选结果进行向量相似度搜索",
                "融合两种检索结果",
                "生成综合回答"
            ]
        
        return suggestions


# 测试代码
if __name__ == "__main__":
    router = QueryRouter()
    
    test_queries = [
        "有多少个员工",
        "什么是机器学习",
        "查找2024年发布的关于机器学习的文章",
        "查询工程部的员工数量",
        "安息平在库多少盒？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"❓ 查询: {query}")
        print(f"{'='*60}")
        result = router.route_query(query)
        print(f"📊 查询类型: {result['query_type']}")
        print(f"🎯 置信度: {result['confidence']}")
        print(f"💡 理由: {result['reasoning']}")
        print(f"🔧 处理器: {result['handler']}")
        print(f"📝 建议: {result['suggestions']}")
