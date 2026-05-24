"""
 ============================================================================
 混合搜索处理器 (hybrid_search.py)
 ============================================================================
 
 功能描述：
 - 实现混合搜索功能，结合元数据过滤和语义检索
 - 先使用元数据过滤器筛选文档，再进行向量相似度搜索
 - 融合结构化查询和语义检索的优势
 - 支持多种检索策略和结果融合
 
 主要类：
 - HybridSearchProcessor: 混合搜索处理器类
   - __init__: 初始化处理器（元数据过滤器、向量检索器、重排序器）
   - search: 执行混合搜索
   - _filter_by_metadata: 使用元数据过滤文档
   - _semantic_search: 对筛选结果进行语义搜索
   - _fuse_results: 融合过滤和语义搜索结果
 
 工作流程：
 1. 接收用户查询
 2. 使用元数据过滤器解析查询条件
 3. 对向量数据库进行元数据过滤
 4. 对过滤结果进行向量相似度搜索
 5. 融合两种检索结果
 6. 返回最终结果
 
 检索策略：
 - 元数据过滤：基于结构化字段（作者、日期、类别等）
 - 语义搜索：基于向量相似度
 - 结果融合：加权融合或 RRF（Reciprocal Rank Fusion）
 
 依赖：
 - langchain: LangChain 框架
 - langchain-openai: OpenAI 兼容客户端
 - local FAISS vector store: 向量数据库
 - db_metadata_filter: 元数据过滤器
 - data_server: 向量检索服务
 
 ============================================================================
"""

import sys
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from db_metadata_filter import DatabaseMetadataFilter
import json


class HybridSearchProcessor:
    """
    混合搜索处理器
    结合元数据过滤和语义检索进行混合搜索
    """
    
    def __init__(
        self,
        metadata_filter: Optional[DatabaseMetadataFilter] = None,
        vector_retriever=None,
        reranker=None,
        llm: Optional[ChatOpenAI] = None
    ):
        """
        初始化混合搜索处理器
        
        Args:
            metadata_filter: 元数据过滤器实例
            vector_retriever: 向量检索器实例
            reranker: 重排序器实例
            llm: LLM 实例，如果为 None 则使用默认配置
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
        
        # 元数据过滤器
        self.metadata_filter = metadata_filter or DatabaseMetadataFilter(llm=self.llm)
        
        # 向量检索器
        self.vector_retriever = vector_retriever
        
        # 重排序器
        self.reranker = reranker
        
        # 向量数据库的元数据字段定义
        self.vector_metadata_fields = {
            "author": "文档作者",
            "publish_date": "发布日期",
            "category": "文档类别",
            "rating": "文档评分"
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_metadata_filter: bool = True,
        use_semantic_search: bool = True,
        fusion_method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        执行混合搜索
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            use_metadata_filter: 是否使用元数据过滤
            use_semantic_search: 是否使用语义搜索
            fusion_method: 融合方法 (weighted, rrf)
            
        Returns:
            dict: 包含搜索结果和元信息的字典
        """
        print(f"[Hybrid Search] 开始混合搜索: {query}", file=sys.stderr)
        
        # 解析查询为元数据过滤条件
        metadata_filters = None
        if use_metadata_filter:
            metadata_filters = self._parse_query_to_filters(query)
            print(f"[Hybrid Search] 元数据过滤条件: {metadata_filters}", file=sys.stderr)
        
        # 执行元数据过滤
        filtered_docs = []
        if use_metadata_filter and metadata_filters:
            filtered_docs = self._filter_by_metadata(metadata_filters, top_k * 2)
            print(f"[Hybrid Search] 过滤后文档数: {len(filtered_docs)}", file=sys.stderr)
        
        # 执行语义搜索
        semantic_docs = []
        if use_semantic_search:
            semantic_docs = self._semantic_search(query, top_k * 2)
            print(f"[Hybrid Search] 语义搜索文档数: {len(semantic_docs)}", file=sys.stderr)
        
        # 融合结果
        if use_metadata_filter and use_semantic_search:
            final_results = self._fuse_results(
                filtered_docs,
                semantic_docs,
                top_k,
                fusion_method
            )
        elif use_metadata_filter:
            final_results = filtered_docs[:top_k]
        else:
            final_results = semantic_docs[:top_k]
        
        # 重排序
        if self.reranker and len(final_results) > 1:
            final_results = self._rerank_results(query, final_results)
            final_results = final_results[:top_k]
        
        return {
            "query": query,
            "results": final_results,
            "metadata_filters": metadata_filters,
            "filtered_count": len(filtered_docs),
            "semantic_count": len(semantic_docs),
            "final_count": len(final_results),
            "fusion_method": fusion_method
        }
    
    def _parse_query_to_filters(self, query: str) -> Dict[str, Any]:
        """
        将自然语言查询解析为元数据过滤条件
        
        Args:
            query: 用户查询
            
        Returns:
            dict: 元数据过滤条件
        """
        # 构建提示词
        prompt = f"""你是一个元数据过滤器。将用户的自然语言查询转换为结构化的过滤条件。

可用的元数据字段：
- author: 文档作者
- publish_date: 发布日期 (格式: YYYY-MM-DD)
- category: 文档类别
- rating: 文档评分 (1-5)

用户查询: {query}

请生成 JSON 格式的过滤条件：
{{
  "filters": [
    {{"field": "author", "operator": "=", "value": "张三"}},
    {{"field": "publish_date", "operator": ">=", "value": "2024-01-01"}}
  ]
}}

只返回 JSON，不要有其他文字：
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 清理 markdown
            content = content.replace("```json", "").replace("```", "")
            
            # 解析 JSON
            result = json.loads(content)
            return result.get("filters", [])
        except Exception as e:
            print(f"[Hybrid Search] 解析过滤条件失败: {e}", file=sys.stderr)
            return []
    
    def _filter_by_metadata(self, filters: List[Dict[str, Any]], top_k: int) -> List[Document]:
        """
        使用元数据过滤文档
        
        Args:
            filters: 过滤条件列表
            top_k: 返回结果数量
            
        Returns:
            List[Document]: 过滤后的文档列表
        """
        try:
            # 如果有向量检索器，使用它的元数据过滤功能
            if self.vector_retriever:
                # 构建元数据过滤字典
                filter_dict = {}
                for filter_item in filters:
                    field = filter_item.get("field")
                    operator = filter_item.get("operator")
                    value = filter_item.get("value")
                    
                    if operator == "=":
                        filter_dict[field] = value
                    elif operator == ">=":
                        filter_dict[field] = {"$gte": value}
                    elif operator == "<=":
                        filter_dict[field] = {"$lte": value}
                    elif operator == ">":
                        filter_dict[field] = {"$gt": value}
                    elif operator == "<":
                        filter_dict[field] = {"$lt": value}
                
                # 执行带过滤的检索
                if filter_dict:
                    results = self.vector_retriever.get_relevant_documents(
                        "",
                        filter=filter_dict,
                        k=top_k
                    )
                    return results
        except Exception as e:
            print(f"[Hybrid Search] 元数据过滤失败: {e}", file=sys.stderr)
        
        return []
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Document]:
        """
        执行语义搜索 (纯向量相似度，不使用元数据过滤)
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Document]: 语义搜索结果
        """
        try:
            if self.vector_retriever:
                # 纯语义检索，不应用任何元数据过滤
                results = self.vector_retriever.get_relevant_documents(
                    query,
                    k=top_k
                )
                return results
        except Exception as e:
            print(f"[Hybrid Search] 语义搜索失败: {e}", file=sys.stderr)
        
        return []
    
    def _fuse_results(
        self,
        filtered_docs: List[Document],
        semantic_docs: List[Document],
        top_k: int,
        fusion_method: str = "weighted"
    ) -> List[Document]:
        """
        融合过滤和语义搜索结果
        
        Args:
            filtered_docs: 元数据过滤结果
            semantic_docs: 语义搜索结果
            top_k: 返回结果数量
            fusion_method: 融合方法
            
        Returns:
            List[Document]: 融合后的结果
        """
        if fusion_method == "weighted":
            return self._weighted_fusion(filtered_docs, semantic_docs, top_k)
        elif fusion_method == "rrf":
            return self._rrf_fusion(filtered_docs, semantic_docs, top_k)
        else:
            # 默认：简单合并去重
            seen = set()
            final_results = []
            for doc in filtered_docs + semantic_docs:
                doc_id = doc.metadata.get("id", str(doc.page_content))
                if doc_id not in seen:
                    seen.add(doc_id)
                    final_results.append(doc)
                    if len(final_results) >= top_k:
                        break
            return final_results
    
    def _weighted_fusion(
        self,
        filtered_docs: List[Document],
        semantic_docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        加权融合结果
        
        Args:
            filtered_docs: 元数据过滤结果
            semantic_docs: 语义搜索结果
            top_k: 返回结果数量
            
        Returns:
            List[Document]: 融合后的结果
        """
        # 为每个文档计算融合得分
        doc_scores = {}
        
        # 元数据过滤结果权重 0.4
        for i, doc in enumerate(filtered_docs):
            doc_id = doc.metadata.get("id", str(doc.page_content))
            score = (len(filtered_docs) - i) / len(filtered_docs) * 0.4
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # 语义搜索结果权重 0.6
        for i, doc in enumerate(semantic_docs):
            doc_id = doc.metadata.get("id", str(doc.page_content))
            score = (len(semantic_docs) - i) / len(semantic_docs) * 0.6
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # 按得分排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建结果列表
        result_map = {doc.metadata.get("id", str(doc.page_content)): doc for doc in filtered_docs + semantic_docs}
        final_results = []
        for doc_id, _ in sorted_docs[:top_k]:
            if doc_id in result_map:
                final_results.append(result_map[doc_id])
        
        return final_results
    
    def _rrf_fusion(
        self,
        filtered_docs: List[Document],
        semantic_docs: List[Document],
        top_k: int,
        k: int = 60
    ) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合
        
        Args:
            filtered_docs: 元数据过滤结果
            semantic_docs: 语义搜索结果
            top_k: 返回结果数量
            k: RRF 参数
            
        Returns:
            List[Document]: 融合后的结果
        """
        doc_scores = {}
        
        # 计算元数据过滤的 RRF 得分
        for i, doc in enumerate(filtered_docs):
            doc_id = doc.metadata.get("id", str(doc.page_content))
            score = 1 / (k + i + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # 计算语义搜索的 RRF 得分
        for i, doc in enumerate(semantic_docs):
            doc_id = doc.metadata.get("id", str(doc.page_content))
            score = 1 / (k + i + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # 按得分排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建结果列表
        result_map = {doc.metadata.get("id", str(doc.page_content)): doc for doc in filtered_docs + semantic_docs}
        final_results = []
        for doc_id, _ in sorted_docs[:top_k]:
            if doc_id in result_map:
                final_results.append(result_map[doc_id])
        
        return final_results
    
    def _rerank_results(self, query: str, docs: List[Document]) -> List[Document]:
        """
        重排序结果
        
        Args:
            query: 查询文本
            docs: 文档列表
            
        Returns:
            List[Document]: 重排序后的文档列表
        """
        if self.reranker:
            try:
                # 使用重排序器重新排序
                reranked = self.reranker.rerank(query, docs)
                return reranked
            except Exception as e:
                print(f"[Hybrid Search] 重排序失败: {e}", file=sys.stderr)
        
        return docs


# 测试代码
if __name__ == "__main__":
    processor = HybridSearchProcessor()
    
    test_query = "查找2024年发布的关于机器学习的文章"
    print(f"测试查询: {test_query}")
    
    result = processor.search(test_query, top_k=5)
    print(f"\n搜索结果:")
    print(f"查询: {result['query']}")
    print(f"过滤条件: {result['metadata_filters']}")
    print(f"过滤结果数: {result['filtered_count']}")
    print(f"语义搜索结果数: {result['semantic_count']}")
    print(f"最终结果数: {result['final_count']}")
