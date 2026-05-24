"""
 ============================================================================
 AI 调用模块 (call_AI.py)
 ============================================================================
 
 功能描述：
 - 封装对 LLM（Ollama/DeepSeek）的调用
 - 支持对话历史传递
 - 生成结构化提示词
 - 处理 AI 响应并提取文本内容
 
 主要类：
 - CallAI: AI 调用封装类
   - __init__: 初始化 AI 客户端
   - generate_prompt: 生成提示词（包含文档和对话历史）
   - call_ollama: 调用 Ollama API
   - _coerce_message_content_to_text: 提取响应文本
 
 提示词特性：
 - 基于文档内容回答
 - 支持逻辑推理
 - 考虑对话历史
 - 处理无相关信息的情况
 
 依赖：
 - openai: OpenAI 兼容客户端（用于 Ollama）
 - ollama: 本地 LLM 服务
 
 ============================================================================
"""

import re
import json
from openai import OpenAI


class CallAI:
    def __init__(
        self,
        question,
        retrieved_docs,
        # 根据情况改变model
        model="blissful_ishizaka_626/gemma4-cloud",
        base_url="http://localhost:11434/v1",
        max_doc_chars=400,
        include_citations=False,
        max_tokens=8192,
        conversation_history=None,
    ):
        self.question = question
        self.retrieved_docs = retrieved_docs
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
        )
        self.max_doc_chars = max_doc_chars
        self.include_citations = include_citations
        self.max_tokens = max_tokens
        self.conversation_history = conversation_history or []

     # 格式化来源文本，提取有用信息并拼接成字符串
    def _format_source_text(self, source):
        if isinstance(source, dict):
            title = source.get("title", "")
            url = source.get("url", "")
            body = source.get("content") or source.get("text") or ""
            parts = [part for part in (title, url, body) if part]
            return "\n".join(parts) if parts else str(source)
        return str(source)
    

 # 截断文本到指定长度，末尾添加省略号
    def _trim_text(self, text):
        if not text:
            return text
        if len(text) <= self.max_doc_chars:
            return text
        return text[: self.max_doc_chars].rstrip() + "..."
    

    def generate_prompt(self):
        sources = []
        for i, source in enumerate(self.retrieved_docs or [], start=1):
            text = self._trim_text(self._format_source_text(source)).strip()
            if text:
                sources.append(f"[{i}] {text}")

        if sources:
            grounding_rule = "必须严格基于提供的文档内容回答问题。"
            source_block = "\n\n【参考文档】：\n" + "\n\n".join(sources)
        else:
            grounding_rule = "当前没有提供文档片段，请明确说明无法回答。"
            source_block = ""

        prompt = (
            f"你是一个专业的问答助手。\n\n"
            f"【核心要求】：\n"
            f"1. 严格基于提供的文档内容回答问题，不得引用文档外的信息。\n"
            f"2. 可以对文档内容进行分析和推理，但不能超出文档范围。\n"
            f"3. 回答要精确、细致。\n"
            f"4. 如果文档中没有相关信息，请明确说明\"根据现有文档无法找到相关信息\"。\n"
            f"5. 不要输出思考过程、推理标记或任何额外信息。\n\n"
            f"{grounding_rule}\n"
            f"{source_block}\n\n"
            f"【回答方式】：\n"
            f"【用户问题】：{self.question}"
        )
        return prompt


# 清理文本，去除思考过程、推理内容和工具调用等非最终答案的部分
    def _strip_reasoning(self, text):
        if not text:
            return ""

        cleaned = text
        cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```(?:thought|thinking|reasoning)?[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)

        lines = []
        for line in cleaned.splitlines():
            lower = line.strip().lower()
            if lower.startswith(("思考", "推理", "analysis", "reasoning", "thought")):
                continue
            lines.append(line)

        return "\n".join(lines).strip()


# 将文本转换为关键词列表，去除重复和无意义的词，限制数量
    def _to_keywords_only(self, text):
        if not text:
            return ""

        normalized = text.replace("\n", "，").replace(";", "，").replace("；", "，")
        parts = [p.strip(" ，。,.:-") for p in normalized.split("，")]
        parts = [p for p in parts if p]

        unique = []
        seen = set()
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
            if len(unique) >= 10:
                break

        return "，".join(unique)

# 标准化查询列表，去除空项和重复项，限制数量
    def _normalize_query_list(self, items):
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


# 将消息内容强制转换为纯文本，处理字符串、列表和字典等不同格式的内容，提取有用文本并拼接成字符串
    def _coerce_message_content_to_text(self, raw_content):
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



# 确保查询列表中至少有一定数量的相关查询，如果不足则基于原始查询生成更多相关查询，使用预定义的模板来扩展查询列表
    def _ensure_min_queries(self, base_query, queries, min_count=5):
        unique = self._normalize_query_list(queries)
        seed = (base_query or "").strip()
        if seed and seed not in unique:
            unique.insert(0, seed)

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

        seed = seed or "该问题"
        for template in templates:
            candidate = template.format(q=seed).strip()
            if candidate and candidate not in unique:
                unique.append(candidate)
            if len(unique) >= min_count:
                break

        return unique[:max(min_count, len(unique))]


# 从文本中提取查询列表，支持纯文本、JSON数组、嵌套结构等多种格式，清理和规范化查询文本
    def _extract_queries_from_text(self, content):
        if not content:
            return []

        text = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return self._normalize_query_list(parsed)
            if isinstance(parsed, dict):
                for key in ("queries", "query", "items", "data", "results"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return self._normalize_query_list(value)
                    if isinstance(value, str) and value.strip():
                        return [value.strip()]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                embedded = json.loads(match.group(0))
                if isinstance(embedded, list):
                    return self._normalize_query_list(embedded)
            except json.JSONDecodeError:
                pass

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = []
        for line in lines:
            line = re.sub(r"^\d+[\.、]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = line.strip("\"' ，。,.")
            if line:
                cleaned.append(line)
        return self._normalize_query_list(cleaned)



  # 调用ollama 本地接口
    def get_muti_query(self,query):
        """生成多个相关查询，返回字符串列表"""
        prompt = f"请基于以下问题生成至少5个类似查询，输出纯JSON数组字符串（例如 [\"q1\",\"q2\"]），不要任何额外文字：{query}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        
        raw_content = getattr(response.choices[0].message, "content", None)
        content = self._coerce_message_content_to_text(raw_content).strip()
        queries = self._extract_queries_from_text(content) if content else []

        if len(queries) < 5:
            retry_prompt = (
                "请仅返回JSON数组，必须包含至少5个不同查询，不要markdown，不要解释。"
                f"\n原始问题：{query}"
            )

            retry_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是查询改写助手，只输出JSON数组。"},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.3,
                max_tokens=2058,
            )

            retry_raw_content = getattr(retry_response.choices[0].message, "content", None)
            retry_content = self._coerce_message_content_to_text(retry_raw_content).strip()
            retry_queries = self._extract_queries_from_text(retry_content) if retry_content else []
            if len(retry_queries) > len(queries):
                queries = retry_queries

        return self._ensure_min_queries(query, queries, min_count=5)

    

    def call_ollama(self):
        """调用 Ollama API 获取答案，支持对话历史"""
        prompt = self.generate_prompt()
        
        # ========== 构建消息列表（包含对话历史） ==========
        messages = [
            {"role": "system", "content": "你是一个严格遵循文档内容的问答助手。用自己的话清晰地解释答案，但绝对不能编造文档中不存在的信息。直接输出答案，不要包含任何思考过程。"}
        ]
        
        # 添加对话历史（限制最近10轮，避免上下文溢出）
        history_limit = 10
        recent_history = self.conversation_history[-history_limit:] if self.conversation_history else []
        
        for msg in recent_history:
            # 将前端的角色格式转换为 OpenAI 格式
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        
        # 添加当前问题（包含检索到的文档）
        messages.append({"role": "user", "content": prompt})
        
        # ========== 调用 Ollama API ==========
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # 降低随机性，提高准确性
            max_tokens=self.max_tokens,
        )

        # ========== 提取答案内容 ==========
        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", "") or ""

        # 对于思考模型，content 可能为空；安全地提取 content 和 reasoning
        answer = self._coerce_message_content_to_text(getattr(message, "content", None)).strip()
        reasoning = self._coerce_message_content_to_text(
            getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        ).strip()

        # 如果 answer 为空，尝试从 model_dump 中提取
        if not answer and hasattr(message, "model_dump"):
            dumped = message.model_dump()
            answer = self._coerce_message_content_to_text(dumped.get("content")).strip()
            if not reasoning:
                reasoning = self._coerce_message_content_to_text(
                    dumped.get("reasoning_content") or dumped.get("reasoning")
                ).strip()

        # ========== 处理思考模型的特殊情况 ==========
        # 思考模型可能将所有 token 用于 reasoning，返回空的 content
        # 如果遇到这种情况，使用明确的最终答案指令重试一次
        if not answer and finish_reason == "length":
            retry_prompt = (
                "请用自己的话直接回答问题，基于提供的文档内容。"
                "不要包含任何思考过程或推理说明。"
                f"问题：{self.question}"
            )

            retry_response = self.client.chat.completions.create(
                model=self.model,
                
                messages=[
                    {"role": "system", "content": "你是一个严格遵循文档的问答助手。用自然的语言解释答案，但只能使用文档中的信息。输出最终答案，不要任何思考过程。"},
                    {"role": "user", "content": retry_prompt},
                ],

                temperature=0.2,
                max_tokens=max(256, self.max_tokens),
            )
            retry_message = retry_response.choices[0].message
            answer = self._coerce_message_content_to_text(getattr(retry_message, "content", None)).strip()
            if not answer and hasattr(retry_message, "model_dump"):
                answer = self._coerce_message_content_to_text(retry_message.model_dump().get("content")).strip()

        answer = self._strip_reasoning(answer)

        # Last-resort fallback: if content is empty, return a short sanitized extract from reasoning.
        if not answer and reasoning:
            reasoning = self._strip_reasoning(reasoning)
            answer = reasoning[:600].strip()

        if not answer:
            answer = "当前模型未返回可用答案，请重试或更换模型。"
        if self.include_citations:
            answer = answer
        answer = self.safety_check(answer)
        return answer

    def call_gpt4(self):
        return self.call_ollama()




   
   # 给答案添加引用，格式化来源信息并拼接到答案后面
    def add_citations(self, answer, sources):
        """给答案添加引用"""
        if not sources:
            return answer

        lines = []
        for i, source in enumerate(sources):
            if isinstance(source, dict):
                title = source.get("title", "")
                url = source.get("url", "")
                source_name = source.get("source", "")
                source_id = source.get("id", "")
                content = (source.get("content") or source.get("text") or "").strip()

                label_parts = []
                if title:
                    label_parts.append(str(title))
                if source_name:
                    label_parts.append(f"source={source_name}")
                if source_id:
                    label_parts.append(f"id={source_id}")
                if url:
                    label_parts.append(str(url))

                label = " | ".join(label_parts).strip() or "未命名来源"
                snippet = self._trim_text(content).replace("\n", " ").strip() if content else ""
                if snippet:
                    lines.append(f"[{i+1}] {label} | 摘要: {snippet}")
                else:
                    lines.append(f"[{i+1}] {label}")
            else:
                text = str(source).replace("\n", " ").strip()
                lines.append(f"[{i+1}] {self._trim_text(text)}")

        return answer + "\n\n参考来源:\n" + "\n".join(lines)



# 安全检查，过滤掉包含敏感内容的答案，返回提示信息
    def safety_check(self, answer):
        harmful_keywords = ["自杀", "违法", "色情", "暴力", "毒品", "赌博", "诈骗"]
        for keyword in harmful_keywords:
            if keyword in answer:
                return "抱歉，这个问题涉及敏感内容，建议咨询专业人士。"
        return answer

    

   


             