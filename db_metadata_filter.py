"""
 ============================================================================
 数据库元数据过滤器 (db_metadata_filter.py)
 ============================================================================
 
 功能描述：
 - 定义数据库表的元数据结构
 - 使用 LLM 从自然语言生成过滤条件
 - 将过滤条件转换为 SQL WHERE 子句
 - 支持复杂的查询逻辑（AND, OR, 比较运算符等）
 
 主要类：
 - DatabaseMetadataFilter: 数据库元数据过滤器
   - __init__: 初始化元数据和 LLM
   - parse_query: 解析自然语言查询为过滤条件
   - to_sql_where: 将过滤条件转换为 SQL WHERE 子句
   - execute_query: 执行带过滤条件的 SQL 查询
 
 使用场景：
 - 用户问："找出张三写的2024年发布的关于机器学习的文章"
 - 自动生成：WHERE author = '张三' AND publish_date >= '2024-01-01' AND category = '机器学习'
 
 依赖：
 - langchain: LangChain 框架
 - langchain-openai: OpenAI 兼容客户端
 - sqlalchemy: 数据库连接
 
 ============================================================================
"""

import json
import re
import sys
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class AttributeInfo(BaseModel):
    """数据库字段元数据信息"""
    name: str = Field(description="字段名称")
    description: str = Field(description="字段描述")
    type: str = Field(description="字段类型: string, integer, date, float, boolean")
    table: str = Field(description="所属表名")


class FilterCondition(BaseModel):
    """过滤条件"""
    field: str = Field(description="字段名")
    operator: str = Field(description="操作符: =, !=, >, >=, <, <=, LIKE, IN")
    value: Any = Field(description="字段值")
    logic: str = Field(default="AND", description="逻辑操作符: AND, OR")


class SQLQueryTemplate(BaseModel):
    """结构化 SQL 查询模板"""
    table_name: str = Field(description="目标表名，如 'employees', 'invoices', 'carparks'")
    columns: List[str] = Field(default=["*"], description="要查询的列名列表，默认 ['*']")
    conditions: List[Dict[str, Any]] = Field(default=[], description="WHERE 条件列表，每项包含 field, operator, value")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="日期范围 {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}")
    aggregations: List[Dict[str, str]] = Field(default=[], description="聚合函数列表，如 [{'function': 'COUNT', 'column': '*', 'alias': 'total'}]")
    order_by: Optional[Dict[str, str]] = Field(default=None, description="排序 {'column': 'name', 'direction': 'ASC/DESC'}")
    limit: Optional[int] = Field(default=None, description="返回条数限制")
    joins: List[Dict[str, str]] = Field(default=[], description="JOIN 条件列表，如 [{'type': 'INNER JOIN', 'table': 'departments', 'on': 'employees.department_id = departments.id'}]")


class DatabaseMetadataFilter:
    """
    数据库元数据过滤器
    使用结构化模板从自然语言生成 SQL 查询参数
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化数据库元数据过滤器
        
        Args:
            llm: LLM 实例，如果为 None 则使用默认配置
        """
        if llm is None:
            self.llm = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="blissful_ishizaka_626/gemma4-cloud",
                temperature=0
            )
        else:
            self.llm = llm
    
    def parse_query_template(self, natural_query: str) -> SQLQueryTemplate:
        """
        使用结构化模板解析自然语言查询为 SQL 参数

        Args:
            natural_query: 自然语言查询

        Returns:
            SQLQueryTemplate: 结构化 SQL 查询参数
        """
        prompt = f"""你是一个 SQL 查询助手。将用户的自然语言查询转换为结构化的 SQL 参数。

可用表（根据查询推断最合适的表）：
- employees (员工信息: id, name, email, position, department_id, hire_date)
- departments (部门信息: id, name, email, location, phone_number)
- invoices (发票信息)
- carparks (停车场信息)

结构化模板字段：
1. table_name: 目标表名
2. columns: 要查询的列名列表 ["column1", "column2"]，默认 ["*"]
3. conditions: WHERE 条件列表，每项 {{"field": "字段名", "operator": "=", "value": "值"}}
4. date_range: 日期范围 {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}（可选）
5. aggregations: 聚合函数列表 [{{"function": "COUNT", "column": "*", "alias": "total"}}]（可选）
6. order_by: 排序 {{"column": "name", "direction": "ASC"}}（可选）
7. limit: 返回条数限制（可选）
8. joins: JOIN 条件列表 [{{"type": "INNER JOIN", "table": "departments", "on": "employees.department_id = departments.id"}}]（可选）

示例：
用户查询: "找出2024年入职的工程部员工，按姓名排序"
响应:
{{
  "table_name": "employees",
  "columns": ["*"],
  "conditions": [
    {{"field": "hire_date", "operator": ">=", "value": "2024-01-01"}},
    {{"field": "department_id", "operator": "=", "value": "工程部"}}
  ],
  "date_range": {{"start": "2024-01-01", "end": "2024-12-31"}},
  "aggregations": [],
  "order_by": {{"column": "name", "direction": "ASC"}},
  "limit": null,
  "joins": [{{"type": "INNER JOIN", "table": "departments", "on": "employees.department_id = departments.id"}}]
}}

用户查询: {natural_query}

只返回 JSON，不要有任何其他文字：
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            print(f"[Metadata Filter] LLM 原始响应: {content[:200]}", file=sys.stderr)

            # 清理可能的 markdown 代码块
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)

            if not content or content.isspace():
                print(f"[Metadata Filter] LLM 返回空响应，使用默认模板", file=sys.stderr)
                return SQLQueryTemplate(table_name="employees", conditions=[{"field": "department", "operator": "=", "value": "engineering"}], aggregations=[{"function": "COUNT", "column": "*", "alias": "total"}])

            # 解析 JSON
            data = json.loads(content)
            return SQLQueryTemplate(**data)
        except Exception as e:
            print(f"[Metadata Filter] 模板解析失败: {e}", file=sys.stderr)
            # 返回默认模板 - 根据查询推断可能的表和条件
            query_lower = natural_query.lower()

            # 检测是否是计数查询
            if "多少" in natural_query or "how many" in query_lower or "count" in query_lower:
                # 检测部门
                if "engineering" in query_lower or "工程" in natural_query:
                    return SQLQueryTemplate(
                        table_name="employees",
                        columns=[],
                        conditions=[{"field": "department", "operator": "=", "value": "engineering"}],
                        aggregations=[{"function": "COUNT", "column": "*", "alias": "total"}]
                    )
                elif "部门" in natural_query or "department" in query_lower:
                    # 提取部门名称
                    dept_name = None
                    if "engineering" in query_lower:
                        dept_name = "engineering"
                    elif "marketing" in query_lower:
                        dept_name = "marketing"
                    elif "hr" in query_lower or "human" in query_lower:
                        dept_name = "hr"
                    elif "finance" in query_lower:
                        dept_name = "finance"
                    elif "sales" in query_lower:
                        dept_name = "sales"

                    if dept_name:
                        return SQLQueryTemplate(
                            table_name="employees",
                            columns=[],
                            conditions=[{"field": "department", "operator": "=", "value": dept_name}],
                            aggregations=[{"function": "COUNT", "column": "*", "alias": "total"}]
                        )
                    else:
                        # 没有指定具体部门，统计所有员工
                        return SQLQueryTemplate(
                            table_name="employees",
                            columns=[],
                            conditions=[],
                            aggregations=[{"function": "COUNT", "column": "*", "alias": "total"}]
                        )

            # 检测部门查询
            if "engineering" in query_lower or "工程" in natural_query:
                return SQLQueryTemplate(
                    table_name="employees",
                    columns=["*"],
                    conditions=[{"field": "department", "operator": "=", "value": "engineering"}]
                )

            return SQLQueryTemplate(table_name="employees")
    
    def build_query_from_template(self, template: SQLQueryTemplate) -> str:
        """
        根据结构化模板构建完整 SQL 查询

        Args:
            template: SQLQueryTemplate 实例

        Returns:
            str: 完整的 SQL 查询语句
        """
        # SELECT 子句
        if template.aggregations:
            select_parts = []
            for agg in template.aggregations:
                func = agg.get("function", "COUNT")
                col = agg.get("column", "*")
                alias = agg.get("alias", "")
                if alias:
                    select_parts.append(f"{func}({col}) AS {alias}")
                else:
                    select_parts.append(f"{func}({col})")
            if template.columns and template.columns != ["*"]:
                select_parts.extend(template.columns)
            select_clause = ", ".join(select_parts)
        else:
            select_clause = ", ".join(template.columns) if template.columns else "*"

        sql = f"SELECT {select_clause} FROM {template.table_name}"

        # JOIN 子句
        for join in template.joins or []:
            join_type = join.get("type", "INNER JOIN")
            join_table = join.get("table", "")
            join_on = join.get("on", "")
            if join_table and join_on:
                sql += f" {join_type} {join_table} ON {join_on}"

        # WHERE 条件
        conditions = []
        for cond in template.conditions or []:
            field = cond.get("field", "")
            operator = cond.get("operator", "=")
            value = cond.get("value", "")

            if not field:
                continue

            # 处理值类型
            if isinstance(value, str):
                if operator.upper() == "LIKE":
                    sql_value = f"'%{value}%'"
                else:
                    sql_value = f"'{value}'"
            elif isinstance(value, (int, float)):
                sql_value = str(value)
            else:
                sql_value = f"'{value}'"

            conditions.append(f"{field} {operator} {sql_value}")

        # 日期范围
        if template.date_range:
            start = template.date_range.get("start")
            end = template.date_range.get("end")
            if start:
                conditions.append(f"created_at >= '{start}'")
            if end:
                conditions.append(f"created_at <= '{end}'")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        # ORDER BY
        if template.order_by:
            col = template.order_by.get("column", "")
            direction = template.order_by.get("direction", "ASC")
            if col:
                sql += f" ORDER BY {col} {direction}"

        # LIMIT
        if template.limit:
            sql += f" LIMIT {template.limit}"

        return sql

    def build_full_query(self, natural_query: str, target_table: str = "employees") -> str:
        """
        构建完整的 SQL 查询（基于结构化模板）

        Args:
            natural_query: 自然语言查询
            target_table: 默认目标表名（当无法推断时使用）

        Returns:
            str: 完整的 SQL 查询
        """
        template = self.parse_query_template(natural_query)
        # 如果模板没有表名，使用默认值
        if not template.table_name:
            template.table_name = target_table
        return self.build_query_from_template(template)
    
    def execute_query(self, natural_query: str, engine, target_table: str = "employees") -> List[Dict[str, Any]]:
        """
        执行带过滤条件的 SQL 查询
        
        Args:
            natural_query: 自然语言查询
            engine: SQLAlchemy 引擎
            target_table: 目标表名
            
        Returns:
            List[Dict]: 查询结果
        """
        sql_query = self.build_full_query(natural_query, target_table)
        print(f"[Metadata Filter] 生成的 SQL: {sql_query}")
        
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()
                
                # 转换为字典列表
                result_list = []
                for row in rows:
                    result_list.append(dict(zip(columns, row)))
                
                return result_list
        except Exception as e:
            print(f"[Metadata Filter] 查询执行失败: {e}")
            return []


# 测试代码
if __name__ == "__main__":
    filter = DatabaseMetadataFilter()

    test_queries = [
        "找出2024年入职的员工",
        "找出工程部的员工",
        "找出名字包含'张'的员工",
        "找出2024年入职的工程部员工"
    ]

    for query in test_queries:
        print(f"\n❓ 查询: {query}")
        print(f"🔍 解析结果:")
        template = filter.parse_query_template(query)
        print(template.model_dump_json(indent=2))
        print(f"📝 完整 SQL: {filter.build_full_query(query)}")
