"""
 ============================================================================
 SQL 生成器 (sql_generator.py)
 ============================================================================
 
 功能描述：
 - 使用 LangChain SQLDatabaseChain 将自然语言转换为 SQL
 - 使用元数据过滤器生成结构化过滤条件
 - 执行 SQL 查询并返回结果
 - 支持回退到关键词匹配生成 SQL
 - 连接 MySQL 数据库
 - 安全 SQL 执行器（带重试机制和安全验证）
 
 主要类：
 - SQLGenerator: SQL 生成器类
   - __init__: 初始化数据库连接和 LLM
   - generate_sql: 将自然语言转换为 SQL 并执行
   - generate_sql_with_metadata: 使用元数据过滤器生成 SQL
   - _fallback_sql_generation: 回退方案（关键词匹配）
 
 - SafeSQLExecutor: 安全 SQL 执行器
   - __init__: 初始化执行器
   - execute_with_retry: 带重试机制的 SQL 执行
   - _validate_sql: SQL 安全验证
   - _generate_sql: 生成 SQL
 
 工作流程：
 1. 用户输入自然语言查询
 2. 元数据过滤器解析查询为结构化过滤条件
 3. 将过滤条件转换为 SQL WHERE 子句
 4. 执行 SQL 查询
 5. 返回查询结果
 
 回退机制：
 - 如果元数据过滤器失败，使用 LangChain SQLDatabaseChain
 - 如果 LangChain 不可用，使用关键词匹配生成简单 SQL
 - 如果 SQL 执行失败，自动重试并修正
 
 依赖：
 - langchain: LangChain 框架
 - langchain-experimental: SQLDatabaseChain
 - langchain-openai: OpenAI 兼容客户端
 - sqlalchemy: 数据库连接
 - pymysql: MySQL 驱动
 - db_metadata_filter: 元数据过滤器
 
 ============================================================================
"""
import os
import json
import re
import sys
from datetime import datetime, date
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from db_metadata_filter import DatabaseMetadataFilter

# 尝试导入 sqlalchemy，如果不可用则使用回退方案
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    import sys
    print("[SQL Generator] sqlalchemy not available, SQL generation disabled", file=sys.stderr)

# 尝试导入 langchain_experimental，如果不可用则使用回退方案
try:
    from langchain_experimental.sql import SQLDatabaseChain
    from langchain_community.utilities import SQLDatabase
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    import sys
    print("[SQL Generator] langchain_experimental not available, using metadata filter fallback", file=sys.stderr)

def serialize_datetime(obj):
    """
    Convert datetime/date objects to ISO format strings for JSON serialization

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj

# 加载环境变量
load_dotenv()

class SQLGenerator:
    def __init__(self):
        """
        初始化 SQL 生成器
        """
        # 检查依赖是否可用
        if not SQLALCHEMY_AVAILABLE:
            import sys
            print("[SQL Generator] SQLAlchemy not available, SQL generation disabled", file=sys.stderr)
            self.engine = None
            self.db = None
            self.sql_chain = None
            self.safe_executor = None
            self.metadata_filter = None
            return
        
        # 从环境变量获取数据库配置
        db_host = os.getenv('DB_HOST', '127.0.0.1')
        db_port = os.getenv('DB_PORT', '3305')
        db_name = os.getenv('DB_DATABASE', 'company_info')
        db_user = os.getenv('DB_USERNAME', 'rootUser')
        db_password = os.getenv('DB_PASSWORD', '123456')
        
        # 尝试连接数据库
        db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(db_url)
        
        # 尝试使用 LangChain 的 SQLDatabase（如果可用）
        self.db = None
        if LANGCHAIN_AVAILABLE:
            try:
                self.db = SQLDatabase.from_uri(db_url)
                print("[SQL Generator] LangChain SQLDatabase initialized successfully")
            except Exception as e:
                print(f"[SQL Generator] LangChain SQLDatabase failed: {e}, using fallback")
                self.db = None
        else:
            print("[SQL Generator] Using SQLAlchemy engine directly (LangChain not available)")
        
        # 初始化 LLM（使用本地 Ollama）
        self.llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="blissful_ishizaka_626/gemma4-cloud",
            temperature=0
        )
        
        # 创建 SQL 链（仅当 LangChain 可用时）
        if LANGCHAIN_AVAILABLE and self.db:
            try:
                self.sql_chain = SQLDatabaseChain.from_llm(
                    llm=self.llm,
                    db=self.db,
                    verbose=True,
                    use_query_checker=True,  # 自动检查 SQL 语法
                    return_intermediate_steps=True
                )
                print("[SQL Generator] SQLDatabaseChain created successfully")
            except Exception as e:
                print(f"[SQL Generator] SQLDatabaseChain failed: {e}")
                self.sql_chain = None
        else:
            self.sql_chain = None
        
        # 创建安全 SQL 执行器
        if self.db:
            self.safe_executor = SafeSQLExecutor(self.db, self.llm)
        else:
            # 如果没有 LangChain SQLDatabase，使用原生引擎
            self.safe_executor = SafeSQLExecutor(self.engine, self.llm)
        
        # 创建元数据过滤器
        self.metadata_filter = DatabaseMetadataFilter(llm=self.llm)
    
    def generate_sql(self, question: str, use_safe_executor=False, use_metadata_filter=True) -> dict:
        """
        将自然语言问题转换为 SQL 并执行

        Args:
            question: 自然语言问题
            use_safe_executor: 是否使用安全执行器（带重试机制）
            use_metadata_filter: 是否使用元数据过滤器

        Returns:
            dict: {
                'sql': 生成的 SQL 语句,
                'result': 查询结果,
                'success': 是否成功,
                'error': 错误信息（如果有）
            }
        """
        # 检查依赖是否可用
        if not SQLALCHEMY_AVAILABLE:
            return {
                'sql': None,
                'result': None,
                'success': False,
                'error': 'SQLAlchemy not available, SQL generation disabled'
            }

        # 优先使用真实数据库 schema 生成 SQL
        try:
            result = self.generate_sql_from_schema(question)
            if result.get('success'):
                return result
        except Exception as e:
            print(f"[SQL Generator] Schema-aware SQL generation failed: {e}", file=sys.stderr)

        # 回退使用元数据过滤器
        if use_metadata_filter and self.metadata_filter:
            try:
                result = self.generate_sql_with_metadata(question)
                if result['success']:
                    return result
            except Exception as e:
                print(f"[SQL Generator] Metadata filter error: {e}", file=sys.stderr)
                # 回退到简单 SQL 生成
                return self._fallback_sql_generation(question)

        # 回退：使用简单的关键词匹配生成 SQL
        return self._fallback_sql_generation(question)

    def _get_database_schema(self) -> str:
        if not self.engine:
            return ""

        schema_lines = []
        with self.engine.connect() as conn:
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            for table_row in tables:
                table_name = table_row[0]
                columns = conn.execute(text(f"SHOW COLUMNS FROM `{table_name}`")).fetchall()
                col_parts = []
                for col in columns:
                    col_parts.append(f"{col[0]} {col[1]}")
                schema_lines.append(f"{table_name}({', '.join(col_parts)})")

        return "\n".join(schema_lines)

    def _clean_sql(self, content: str) -> str:
        content = content.strip()
        content = re.sub(r"```sql\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*", "", content)
        match = re.search(r"select\s+.*", content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(0)
        content = content.strip().rstrip(";")
        return content

    def _execute_sql(self, sql: str) -> dict:
        if not re.match(r"^\s*select\b", sql, flags=re.IGNORECASE):
            return {
                'sql': sql,
                'result': None,
                'success': False,
                'error': 'Only SELECT SQL is allowed'
            }

        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            result_list = [dict(zip(columns, row)) for row in rows]
            result_list = serialize_datetime(result_list)

        return {
            'sql': sql,
            'result': result_list,
            'success': True,
            'error': None
        }

    def generate_sql_from_schema(self, question: str) -> dict:
        schema = self._get_database_schema()
        prompt = f"""你是 MySQL SQL 生成器。请根据真实数据库 schema，把用户问题转换成一条可执行的 SELECT SQL。

数据库 schema：
{schema}

重要规则：
1. 只返回 SQL，不要解释，不要 Markdown。
2. 只能生成 SELECT 查询。
3. 如果 JOIN 后多个表有同名字段，必须使用表别名限定字段，例如 e.name、d.name。
4. employees 表别名用 e，departments 表别名用 d。
5. 查询部门名称时，employees.department_id 应 JOIN departments.id，并用 departments.name 过滤。
6. 如果用户问“多少人/多少员工”，但也问“谁最大/每个人年纪”，不要只返回 COUNT；要返回每个人的 name、age、position、department，并用 ORDER BY age DESC，让 AI 基于结果回答数量和最大年龄。

用户问题：
{question}

SQL："""
        response = self.llm.invoke(prompt)
        sql = self._clean_sql(response.content)
        print(f"[SQL Generator] Schema-aware generated SQL: {sql}", file=sys.stderr)
        return self._execute_sql(sql)
    
    def generate_sql_with_metadata(self, question: str, target_table: str = "employees") -> dict:
        """
        使用元数据过滤器生成 SQL 并执行
        
        Args:
            question: 自然语言问题
            target_table: 目标表名
            
        Returns:
            dict: {
                'sql': 生成的 SQL 语句,
                'result': 查询结果,
                'success': 是否成功,
                'error': 错误信息（如果有）
            }
        """
        # 检查依赖是否可用
        if not SQLALCHEMY_AVAILABLE or not self.engine:
            return {
                'sql': None,
                'result': None,
                'success': False,
                'error': 'SQLAlchemy not available, SQL generation disabled'
            }
        
        try:
            # 使用元数据过滤器生成 SQL
            sql_query = self.metadata_filter.build_full_query(question, target_table)
            print(f"[SQL Generator] Metadata filter generated SQL: {sql_query}")
            
            # 执行 SQL
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()
                
                # 转换为字典列表
                result_list = []
                for row in rows:
                    result_list.append(dict(zip(columns, row)))

                # 序列化 datetime 对象
                result_list = serialize_datetime(result_list)

                return {
                    'sql': sql_query,
                    'result': result_list,
                    'success': True,
                    'error': None
                }
        except Exception as e:
            print(f"[SQL Generator] Metadata filter execution error: {e}")
            return self._fallback_sql_generation(question)
    
    
    def _fallback_sql_generation(self, question: str) -> dict:
        """
        回退方案：使用简单的关键词匹配生成 SQL
        """
        question_lower = question.lower()
        
        # 简单的 SQL 生成规则
        if (
            ('engineering' in question_lower or '工程' in question)
            and ('年纪' in question or '年龄' in question or 'age' in question_lower or '最大' in question)
        ):
            sql = """
            SELECT
                e.name,
                e.age,
                e.position,
                d.name AS department
            FROM employees e
            INNER JOIN departments d ON e.department_id = d.id
            WHERE LOWER(d.name) = 'engineering'
            ORDER BY e.age DESC
            """
        elif (
            ('engineering' in question_lower or '工程' in question)
            and ('员工' in question or 'employee' in question_lower or '人' in question)
        ):
            sql = """
            SELECT
                e.name,
                e.age,
                e.position,
                d.name AS department
            FROM employees e
            INNER JOIN departments d ON e.department_id = d.id
            WHERE LOWER(d.name) = 'engineering'
            ORDER BY e.age DESC
            """
        elif '员工' in question or 'employee' in question_lower:
            if '数量' in question or 'count' in question_lower or '多少' in question:
                sql = "SELECT COUNT(*) as employee_count FROM employees"
            elif '部门' in question or 'department' in question_lower:
                sql = "SELECT e.id, e.name, e.email, d.name as department FROM employees e JOIN departments d ON e.department_id = d.id"
            else:
                sql = "SELECT * FROM employees"
        elif '部门' in question or 'department' in question_lower:
            sql = "SELECT * FROM departments"
        else:
            sql = "SELECT 'Query not understood' as message"
        
        # 执行 SQL
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()
                
                # 转换为字典列表
                result_list = []
                for row in rows:
                    result_list.append(dict(zip(columns, row)))

                result_list = serialize_datetime(result_list)
                
                return {
                    'sql': sql,
                    'result': result_list,
                    'success': True,
                    'error': None
                }
        except Exception as e:
            return {
                'sql': sql,
                'result': None,
                'success': False,
                'error': str(e)
            }


class SafeSQLExecutor:
    """
    安全 SQL 执行器 - 带重试机制和安全验证
    """
    def __init__(self, db_or_engine, llm, max_retries=3):
        """
        初始化安全 SQL 执行器
        
        Args:
            db_or_engine: 数据库连接对象（LangChain SQLDatabase 或 SQLAlchemy Engine）
            llm: LLM 实例
            max_retries: 最大重试次数
        """
        self.db_or_engine = db_or_engine
        self.llm = llm
        self.max_retries = max_retries
        # 检测传入的是 LangChain SQLDatabase 还是 SQLAlchemy Engine
        if SQLALCHEMY_AVAILABLE:
            from sqlalchemy.engine import Engine
            self.is_engine = isinstance(db_or_engine, Engine)
        else:
            self.is_engine = False
    
    def execute_with_retry(self, natural_query):
        """
        带重试机制的 SQL 执行
        
        Args:
            natural_query: 自然语言查询
            
        Returns:
            查询结果或错误信息
        """
        for attempt in range(self.max_retries):
            try:
                # 生成 SQL
                sql = self._generate_sql(natural_query)
                print(f"尝试 {attempt + 1}: {sql}")
                
                # 执行前验证
                if not self._validate_sql(sql):
                    raise ValueError("SQL验证失败：可能包含危险操作")
                
                # 执行查询（根据传入的是 Engine 还是 SQLDatabase）
                if self.is_engine:
                    # 使用 SQLAlchemy Engine
                    with self.db_or_engine.connect() as conn:
                        result = conn.execute(text(sql))
                        rows = result.fetchall()
                        columns = result.keys()
                        # 转换为字典列表
                        result_list = []
                        for row in rows:
                            result_list.append(dict(zip(columns, row)))
                        return result_list
                else:
                    # 使用 LangChain SQLDatabase
                    result = self.db_or_engine.execute(sql)
                    return result
            except Exception as e:
                error_msg = str(e)
                print(f"❌ 错误: {error_msg}")
                if attempt < self.max_retries - 1:
                    # 让 LLM 修正错误
                    natural_query = f"""上次查询失败了。
                    原查询: {natural_query}
                    生成的SQL: {sql}
                    错误信息: {error_msg}
                    请修正SQL并重试。
                    """
                else:
                    return f"查询失败，已重试{self.max_retries}次: {error_msg}"
    
    def _validate_sql(self, sql):
        """
        SQL 安全验证
        
        Args:
            sql: SQL 语句
            
        Returns:
            bool: 是否安全
        """
        # 检查危险关键词
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "UPDATE", "INSERT"]
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False
        
        # 检查是否只读查询
        if not sql_upper.strip().startswith("SELECT"):
            return False
        
        return True
    
    def _generate_sql(self, query):
        """
        生成 SQL
        
        Args:
            query: 自然语言查询
            
        Returns:
            str: SQL 语句
        """
        prompt = f"""将以下自然语言转换为SQL查询。
        数据库schema:
        - employees表: id, name, email, department_id, position
        - departments表: id, name, email, location, phone_number
        查询: {query}
        只返回SQL语句，不要有其他文字：
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()


# 测试代码
if __name__ == "__main__":
    generator = SQLGenerator()
    
    test_questions = [
        "有多少个员工",
        "查询所有部门",
        "员工信息"
    ]
    
    print("=== 测试 SQLGenerator ===")
    for question in test_questions:
        print(f"\n❓ 问题: {question}")
        result = generator.generate_sql(question)
        print(f"🔍 生成的SQL: {result['sql']}")
        print(f"✅ 结果: {result['result']}")
    
    print("\n=== 测试 SafeSQLExecutor ===")
    # 使用 SafeSQLExecutor
    if generator.db:
        executor = SafeSQLExecutor(generator.db, generator.llm)
        result = executor.execute_with_retry("有多少个员工")
        print(f"🔍 安全执行结果: {result}")
    else:
        print("SafeSQLExecutor 需要 langchain-community 的 SQLDatabase 支持")
