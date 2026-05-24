/**
 * ============================================================================
 * 后端服务器文件 (server.js)
 * ============================================================================
 * 
 * 功能描述：
 * - Node.js Express 服务器，提供 RESTful API
 * - 处理前端请求并路由到 Python 后端
 * - 管理数据库查询（MySQL）
 * - 集成意图分类和 SQL 生成
 * 
 * 主要 API 端点：
 * - GET /api/health - 健康检查
 * - POST /api/query - 普通 LLM 查询
 * - POST /api/rag - RAG 检索增强生成查询
 * - POST /api/deep-think - 深度思考模式
 * - POST /api/upload-text - 上传文本到向量数据库
 * - POST /api/classify-intent - 意图分类
 * - POST /api/database-query - 数据库查询
 * 
 * 架构：
 * 前端 -> Node.js 后端 -> Python api_handler.py -> 各种处理器
 * 
 * 依赖：
 * - express: Web 框架
 * - mysql2: MySQL 数据库连接
 * - dotenv: 环境变量管理
 * 
 * ============================================================================
 */

const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env') });
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const fs = require('fs');
const mysql = require('mysql2/promise');

const app = express();
const PORT = process.env.PORT || 3000;
const frontendRoot = path.join(__dirname, '../frontend');

app.use(cors());
app.use(bodyParser.json());
app.use('/css', express.static(path.join(frontendRoot, 'css')));
app.use('/js', express.static(path.join(frontendRoot, 'js')));
app.use(express.static(path.join(frontendRoot, 'public')));

function findPythonPath() {
    // 1. Check PYTHON_PATH env variable
    if (process.env.PYTHON_PATH && fs.existsSync(process.env.PYTHON_PATH)) {
        console.log(`[DEBUG] Using PYTHON_PATH from env: ${process.env.PYTHON_PATH}`);
        return process.env.PYTHON_PATH;
    }

    // 1.5. Hardcoded fallback to Python that has faiss installed
    const hardcodedPython = 'C:\\Users\\billychen\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe';
    if (fs.existsSync(hardcodedPython)) {
        console.log(`[DEBUG] Using hardcoded Python path: ${hardcodedPython}`);
        return hardcodedPython;
    }

    // 2a. Check virtual environment Scripts\python.exe (standard venv) - PRIORITY
    const venvPython = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
    if (fs.existsSync(venvPython)) {
        return venvPython;
    }

    // 2b. Check pyvenv.cfg for uv/poetry managed Python - ONLY if venv doesn't exist
    const pyvenvCfg = path.join(__dirname, '..', '.venv', 'pyvenv.cfg');
    if (fs.existsSync(pyvenvCfg) && !fs.existsSync(venvPython)) {
        try {
            const cfgContent = fs.readFileSync(pyvenvCfg, 'utf8');
            const homeMatch = cfgContent.match(/^home\s*=\s*(.+)$/m);
            if (homeMatch) {
                const homePath = homeMatch[1].trim();
                const candidate = path.join(homePath, 'python.exe');
                if (fs.existsSync(candidate)) {
                    return candidate;
                }
            }
        } catch (e) {
            // Ignore parse errors
        }
    }

    // 3. Check common Windows Python installation paths
    const commonPaths = [
        'C:\\Python311\\python.exe',
        'C:\\Python310\\python.exe',
        'C:\\Python39\\python.exe',
        'C:\\Python312\\python.exe',
        'C:\\Users\\billychen\\AppData\\Local\\Python\\bin\\python.exe',
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'python.exe'),
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python310', 'python.exe'),
        path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python312', 'python.exe'),
        path.join(process.env.APPDATA || '', 'Python', 'Python311', 'python.exe'),
    ];
    for (const p of commonPaths) {
        if (fs.existsSync(p)) {
            return p;
        }
    }

    // 4. Scan user profile directory for Python installations
    const userProfile = process.env.USERPROFILE || process.env.HOME || '';
    if (userProfile) {
        const userSearchPaths = [
            path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python'),
            path.join(userProfile, 'anaconda3'),
            path.join(userProfile, 'miniconda3'),
            path.join(userProfile, '.pyenv', 'pyenv-win', 'versions'),
            path.join(userProfile, '.conda'),
        ];
        for (const dir of userSearchPaths) {
            if (fs.existsSync(dir)) {
                try {
                    const entries = fs.readdirSync(dir, { withFileTypes: true });
                    for (const entry of entries) {
                        if (entry.isDirectory()) {
                            const candidate = path.join(dir, entry.name, 'python.exe');
                            if (fs.existsSync(candidate)) {
                                return candidate;
                            }
                        }
                        if (entry.isFile() && entry.name === 'python.exe') {
                            return path.join(dir, entry.name);
                        }
                    }
                } catch (e) {
                    // Ignore permission errors
                }
            }
        }
    }

    // 5. Scan all user directories under C:\Users
    const usersDir = 'C:\\Users';
    if (fs.existsSync(usersDir)) {
        try {
            const userDirs = fs.readdirSync(usersDir, { withFileTypes: true })
                .filter(e => e.isDirectory() && !e.name.startsWith('.'))
                .map(e => path.join(usersDir, e.name));
            
            for (const userDir of userDirs) {
                const commonPaths = [
                    path.join(userDir, 'AppData', 'Local', 'Programs', 'Python'),
                    path.join(userDir, 'anaconda3'),
                    path.join(userDir, 'miniconda3'),
                    path.join(userDir, '.pyenv', 'pyenv-win', 'versions'),
                    path.join(userDir, '.conda'),
                ];
                for (const dir of commonPaths) {
                    if (!fs.existsSync(dir)) continue;
                    try {
                        const entries = fs.readdirSync(dir, { withFileTypes: true });
                        for (const entry of entries) {
                            if (entry.isDirectory()) {
                                const candidate = path.join(dir, entry.name, 'python.exe');
                                if (fs.existsSync(candidate)) {
                                    return candidate;
                                }
                            }
                            if (entry.isFile() && entry.name === 'python.exe') {
                                return path.join(dir, entry.name);
                            }
                        }
                    } catch (e) {
                        // Ignore permission errors
                    }
                }
            }
        } catch (e) {
            // Ignore errors reading C:\Users
        }
    }

    // 6. Fallback to 'python' command (requires PATH)
    return 'python';
}

function callPythonScript(scriptName, args = []) {
    return new Promise((resolve, reject) => {
        const pythonPath = findPythonPath();
        console.log(`[DEBUG] Using Python path: ${pythonPath}`);

        if (!fs.existsSync(pythonPath) && pythonPath !== 'python') {
            reject(new Error(
                `Python not found at: ${pythonPath}\n` +
                `Please set PYTHON_PATH environment variable to your python.exe location, ` +
                `or ensure Python is installed and on your PATH.`
            ));
            return;
        }

        const scriptPath = path.join(__dirname, '..', scriptName);
        
        // Add .venv site-packages to PYTHONPATH for uv-managed Python
        const venvSitePackages = path.join(__dirname, '..', '.venv', 'Lib', 'site-packages');
        let pythonPathEnv = process.env.PYTHONPATH || '';
        if (fs.existsSync(venvSitePackages)) {
            pythonPathEnv = pythonPathEnv 
                ? `${venvSitePackages}${path.delimiter}${pythonPathEnv}` 
                : venvSitePackages;
        }

        const pythonProcess = spawn(
            pythonPath,
            ['-X', 'utf8', scriptPath, ...args],
            {
                env: {
                    ...process.env,
                    PYTHONUTF8: '1',
                    PYTHONIOENCODING: 'utf-8',
                    PYTHONPATH: pythonPathEnv,
                },
            }
        );

        let stdout = '';
        let stderr = '';


        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString('utf8');
        });


        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString('utf8');
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                resolve(stdout);
                return;
            }
            reject(new Error(`Python script failed (code=${code}): ${stderr || stdout || 'no output'}`));
        });

        pythonProcess.on('error', (err) => {
            reject(err);
        });
    });
}

async function runApiHandler(payload) {
    const args = JSON.stringify(payload);
    const result = await callPythonScript('api_handler.py', [args]);
    return result.trim();
}

app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'LLM Backend is running' });
});

app.post('/api/query', async (req, res) => {
    try {
        const { question, model = 'blissful_ishizaka_626/gemma4-cloud', max_tokens = 256 } = req.body;
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        const answer = await runApiHandler({
            question,
            type: 'smart_query',
            model,
            max_tokens,
        });

        res.json({
            success: true,
            answer,
            model,
        });
    } catch (error) {
        console.error('Error processing query:', error);
        res.status(500).json({ error: error.message });
    }
});




app.post('/api/rag', async (req, res) => {
    try {
        const { question, model = 'blissful_ishizaka_626/gemma4-cloud', max_tokens = 4096, top_k = 8, per_retriever_k = 16, self_rag_max_retries = 1 } = req.body;
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        // 使用智能路由自动判断查询类型（SQL/RAG）
        const answer = await runApiHandler({
            question,
            type: 'smart_query',
            model,
            max_tokens,
            top_k,
            per_retriever_k,
            self_rag_max_retries,
        });

        res.json({ success: true, answer });
    } catch (error) {
        console.error('Error in RAG query:', error);
        res.status(500).json({ error: error.message });
    }
});



app.post('/api/deep-think', async (req, res) => {
    try {
        const { query, model = 'blissful_ishizaka_626/gemma4-cloud', max_tokens = 256 } = req.body;
        if (!query) {
            return res.status(400).json({ error: 'Query is required' });
        }

        const answer = await runApiHandler({
            question: query,
            type: 'deepthink',
            model,
            max_tokens,
        });

        res.json({ success: true, answer });
    } catch (error) {
        console.error('Error in Deep Think query:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/upload-text', async (req, res) => {
    try {
        const {
            text,
            source = 'manual_upload',
            chunk_size = 500,
            chunk_overlap = 50,
        } = req.body;

        if (!text || !String(text).trim()) {
            return res.status(400).json({ success: false, error: 'Text is required' });
        }

        const raw = await runApiHandler({
            type: 'upload',
            text,
            source,
            chunk_size,
            chunk_overlap,
        });

        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (_err) {
            throw new Error(`Invalid upload response: ${raw}`);
        }

        if (!parsed.success) {
            return res.status(500).json(parsed);
        }

        return res.json(parsed);
    } catch (error) {
        console.error('Error uploading text:', error);
        return res.status(500).json({ success: false, error: error.message });
    }
});

app.get('/', (req, res) => {
    res.sendFile(path.join(frontendRoot, 'public', 'index.html'));
});

app.get('/upload', (req, res) => {
    res.sendFile(path.join(frontendRoot, 'public', 'upload.html'));
});

/**
 * 数据库查询 API 端点
 * 接收用户的自然语言查询，转换为 SQL 执行，然后将结果交给 AI 生成自然语言回答
 */
/**
 * 意图分类 API 端点
 * 使用训练好的 BERT 模型判断用户查询意图
 */
app.post('/api/classify-intent', async (req, res) => {
    try {
        const { query } = req.body;
        
        if (!query || !String(query).trim()) {
            return res.status(400).json({ success: false, error: 'Query is required' });
        }

        // 调用 Python 意图分类脚本
        const aiResponse = await runApiHandler({
            type: 'classify_intent',
            query: query,
        });

        let parsed;
        try {
            parsed = JSON.parse(aiResponse);
        } catch (_err) {
            return res.json({ success: true, intent: 'rag_query' }); // 默认回退
        }

        return res.json(parsed);

    } catch (error) {
        console.error('Intent classification error:', error);
        // 回退到 rag_query
        return res.json({ success: true, intent: 'rag_query' });
    }
});

app.post('/api/database-query', async (req, res) => {
    try {
        // 从请求体中获取查询文本
        const { query } = req.body;
        
        // 验证查询参数
        if (!query || !String(query).trim()) {
            return res.status(400).json({ success: false, error: 'Query is required' });
        }

        // ========== 步骤1：使用 LangChain 将自然语言转换为 SQL ==========
        let sqlQuery = '';
        let dbResults = [];
        
        try {
            // 调用 Python SQL 生成器
            const aiResponse = await runApiHandler({
                type: 'generate_sql',
                query: query,
            });

            let parsed;
            try {
                parsed = JSON.parse(aiResponse);
            } catch (_err) {
                console.log('[DB Query] SQL generation failed, using fallback');
                parsed = { success: false, error: 'SQL generation failed' };
            }

            if (parsed.success && parsed.sql) {
                sqlQuery = parsed.sql;
                dbResults = parsed.result || [];
                console.log(`[DB Query] Generated SQL: ${sqlQuery}`);
            } else {
                throw new Error(parsed.error || 'SQL generation failed');
            }
        } catch (error) {
            console.log(`[DB Query] LangChain SQL generation failed: ${error}, using keyword fallback`);
            
            // 回退到关键词匹配
            const lowerQuery = query.toLowerCase();
            
            // 员工相关查询
            if (lowerQuery.includes('员工') || lowerQuery.includes('employee')) {
                if (lowerQuery.includes('数量') || lowerQuery.includes('count') || lowerQuery.includes('多少')) {
                    sqlQuery = 'SELECT COUNT(*) as employee_count FROM employees';
                } else if (lowerQuery.includes('部门') || lowerQuery.includes('department')) {
                    sqlQuery = 'SELECT e.id, e.name, e.email, d.name as department FROM employees e JOIN departments d ON e.department_id = d.id';
                } else {
                    sqlQuery = 'SELECT * FROM employees';
                }
            }
            // 部门相关查询
            else if (lowerQuery.includes('部门') || lowerQuery.includes('department')) {
                if (lowerQuery.includes('邮箱') || lowerQuery.includes('email')) {
                    sqlQuery = 'SELECT name, email FROM departments';
                } else {
                    sqlQuery = 'SELECT * FROM departments';
                }
            }
            // 默认查询
            else {
                sqlQuery = 'SELECT "Query not understood" as message';
            }
            
            console.log(`[DB Query] Fallback SQL: ${sqlQuery}`);
            
            // 执行 SQL 查询
            const connection = await mysql.createConnection({
                host: process.env.DB_HOST,
                port: process.env.DB_PORT,
                user: process.env.DB_USERNAME,
                password: process.env.DB_PASSWORD,
                database: process.env.DB_DATABASE,
            });
            
            const [rows] = await connection.execute(sqlQuery);
            await connection.end();
            
            // 转换为字典列表
            dbResults = rows.map(row => {
                const formatted = {};
                Object.entries(row).forEach(([key, value]) => {
                    if (value !== null) {
                        formatted[key] = value;
                    }
                });
                return formatted;
            });
        }

        // 如果查询结果为空，直接返回
        if (!dbResults || dbResults.length === 0) {
            return res.json({ 
                success: true, 
                answer: '未找到相关数据' 
            });
        }

        // 将查询和结果发送给 Python 后端，让 AI 生成自然语言回答
        const aiResponse = await runApiHandler({
            type: 'database_query',
            query: query,
            db_results: JSON.stringify(dbResults),
            model: 'blissful_ishizaka_626/gemma4-cloud',
        });

        // 解析 AI 的响应
        let parsed;
        try {
            parsed = JSON.parse(aiResponse);
        } catch (_err) {
            // 如果 AI 返回的不是 JSON 格式，直接返回原始响应
            return res.json({ success: true, answer: aiResponse });
        }

        // 如果 AI 处理失败，返回原始查询结果作为回退
        if (!parsed.success) {
            let fallbackAnswer = '查询结果：\n\n';
            dbResults.forEach((row, index) => {
                fallbackAnswer += `${index + 1}. `;
                Object.entries(row).forEach(([key, value]) => {
                    fallbackAnswer += `${key}: ${value} | `;
                });
                fallbackAnswer = fallbackAnswer.slice(0, -2) + '\n';
            });
            return res.json({ success: true, answer: fallbackAnswer });
        }

        // 返回 AI 生成的自然语言回答
        return res.json(parsed);

    } catch (error) {
        // 错误处理：记录错误并返回错误信息
        console.error('Database query error:', error);
        return res.status(500).json({ success: false, error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`LLM Backend running on http://localhost:${PORT}`);
    console.log('API endpoints:');
    console.log('  POST /api/query - Main LLM query');
    console.log('  POST /api/rag - RAG-based query');
    console.log('  POST /api/deep-think - Placeholder deep think mode');
    console.log('  POST /api/upload-text - Upload text to vector store');
});
