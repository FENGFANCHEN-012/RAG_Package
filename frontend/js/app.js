/**
 * ============================================================================
 * 前端主应用文件 (app.js)
 * ============================================================================
 * 
 * 功能描述：
 * - 管理聊天界面的用户交互
 * - 处理消息发送和显示
 * - 根据查询类型（数据库查询、RAG、普通查询）路由到不同的后端 API
 * - 使用关键词匹配检测数据库查询意图
 * - 管理对话历史和加载状态
 * 
 * 主要流程：
 * 1. 用户输入消息
 * 2. 检测是否为数据库查询（关键词匹配）
 * 3. 根据检测结果路由到相应的 API 端点
 * 4. 显示后端返回的答案
 * 
 * 依赖：
 * - 后端 API: http://localhost:3000/api
 * - DOM 元素: chat-container, user-input, send-btn, status, query-type
 * 
 * ============================================================================
 */

const API_BASE_URL = 'http://localhost:3000/api';

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
const queryTypeSelect = document.getElementById('query-type');

// State
let isLoading = false;
let conversationHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    setupEventListeners();
});

function setupEventListeners() {
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
    });
}

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            setStatus('✅ 服务器已连接', 'success');
        } else {
            setStatus('⚠️ 服务器响应异常', 'error');
        }
    } catch (error) {
        setStatus('❌ 无法连接到服务器，请确保后端已启动', 'error');
    }
}

function setStatus(message, type = '') {
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

function setQuickQuestion(question) {
    userInput.value = question;
    userInput.focus();
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
}

/**
 * 检测用户输入是否为数据库查询指令
 * 使用关键词匹配判断用户是否想要查询数据库
 * @param {string} message - 用户输入的消息
 * @returns {boolean} - 如果包含数据库相关关键词返回 true
 */


function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * 发送用户消息到后端处理
 * 根据消息类型（数据库查询或普通查询）路由到不同的处理函数
 */
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isLoading) return;

    // 清除欢迎消息（首次发送时）
    const welcomeMsg = chatContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // 添加用户消息到聊天界面
    addMessage(message, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';

    setLoading(true);
    setStatus('⏳ 正在处理...', 'loading');

    try {
        let response;
        
        // 检测是否为数据库查询，如果是则路由到数据库查询 API
       
        const queryType = queryTypeSelect.value;

        if (queryType === 'rag') {
                // RAG 模式：基于文档检索的问答
                response = await handleRAGQuery(message);
            } else if (queryType === 'deepthink') {
                // 深度思考模式（暂未实现）
                response = await handleDeepThink(message);
            } else {
                // 普通查询模式
                response = await handleNormalQuery(message);
            }
        

        // 检查响应是否成功
        if (!response.success) {
            throw new Error(response.error || 'Unknown error');
        }

        // 显示助手回答
        const answerText = (response.answer || '').trim();
        addMessage(answerText || '模型返回了空结果，请稍后重试。', 'assistant');
        setStatus('✅ 完成', 'success');
    } catch (error) {
        console.error('Error:', error);
        addMessage(`❌ 错误: ${error.message}`, 'assistant');
        setStatus('❌ 请求失败', 'error');
    } finally {
        setLoading(false);
    }
}

async function handleNormalQuery(question) {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Query request failed');
    }
    return await response.json();
}

async function handleDeepThink(question) {
    const response = await fetch(`${API_BASE_URL}/deep-think`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question }),
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Deep think request failed');
    }
    return await response.json();
}

// User should not provide docs; backend always retrieves from the local vector store.
async function handleRAGQuery(question) {
    const response = await fetch(`${API_BASE_URL}/rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            question,
            conversation_history: conversationHistory 
        }),
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'RAG request failed');
    }
    return await response.json();
}



function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;

    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = role === 'user' ? '👤 您' : '🤖 助手';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;

    messageDiv.appendChild(label);
    messageDiv.appendChild(bubble);

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    conversationHistory.push({ role, content, timestamp: new Date() });
}

function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    userInput.disabled = loading;

    if (loading) {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message message-assistant loading-message';
        loadingDiv.innerHTML = `
            <div class="message-label">🤖 助手</div>
            <div class="bubble">
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatContainer.appendChild(loadingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } else {
        const loadingMsg = chatContainer.querySelector('.loading-message');
        if (loadingMsg) loadingMsg.remove();
    }
}

// Export for debugging
window.LLMApp = {
    conversationHistory,
    clearHistory: () => {
        conversationHistory = [];
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <h2>👋 欢迎使用 LLM Assistant</h2>
                <p>请输入您的问题，我会尽力为您解答。</p>
                <div class="quick-actions">
                    <button class="quick-btn" onclick="setQuickQuestion('阿司匹林的主要作用是什么？')">
                        💊 阿司匹林的作用
                    </button>
                    <button class="quick-btn" onclick="setQuickQuestion('什么是机器学习？')">
                        🤖 机器学习介绍
                    </button>
                    <button class="quick-btn" onclick="setQuickQuestion('如何预防感冒？')">
                        🤧 感冒预防
                    </button>
                </div>
            </div>
        `;
    },
};
