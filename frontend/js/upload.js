const API_BASE_URL = 'http://localhost:3000/api';

const sourceInput = document.getElementById('source-name');
const chunkSizeInput = document.getElementById('chunk-size');
const chunkOverlapInput = document.getElementById('chunk-overlap');
const splitMethodInput = document.getElementById('split-method');
const textInput = document.getElementById('upload-text');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const resultBox = document.getElementById('result-box');

uploadBtn.addEventListener('click', uploadTextToVectorStore);

function setStatus(message, type = '') {
    uploadStatus.textContent = message;
    uploadStatus.className = `status ${type}`;
}

function renderResult(content) {
    resultBox.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'message message-assistant';

    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = '系统';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;

    wrapper.appendChild(label);
    wrapper.appendChild(bubble);
    resultBox.appendChild(wrapper);
}

async function uploadTextToVectorStore() {
    const text = textInput.value.trim();
    const source = sourceInput.value.trim() || 'manual_upload';
    const chunkSize = Number(chunkSizeInput.value) || 500;
    const chunkOverlap = Number(chunkOverlapInput.value) || 50;
    const splitMethod = splitMethodInput.value || 'character';

    if (!text) {
        setStatus('请先输入文本内容', 'error');
        return;
    }

    setStatus('上传中...', 'loading');
    uploadBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/upload-text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                source,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap,
                split_method: splitMethod,
            }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || 'Upload failed');
        }

        const fullDocIds = Array.isArray(payload.full_doc_ids) ? payload.full_doc_ids : [];
        const summaryIds = Array.isArray(payload.summary_ids) ? payload.summary_ids : [];

        const msg = [
            '上传成功',
            `source: ${payload.source}`,
            `分块方式: ${payload.split_method === 'markdown_headers' ? '按Header切割' : '普通切割'}`,
            `chunks: ${payload.chunks_uploaded}`,
            `full_doc_ids: ${fullDocIds.join(', ') || 'N/A'}`,
            `summary_ids: ${summaryIds.join(', ') || 'N/A'}`,
        ].join('\n');

        setStatus('上传完成', 'success');
        renderResult(msg);
    } catch (error) {
        setStatus('上传失败', 'error');
        renderResult(`错误: ${error.message}`);
    } finally {
        uploadBtn.disabled = false;
    }
}
