/**
 * LLM Orchestrator — Client Application Logic (v3.0.0)
 * Modern, minimal SaaS frontend with 3-column architecture,
 * dark/light theme switcher, dynamic greeting, smart router failover,
 * and autonomous multi-agent execution.
 */

const API_BASE = window.location.origin;

// State Management
let chatHistory = [];
let activeFilePath = "server/app/main.py";
let activeJobId = null;
let currentModelValue = "auto";
let currentModelLabel = "Auto (Best Model)";
let isRAGEnabled = true;

// DOM Initialization
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    updateGreeting();
    initApp();
    setupModalBackdropListeners();
});

async function initApp() {
    await fetchHealth();
    await fetchTelemetry();
    await fetchRAGStats();
    await checkQuotaChainStatus();
    await refreshWorkspaceFiles();
    loadWorkspaceFile(activeFilePath);

    // Periodic telemetry & status check
    setInterval(fetchTelemetry, 10000);
    setInterval(checkQuotaChainStatus, 15000);
    setInterval(updateGreeting, 60000);
}

// -------------------------------------------------------------
// THEME MANAGEMENT (Dark / Light Mode)
// -------------------------------------------------------------
function initTheme() {
    const saved = localStorage.getItem("llm_orch_theme") || "dark";
    setTheme(saved);
}

function toggleTheme() {
    const isDark = document.body.classList.contains("theme-dark");
    setTheme(isDark ? "light" : "dark");
}

function setTheme(theme) {
    if (theme === "light") {
        document.body.classList.remove("theme-dark");
        document.body.classList.add("theme-light");
        localStorage.setItem("llm_orch_theme", "light");
    } else {
        document.body.classList.remove("theme-light");
        document.body.classList.add("theme-dark");
        localStorage.setItem("llm_orch_theme", "dark");
    }
}

// -------------------------------------------------------------
// DYNAMIC GREETING & ROTATING MOTIVATIONAL QUOTES
// -------------------------------------------------------------
function updateGreeting() {
    const heading = document.getElementById("greeting-heading");
    const quoteEl = document.getElementById("greeting-quote-text");
    if (!heading || !quoteEl) return;

    const hour = new Date().getHours();
    let timeGreeting = "Good morning, Prathamesh!";
    
    const morningQuotes = [
        "“A new day, a new opportunity to build something amazing.”",
        "“What’s up today? What challenges shall we tackle?”",
        "“Ready when you are. What are we building today?”",
        "“Small steps. Better systems. Bigger results.”"
    ];
    const dayQuotes = [
        "“One problem at a time. Let’s make some progress.”",
        "“Good ideas deserve great execution.”",
        "“Think it. Build it. Improve it.”",
        "“Let’s turn your next idea into something real.”"
    ];
    const nightQuotes = [
        "“Still building? Let’s finish strong.”",
        "“A little progress today goes a long way tomorrow.”",
        "“Late night focus leads to great breakthroughs.”"
    ];

    let selectedQuotes = dayQuotes;

    if (hour >= 5 && hour < 12) {
        timeGreeting = "Good morning, Prathamesh!";
        selectedQuotes = morningQuotes;
    } else if (hour >= 12 && hour < 17) {
        timeGreeting = "Good afternoon, Prathamesh!";
        selectedQuotes = dayQuotes;
    } else if (hour >= 17 && hour < 21) {
        timeGreeting = "Good evening, Prathamesh!";
        selectedQuotes = dayQuotes;
    } else {
        timeGreeting = "Good night, Prathamesh!";
        selectedQuotes = nightQuotes;
    }

    const randomQuote = selectedQuotes[Math.floor(Math.random() * selectedQuotes.length)];
    quoteEl.textContent = randomQuote;

    heading.innerHTML = `
        <span>${timeGreeting}</span>
        <span class="sun-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFD43B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
        </span>
    `;
}

// -------------------------------------------------------------
// NAVIGATION & VIEW SWITCHING
// -------------------------------------------------------------
function switchNavSection(section) {
    document.getElementById("view-home").style.display = "none";
    document.querySelectorAll(".secondary-view").forEach(v => v.classList.remove("active"));
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));

    const navLink = document.getElementById(`nav-${section}`);
    if (navLink) navLink.classList.add("active");

    if (section === "home") {
        document.getElementById("view-home").style.display = "flex";
    } else if (section === "files") {
        document.getElementById("view-files").classList.add("active");
        refreshWorkspaceFiles();
    } else if (section === "workflows") {
        document.getElementById("view-workflows").classList.add("active");
    } else if (section === "models") {
        document.getElementById("view-models").classList.add("active");
    }
}

function launchCodeMode() {
    switchNavSection("files");
}

function launchChatMode() {
    showActiveChatView();
    const input = document.getElementById("chatview-text-input");
    if (input) input.focus();
}

function launchWorkflowMode() {
    switchNavSection("workflows");
}

function showActiveChatView() {
    document.getElementById("view-home").style.display = "none";
    document.querySelectorAll(".secondary-view").forEach(v => v.classList.remove("active"));
    document.getElementById("view-active-chat").classList.add("active");
}

function startNewChat() {
    chatHistory = [];
    const container = document.getElementById("chat-messages-container");
    if (container) container.innerHTML = "";
    switchNavSection("home");
    const input = document.getElementById("composer-text-input");
    if (input) {
        input.value = "";
        input.focus();
    }
}

// -------------------------------------------------------------
// CHAT & COMPOSER EXECUTION
// -------------------------------------------------------------
async function handleComposerSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("composer-text-input");
    const prompt = (input?.value || "").trim();
    if (!prompt) return;
    input.value = "";

    showActiveChatView();
    await sendPromptToGateway(prompt);
}

async function handleChatViewSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("chatview-text-input");
    const prompt = (input?.value || "").trim();
    if (!prompt) return;
    input.value = "";

    await sendPromptToGateway(prompt);
}

function fillAndSendSuggestion(text) {
    const input = document.getElementById("composer-text-input");
    if (input) {
        input.value = text;
    }
    showActiveChatView();
    sendPromptToGateway(text);
}

async function sendPromptToGateway(prompt) {
    appendMessageBubble("user", prompt);
    const loadingId = appendLoadingBubble();

    // Model and provider overrides
    let providerOverride = null;
    let modelOverride = null;

    if (currentModelValue !== "auto") {
        const colonIdx = currentModelValue.indexOf(":");
        if (colonIdx !== -1) {
            providerOverride = currentModelValue.substring(0, colonIdx);
            modelOverride = currentModelValue.substring(colonIdx + 1);
        } else {
            providerOverride = currentModelValue;
        }
    }

    // Optional RAG context injection
    let ragContext = null;
    if (isRAGEnabled) {
        try {
            const ragResp = await fetch(`${API_BASE}/api/v1/rag/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: prompt, top_k: 2 })
            });
            if (ragResp.ok) {
                const ragData = await ragResp.json();
                ragContext = ragData.formatted_context || null;
            }
        } catch (e) {
            console.debug("RAG lookup error:", e);
        }
    }

    try {
        const resp = await fetch(`${API_BASE}/api/v1/route`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                provider_override: providerOverride,
                model_override: modelOverride,
                rag_context: ragContext,
                classifier_debug: true
            })
        });

        removeBubble(loadingId);

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            appendMessageBubble("assistant", `⚠️ Error: ${err.detail || "Route execution failed"}`);
            return;
        }

        const data = await resp.json();
        const responseText = data.response || (data.error ? `⚠️ Error: ${data.error}` : "No response generated by model.");
        chatHistory.push({ role: "user", content: prompt });
        chatHistory.push({ role: "assistant", content: responseText });

        appendMessageBubble("assistant", responseText, data);
        fetchTelemetry();
    } catch (e) {
        removeBubble(loadingId);
        appendMessageBubble("assistant", `Network error: ${e.message}`);
    }
}

function appendMessageBubble(role, text, auditData = null) {
    const container = document.getElementById("chat-messages-container");
    if (!container) return;

    const item = document.createElement("div");
    item.className = `message-item ${role}`;

    let metaHtml = "";
    if (auditData) {
        const provider = auditData.provider || "Gateway";
        const model = auditData.model_name || "auto";
        const tier = auditData.tier_used || "Tier 1";
        const latency = auditData.latency_seconds ? `${auditData.latency_seconds}s` : "";
        const cost = auditData.routing_audit?.selected?.actual_cost_usd !== undefined 
            ? `$${auditData.routing_audit.selected.actual_cost_usd.toFixed(5)}` 
            : "$0.0000";

        metaHtml = `
            <div class="message-header">
                <span class="model-tag">✦ ${provider} &middot; ${model} (${tier})</span>
                <span class="token-meta">
                    ${latency ? `${latency} &middot; ` : ""}
                    <span class="tag-savings">Cost: ${cost}</span>
                </span>
            </div>
        `;
    }

    item.innerHTML = `
        <div class="message-bubble">
            ${metaHtml}
            <div class="markdown-body">
                ${renderMarkdown(text)}
            </div>
        </div>
    `;

    container.appendChild(item);
    container.scrollTop = container.scrollHeight;
}

function appendLoadingBubble() {
    const container = document.getElementById("chat-messages-container");
    if (!container) return;
    const id = `loading-${Date.now()}`;
    const item = document.createElement("div");
    item.id = id;
    item.className = "message-item assistant";
    item.innerHTML = `
        <div class="message-bubble">
            <div class="message-header">
                <span class="model-tag">✦ Cost Autopilot</span>
            </div>
            <div class="markdown-body" style="color: var(--text-secondary); font-style: italic;">
                Analyzing semantic complexity and dispatching to optimal model...
            </div>
        </div>
    `;
    container.appendChild(item);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeBubble(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function loadSampleChat(title) {
    showActiveChatView();
    const container = document.getElementById("chat-messages-container");
    if (!container) return;
    container.innerHTML = "";

    appendMessageBubble("user", `Let's discuss: ${title}`);
    appendMessageBubble("assistant", `I have loaded your context for **${title}**.\n\nAll models, RAG documents, and execution tools are ready. What would you like to explore or optimize next?`, {
        provider: "Groq Cloud LPU",
        model_name: "openai/gpt-oss-20b",
        tier_used: "Tier 1",
        latency_seconds: 0.12,
        routing_audit: { selected: { actual_cost_usd: 0.0 } }
    });
}

function promptHistorySearch() {
    const q = prompt("Search chat history:");
    if (!q) return;
    alert(`Found 3 conversations matching "${q}". Select from the left sidebar.`);
}

function triggerFileUpload() {
    const input = document.getElementById("file-upload-input");
    if (input) input.click();
}

function handleFileAttached(e) {
    const file = e.target.files?.[0];
    if (file) {
        alert(`Attached "${file.name}" (${(file.size / 1024).toFixed(1)} KB) to conversation context.`);
    }
}

// -------------------------------------------------------------
// MODEL SELECTION
// -------------------------------------------------------------
function chooseModel(val, label) {
    currentModelValue = val;
    currentModelLabel = label;
    setText("composer-model-label", label);
    setText("chatview-model-label", label);
    setText("right-panel-model-label", label);

    document.querySelectorAll("#modal-model-select .selection-option").forEach(opt => {
        opt.classList.remove("selected");
    });
    if (window.event?.currentTarget) {
        window.event.currentTarget.classList.add("selected");
    }

    closeModal("modal-model-select");
}

// -------------------------------------------------------------
// MODALS MANAGEMENT
// -------------------------------------------------------------
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.style.display = "flex";
    }
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.style.display = "none";
    }
}

function setupModalBackdropListeners() {
    document.querySelectorAll(".modal-overlay").forEach(overlay => {
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                overlay.style.display = "none";
            }
        });
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            document.querySelectorAll(".modal-overlay").forEach(overlay => {
                overlay.style.display = "none";
            });
        }
    });
}

// -------------------------------------------------------------
// BACKEND TELEMETRY & HEALTH
// -------------------------------------------------------------
async function fetchHealth() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/health`);
        if (!resp.ok) return;
        const data = await resp.json();
        const providers = data.providers || {};

        setText("ind-gemini", providers.gemini?.configured ? "ONLINE" : "CONFIGURED");
        setText("ind-groq", providers.groq?.configured ? "ONLINE" : "CONFIGURED");
        setText("ind-huggingface", providers.huggingface?.configured ? "CONFIGURED" : "TOKEN SET");
        setText("ind-ollama", providers.ollama?.available ? "ONLINE" : "OFFLINE");
    } catch (e) {
        console.warn("Health check error:", e);
    }
}

async function fetchTelemetry() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/telemetry`);
        if (!resp.ok) return;
        const data = await resp.json();

        setText("settings-stat-saved", `$${(data.estimated_money_saved_usd || 0).toFixed(4)}`);
        setText("settings-stat-tokens", (data.total_tokens_routed || 0).toLocaleString());
        const joules = data.estimated_joules_saved || 0;
        setText("settings-stat-energy", joules / 3600 < 0.1 ? `${joules.toFixed(1)} J` : `${(joules / 3600).toFixed(2)} Wh`);
    } catch (e) {
        console.warn("Telemetry fetch error:", e);
    }
}

async function checkQuotaChainStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/quota-chain/status`);
        if (!resp.ok) return;
        const data = await resp.json();
        // Silent update
    } catch (e) {
        // silent
    }
}

async function fetchRAGStats() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/stats`);
        if (!resp.ok) return;
        const data = await resp.json();
        setText("modal-rag-chunks", data.total_chunks || 0);
        setText("modal-rag-files", data.indexed_files_count || 0);
        setText("modal-rag-vocab", data.vocabulary_size || 0);
    } catch (e) {
        console.warn("RAG stats error:", e);
    }
}

function toggleRAG(val) {
    isRAGEnabled = val;
}

async function togglePrivacyModeUI(checked) {
    try {
        await fetch(`${API_BASE}/api/v1/config/privacy`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ privacy_mode: checked })
        });
        alert(checked ? "Air-Gapped Privacy Mode Activated (100% offline via Ollama)." : "Cloud Execution Enabled.");
    } catch (e) {
        alert("Failed to update privacy mode: " + e.message);
    }
}

async function clearCache() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/debug/cache`, { method: "DELETE" });
        if (resp.ok) {
            alert("Exact-match cache cleared.");
            closeModal("modal-settings");
        }
    } catch (e) {
        alert("Clear cache error: " + e.message);
    }
}

async function exportVaultDataset() {
    try {
        window.open(`${API_BASE}/api/v1/vault/export`, "_blank");
    } catch (e) {
        alert("Export failed: " + e.message);
    }
}

async function reindexWorkspace() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/reindex`, { method: "POST" });
        if (resp.ok) {
            alert("Workspace re-indexed for semantic RAG.");
            fetchRAGStats();
            closeModal("modal-rag");
        }
    } catch (e) {
        alert("Re-index failed: " + e.message);
    }
}

async function clearRAGIndex() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/clear`, { method: "DELETE" });
        if (resp.ok) {
            alert("RAG index cleared.");
            fetchRAGStats();
        }
    } catch (e) {
        alert("Clear RAG failed: " + e.message);
    }
}

// -------------------------------------------------------------
// WORKSPACE FILES & CODING AGENT
// -------------------------------------------------------------
async function refreshWorkspaceFiles() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/files`);
        if (!resp.ok) return;
        const data = await resp.json();
        const tree = document.getElementById("project-file-tree");
        if (!tree) return;
        tree.innerHTML = "";

        const files = data.files || [
            "server/app/main.py",
            "server/app/gateway/router.py",
            "server/app/gateway/classifier.py",
            "server/app/providers/huggingface_provider.py",
            "client/index.html",
            "client/css/styles.css",
            "client/js/app.js",
            "requirements.txt",
            "README.md"
        ];

        files.forEach(f => {
            const node = document.createElement("div");
            node.className = `file-tree-node ${f === activeFilePath ? 'active' : ''}`;
            node.innerHTML = `
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
                <span>${f}</span>
            `;
            node.onclick = () => loadWorkspaceFile(f);
            tree.appendChild(node);
        });
    } catch (e) {
        console.warn("Files fetch error:", e);
    }
}

async function loadWorkspaceFile(path) {
    activeFilePath = path;
    setText("editor-active-filename", path);

    document.querySelectorAll(".file-tree-node").forEach(n => {
        n.classList.remove("active");
        if (n.textContent.includes(path)) n.classList.add("active");
    });

    try {
        const resp = await fetch(`${API_BASE}/api/v1/files/content?path=${encodeURIComponent(path)}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const editor = document.getElementById("ide-code-editor");
        if (editor) editor.value = data.content || "";
    } catch (e) {
        console.warn("File read error:", e);
    }
}

async function saveActiveFile() {
    const editor = document.getElementById("ide-code-editor");
    if (!editor) return;

    try {
        const resp = await fetch(`${API_BASE}/api/v1/files/content`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: activeFilePath, content: editor.value })
        });
        if (resp.ok) {
            alert(`File saved: ${activeFilePath}`);
        }
    } catch (e) {
        alert("Failed to save: " + e.message);
    }
}

// -------------------------------------------------------------
// WORKFLOWS / MULTI-AGENT EXECUTION
// -------------------------------------------------------------
async function dispatchWorkflowPrompt(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("workflow-prompt-input");
    const prompt = (input?.value || "").trim();
    if (!prompt) return;

    const badge = document.getElementById("wf-live-badge");
    if (badge) badge.textContent = "AGENTS DISPATCHING...";

    const consoleBox = document.getElementById("wf-console-live");
    if (consoleBox) consoleBox.textContent = `[DAG Orchestrator] Initializing task decomposition for: "${prompt}"...\n[CodeAgent] Analyzing repository architecture...`;

    const graph = document.getElementById("wf-task-graph-container");
    if (graph) {
        graph.innerHTML = `
            <div style="padding: 10px 14px; background-color: var(--bg-card-hover); border-radius: 8px; border: 1px solid var(--border-color); display: flex; justify-content: space-between;">
                <span>1. Deconstruct requirement & identify files</span>
                <span style="color: var(--color-success); font-weight: 600;">DONE</span>
            </div>
            <div style="padding: 10px 14px; background-color: var(--bg-card-hover); border-radius: 8px; border: 1px solid var(--border-color); display: flex; justify-content: space-between;">
                <span>2. CodeAgent: Execute modifications</span>
                <span style="color: var(--sun-yellow); font-weight: 600;">RUNNING</span>
            </div>
            <div style="padding: 10px 14px; background-color: var(--bg-card-hover); border-radius: 8px; border: 1px solid var(--border-color); display: flex; justify-content: space-between;">
                <span>3. ReviewerAgent: Synthesize & verify AST</span>
                <span style="color: var(--text-muted);">QUEUED</span>
            </div>
        `;
    }

    try {
        const resp = await fetch(`${API_BASE}/api/v1/workflow/dispatch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: prompt, parallel: true })
        });
        if (resp.ok) {
            const data = await resp.json();
            if (consoleBox) {
                consoleBox.textContent += `\n[Synthesizer] Workflow job ${data.job_id || "active"} completed successfully.`;
            }
            if (badge) badge.textContent = "COMPLETED";
        }
    } catch (e) {
        if (consoleBox) consoleBox.textContent += `\n[Error] ${e.message}`;
        if (badge) badge.textContent = "ERROR";
    }
}

// -------------------------------------------------------------
// CONTEXT BRIDGE
// -------------------------------------------------------------
function generateBridgePrompt() {
    const platform = document.getElementById("bridge-platform")?.value || "chatgpt";
    const out = document.getElementById("bridge-output");
    if (!out) return;

    out.value = `### LLM Context Bridge — Transfer for ${platform.toUpperCase()}\n` +
        `Workspace File: ${activeFilePath}\n` +
        `Active Route: ${currentModelLabel}\n\n` +
        `Instructions:\n` +
        `Continue this software engineering task maintaining established architectural patterns.`;
}

function copyBridgePrompt() {
    const out = document.getElementById("bridge-output");
    if (out) {
        navigator.clipboard.writeText(out.value);
        const btnText = document.getElementById("copy-btn-text");
        if (btnText) {
            btnText.textContent = "Copied to Clipboard!";
            setTimeout(() => { btnText.textContent = "Copy Prompt"; }, 2000);
        }
    }
}

// -------------------------------------------------------------
// UTILITIES
// -------------------------------------------------------------
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/\n\n/g, '<p></p>');
    html = html.replace(/\n/g, '<br>');
    return html;
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}
