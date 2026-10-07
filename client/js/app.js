/**
 * Agent Orchestrator & LLM Cost Autopilot - Client Application Logic
 * Supports: LLM Chat, Coding Agent IDE, Local RAG, Context Bridge, SSE Progress, and Telemetry.
 */

const API_BASE = window.location.origin;

// State
let chatHistory = [];
let activeFilePath = "server/app/main.py";
let activeJobId = null;
let eventSource = null;
let currentWorkflowState = null;

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    await fetchHealth();
    await fetchTelemetry();
    await fetchRAGStats();
    await refreshWorkspaceFiles();
    loadWorkspaceFile(activeFilePath);

    // Periodic telemetry update
    setInterval(fetchTelemetry, 8000);
}

// -------------------------------------------------------------
// Section Navigation
// -------------------------------------------------------------
function switchMainSection(section) {
    document.querySelectorAll(".workspace-section").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-tab").forEach(el => el.classList.remove("active"));

    if (section === "chat") {
        document.getElementById("section-chat").classList.add("active");
        document.getElementById("tab-llm-chat").classList.add("active");
    } else if (section === "coding") {
        document.getElementById("section-coding").classList.add("active");
        document.getElementById("tab-coding-agent").classList.add("active");
    } else if (section === "analytics") {
        document.getElementById("section-analytics").classList.add("active");
        document.getElementById("tab-analytics").classList.add("active");
        loadRouterAnalytics();
    }
}

// -------------------------------------------------------------
// Health & Provider Badges
// -------------------------------------------------------------
async function fetchHealth() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/health`);
        if (!resp.ok) return;
        const data = await resp.json();

        // Update provider chips
        const providers = data.providers || {};
        updateChip("chip-gemini", providers.gemini?.configured);
        updateChip("chip-groq", providers.groq?.configured);
        updateChip("chip-huggingface", providers.huggingface?.configured);
        updateChip("chip-openrouter", providers.openrouter?.configured);
        updateChip("chip-openai", providers.openai?.configured);
        updateChip("chip-ollama", providers.ollama?.online);

        // Update Privacy Mode indicator
        const privIcon = document.getElementById("privacy-icon");
        const privLabel = document.getElementById("privacy-label");
        if (data.privacy_mode) {
            privIcon.textContent = "🔒";
            privLabel.textContent = "Offline";
        } else {
            privIcon.textContent = "🌐";
            privLabel.textContent = "Hybrid";
        }

        // ML Status
        const mlStatus = document.getElementById("ml-status-text");
        if (mlStatus) {
            mlStatus.textContent = data.ml_classifier_active ? "XGBoost/GBDT Active" : "Intent Heuristics Active";
        }

        // RAG Chunks count
        const ragCount = document.getElementById("rag-chunks-count");
        if (ragCount) {
            ragCount.textContent = `${data.rag_chunks_indexed || 0} chunks`;
        }
    } catch (e) {
        console.warn("Health fetch failed:", e);
    }
}

function updateChip(elementId, isOnline) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.classList.remove("online", "offline");
    el.classList.add(isOnline ? "online" : "offline");
}

async function fetchTelemetry() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/telemetry`);
        if (!resp.ok) return;
        const data = await resp.json();

        const savedEl = document.getElementById("stat-saved");
        const tokensEl = document.getElementById("stat-tokens");
        const energyEl = document.getElementById("stat-energy");

        if (savedEl) savedEl.textContent = `$${(data.estimated_money_saved_usd || 0).toFixed(4)}`;
        if (tokensEl) tokensEl.textContent = (data.total_tokens_routed || 0).toLocaleString();
        if (energyEl) energyEl.textContent = `${(data.estimated_joules_saved || 0) / 3600 < 0.1 ? (data.estimated_joules_saved || 0).toFixed(1) + " J" : ((data.estimated_joules_saved || 0) / 3600).toFixed(2) + " Wh"}`;

        renderParetoChart();
    } catch (e) {
        console.warn("Telemetry fetch error:", e);
    }
}

async function togglePrivacyMode() {
    try {
        const healthResp = await fetch(`${API_BASE}/api/v1/health`);
        const healthData = await healthResp.json();
        const newMode = !healthData.privacy_mode;

        await fetch(`${API_BASE}/api/v1/config/privacy`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ privacy_mode: newMode })
        });

        await fetchHealth();
    } catch (e) {
        alert("Failed to toggle privacy mode: " + e.message);
    }
}

// -------------------------------------------------------------
// SECTION 1: LLM CHAT LOGIC
// -------------------------------------------------------------
function startNewChat() {
    chatHistory = [];
    const container = document.getElementById("chat-messages");
    container.innerHTML = `
        <div class="chat-welcome">
            <h2>LLM Chat & Intelligent Cost Autopilot</h2>
            <p>Every query is analyzed for semantic complexity and routed to the most cost-effective model tier.</p>
            <div class="prompt-suggestions">
                <button class="suggestion-chip" onclick="setChatInput('Write a Python function to parse and validate semantic version strings')">Python SemVer Validator</button>
                <button class="suggestion-chip" onclick="setChatInput('Correct the grammar in the sentence: She do not likes to code in Python.')">Grammar Proofreading (Tier 1)</button>
                <button class="suggestion-chip" onclick="setChatInput('Explain how distributed consensus works in Raft and compare with Paxos')">Distributed Consensus (Tier 3)</button>
            </div>
        </div>
    `;
}

function setChatInput(text) {
    const input = document.getElementById("chat-input");
    input.value = text;
    input.focus();
}

function handleChatKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleChatSubmit(event);
    }
}

async function handleChatSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("chat-input");
    const prompt = input.value.trim();
    if (!prompt) return;

    input.value = "";
    appendChatMessage("user", prompt);

    // Selected model override
    const modelSelect = document.getElementById("chat-model-select").value;
    let providerOverride = null;
    let modelOverride = null;

    if (modelSelect !== "auto") {
        const parts = modelSelect.split(":");
        providerOverride = parts[0];
        modelOverride = parts[1];
    }

    // Optional RAG Context
    const ragToggle = document.getElementById("chat-rag-toggle");
    let ragContext = null;
    if (ragToggle && ragToggle.checked) {
        try {
            const ragResp = await fetch(`${API_BASE}/api/v1/rag/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: prompt, top_k: 3 })
            });
            if (ragResp.ok) {
                const ragData = await ragResp.json();
                ragContext = ragData.formatted_context || null;
            }
        } catch (e) {
            console.debug("RAG search error:", e);
        }
    }

    // Loading indicator
    const loadingId = appendChatLoading();

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

        removeChatLoading(loadingId);

        if (!resp.ok) {
            const err = await resp.json();
            appendChatMessage("assistant", `Error: ${err.detail || "Route execution failed"}`);
            return;
        }

        const data = await resp.json();
        chatHistory.push({ role: "user", content: prompt });
        chatHistory.push({ role: "assistant", content: data.response });

        appendChatMessage("assistant", data.response, data);
        fetchTelemetry();
    } catch (e) {
        removeChatLoading(loadingId);
        appendChatMessage("assistant", `Network error: ${e.message}`);
    }
}

function appendChatMessage(role, text, auditData = null) {
    const container = document.getElementById("chat-messages");
    const welcome = container.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${role}`;

    const roleName = role === "user" ? "You" : "Assistant";
    let auditHtml = "";

    if (auditData) {
        const audit = auditData.routing_audit || {};
        const selected = audit.selected || {};
        const score = auditData.complexity_score !== undefined ? auditData.complexity_score : 0.0;
        const tier = auditData.tier_used || "Tier 1";
        const tierClass = tier.toLowerCase().includes("1") ? "tier-1" : tier.toLowerCase().includes("2") ? "tier-2" : "tier-3";
        const costUsd = selected.actual_cost_usd !== undefined ? `$${selected.actual_cost_usd.toFixed(5)}` : "$0.0000";
        const latency = auditData.latency_seconds ? `${auditData.latency_seconds}s` : "";

        auditHtml = `
            <div class="audit-badge-row">
                <span class="audit-tag ${tierClass}">${auditData.provider || "Gateway"} &middot; ${auditData.model_name || "auto"}</span>
                <span class="audit-tag ${tierClass}">Score: ${score.toFixed(2)}</span>
                <span class="audit-tag">Cost: ${costUsd}</span>
                ${latency ? `<span class="audit-tag">Latency: ${latency}</span>` : ""}
                ${auditData.is_cached ? `<span class="audit-tag">Exact Cache Match</span>` : ""}
                ${auditData.was_redacted ? `<span class="audit-tag">PII Redacted</span>` : ""}
            </div>
        `;
    }

    let feedbackHtml = "";
    if (role === "assistant" && auditData && auditData.model_name && auditData.provider && !auditData.is_cached) {
        const feedbackId = `fb-${Date.now()}`;
        const safeModel = (auditData.model_name || "").replace(/'/g, "\\'");
        const safeProvider = (auditData.provider || "").replace(/'/g, "\\'");
        feedbackHtml = `
            <div class="feedback-row" id="${feedbackId}">
                <span class="feedback-label">Rate response:</span>
                <button class="feedback-btn" onclick="submitFeedback('${safeModel}', '${safeProvider}', 1, '${feedbackId}')" title="1 - Poor">★ 1</button>
                <button class="feedback-btn" onclick="submitFeedback('${safeModel}', '${safeProvider}', 2, '${feedbackId}')" title="2 - Fair">★ 2</button>
                <button class="feedback-btn" onclick="submitFeedback('${safeModel}', '${safeProvider}', 3, '${feedbackId}')" title="3 - Good">★ 3</button>
                <button class="feedback-btn" onclick="submitFeedback('${safeModel}', '${safeProvider}', 4, '${feedbackId}')" title="4 - Very Good">★ 4</button>
                <button class="feedback-btn" onclick="submitFeedback('${safeModel}', '${safeProvider}', 5, '${feedbackId}')" title="5 - Excellent">★ 5</button>
            </div>
        `;
    }

    msgDiv.innerHTML = `
        <div class="message-header">
            <span class="message-role">${roleName}</span>
        </div>
        <div class="message-body markdown-content">
            ${renderMarkdown(text)}
            ${auditHtml}
            ${feedbackHtml}
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendChatLoading() {
    const container = document.getElementById("chat-messages");
    const id = `loading-${Date.now()}`;
    const loadingDiv = document.createElement("div");
    loadingDiv.id = id;
    loadingDiv.className = "chat-message assistant";
    loadingDiv.innerHTML = `
        <div class="message-header"><span class="message-role">Cost Autopilot Router</span></div>
        <div class="message-body"><em>Analyzing complexity and routing to optimal tier...</em></div>
    `;
    container.appendChild(loadingDiv);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeChatLoading(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// -------------------------------------------------------------
// SECTION 2: CODING AGENT WORKSPACE LOGIC
// -------------------------------------------------------------
async function refreshWorkspaceFiles() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/workspace/files`);
        if (!resp.ok) return;
        const data = await resp.json();
        const treeList = document.getElementById("file-tree-list");
        treeList.innerHTML = "";

        data.files.forEach(f => {
            const item = document.createElement("div");
            item.className = `file-item ${f.path === activeFilePath ? "active" : ""}`;
            item.innerHTML = `<span>${getFileIcon(f.extension)}</span> <span>${f.path}</span>`;
            item.onclick = () => {
                document.querySelectorAll(".file-item").forEach(el => el.classList.remove("active"));
                item.classList.add("active");
                loadWorkspaceFile(f.path);
            };
            treeList.appendChild(item);
        });
    } catch (e) {
        console.warn("Error refreshing workspace files:", e);
    }
}

function getFileIcon(ext) {
    if (ext === ".py") return "🐍";
    if (ext === ".json") return "📦";
    if (ext === ".md") return "📝";
    if (ext === ".html" || ext === ".css" || ext === ".js") return "🌐";
    return "📄";
}

async function loadWorkspaceFile(path) {
    try {
        activeFilePath = path;
        const resp = await fetch(`${API_BASE}/api/v1/workspace/file?path=${encodeURIComponent(path)}`);
        if (!resp.ok) return;
        const data = await resp.json();

        document.getElementById("code-editor").value = data.content;
        const tab = document.getElementById("active-file-tab");
        if (tab) tab.textContent = path.split("/").pop();
    } catch (e) {
        console.warn("Failed to load file:", e);
    }
}

async function saveActiveFile() {
    try {
        const content = document.getElementById("code-editor").value;
        const resp = await fetch(`${API_BASE}/api/v1/workspace/file`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file_path: activeFilePath, content: content })
        });
        if (resp.ok) {
            alert(`File saved: ${activeFilePath}`);
        } else {
            alert("Failed to save file.");
        }
    } catch (e) {
        alert("Error saving file: " + e.message);
    }
}

// -------------------------------------------------------------
// Multi-Agent Workflow Execution (SSE Streaming)
// -------------------------------------------------------------
async function dispatchWorkflow() {
    const input = document.getElementById("wf-prompt-input");
    const prompt = input.value.trim();
    if (!prompt) return;

    const parallel = document.getElementById("wf-parallel-toggle")?.checked ?? true;
    const badge = document.getElementById("wf-status-badge");
    badge.className = "status-badge executing";
    badge.textContent = "EXECUTING";

    const dagContainer = document.getElementById("wf-task-list");
    dagContainer.innerHTML = `<div class="empty-hint">Supervisor is decomposing goal into subtasks...</div>`;

    const consoleOutput = document.getElementById("wf-console-output");
    consoleOutput.innerHTML = `<em>Workflow started: "${prompt.substring(0, 60)}..."</em><br>`;

    try {
        const resp = await fetch(`${API_BASE}/api/v1/workflow/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_prompt: prompt, parallel: parallel })
        });

        if (!resp.ok) {
            badge.className = "status-badge";
            badge.textContent = "FAILED";
            return;
        }

        const jobData = await resp.json();
        activeJobId = jobData.job_id;
        subscribeToWorkflowEvents(activeJobId);
    } catch (e) {
        badge.className = "status-badge";
        badge.textContent = "ERROR";
        consoleOutput.innerHTML += `<span style="color:red">Error: ${e.message}</span>`;
    }
}

function subscribeToWorkflowEvents(jobId) {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`${API_BASE}/api/v1/workflow/jobs/${jobId}/events`);

    eventSource.onmessage = (e) => {
        try {
            const event = JSON.parse(e.data);
            handleWorkflowEvent(event);
        } catch (err) {
            console.debug("SSE Parse skip:", e.data);
        }
    };

    eventSource.onerror = () => {
        if (eventSource) eventSource.close();
    };
}

function handleWorkflowEvent(event) {
    const dagContainer = document.getElementById("wf-task-list");
    const consoleOutput = document.getElementById("wf-console-output");
    const badge = document.getElementById("wf-status-badge");

    if (event.type === "plan_created") {
        dagContainer.innerHTML = "";
        (event.subtasks || []).forEach(task => {
            const card = document.createElement("div");
            card.id = `card-${task.task_id}`;
            card.className = "task-card";
            card.innerHTML = `
                <div class="task-top">
                    <span class="task-agent">${task.assigned_agent}</span>
                    <span class="task-tier-tag" id="tag-${task.task_id}">Pending</span>
                </div>
                <div class="task-desc">${task.description}</div>
            `;
            dagContainer.appendChild(card);
        });
        consoleOutput.innerHTML += `<strong>Plan:</strong> ${event.plan_overview}<br>`;
    } else if (event.type === "task_started") {
        const card = document.getElementById(`card-${event.task_id}`);
        if (card) {
            card.className = "task-card in_progress";
            const tag = document.getElementById(`tag-${event.task_id}`);
            if (tag) tag.textContent = "In Progress...";
        }
    } else if (event.type === "task_completed") {
        const card = document.getElementById(`card-${event.task_id}`);
        if (card) {
            card.className = "task-card completed";
            const tag = document.getElementById(`tag-${event.task_id}`);
            if (tag) tag.textContent = `${event.routed_tier || "Tier 2"} (Score: ${(event.complexity_score || 0).toFixed(2)})`;
        }
        consoleOutput.innerHTML += `[${event.assigned_agent}] Completed task.<br>`;
    } else if (event.type === "workflow_finished") {
        badge.className = "status-badge completed";
        badge.textContent = event.status === "completed" ? "FINISHED" : event.status.toUpperCase();

        if (event.status === "pending_hitl") {
            document.getElementById("hitl-panel").style.display = "block";
            currentWorkflowState = event;
        }

        consoleOutput.innerHTML += `<br><strong>=== FINAL SYNTHESIS ===</strong><br>${renderMarkdown(event.final_output || "Workflow finished.")}<br>`;
        if (eventSource) eventSource.close();
        fetchTelemetry();
    }
}

async function approveHITL() {
    if (!currentWorkflowState) return;
    try {
        await fetch(`${API_BASE}/api/v1/hitl/${currentWorkflowState.workflow_id}/approve`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: "Approved via UI" })
        });
        document.getElementById("hitl-panel").style.display = "none";
        document.getElementById("wf-status-badge").textContent = "APPROVED";
    } catch (e) {
        alert("Approval failed: " + e.message);
    }
}

async function rejectHITL() {
    if (!currentWorkflowState) return;
    try {
        await fetch(`${API_BASE}/api/v1/hitl/${currentWorkflowState.workflow_id}/reject`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: "Rejected via UI" })
        });
        document.getElementById("hitl-panel").style.display = "none";
        document.getElementById("wf-status-badge").textContent = "REJECTED";
    } catch (e) {
        alert("Rejection failed: " + e.message);
    }
}

// -------------------------------------------------------------
// Pareto Chart Rendering (Zero Dependency Canvas)
// -------------------------------------------------------------
async function renderParetoChart() {
    const canvas = document.getElementById("pareto-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const resp = await fetch(`${API_BASE}/api/v1/telemetry/pareto`);
        if (!resp.ok) return;
        const data = await resp.json();
        const routes = data.recent_routes || [];

        // Grid lines
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(25, 10); ctx.lineTo(25, 110); ctx.lineTo(250, 110);
        ctx.stroke();

        if (routes.length === 0) {
            ctx.fillStyle = "#64748b";
            ctx.font = "10px sans-serif";
            ctx.fillText("Waiting for routed traffic...", 50, 60);
            return;
        }

        routes.forEach(r => {
            const x = 30 + Math.min(r.complexity_score || 0.5, 1.0) * 200;
            const y = 105 - Math.min((r.estimated_cost_usd || 0.0001) * 20000, 90);

            ctx.fillStyle = (r.tier || "").includes("1") ? "#10b981" : (r.tier || "").includes("2") ? "#38bdf8" : "#f59e0b";
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        });

        const summary = document.getElementById("pareto-summary");
        if (summary) summary.textContent = `${routes.length} requests plotted across cost/complexity tiers`;
    } catch (e) {
        console.debug("Pareto draw skip:", e);
    }
}

// -------------------------------------------------------------
// Modals & Context Bridge
// -------------------------------------------------------------
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.style.display = "flex";
    if (id === "modal-bridge") generateBridgePrompt();
    if (id === "modal-rag") fetchRAGStats();
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.style.display = "none";
}

async function generateBridgePrompt() {
    const platform = document.getElementById("bridge-platform").value;
    const activeCode = document.getElementById("code-editor").value;
    const outputArea = document.getElementById("bridge-output");

    try {
        const resp = await fetch(`${API_BASE}/api/v1/bridge/export`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target_platform: platform,
                messages: chatHistory.length > 0 ? chatHistory : [{ role: "user", content: "Agent Orchestration Workspace" }],
                active_code: activeCode.substring(0, 2500)
            })
        });

        if (resp.ok) {
            const data = await resp.json();
            outputArea.value = data.handoff_prompt;
        }
    } catch (e) {
        outputArea.value = "Failed to generate bridge export: " + e.message;
    }
}

function copyBridgePrompt() {
    const outputArea = document.getElementById("bridge-output");
    navigator.clipboard.writeText(outputArea.value);
    const btnText = document.getElementById("copy-btn-text");
    btnText.textContent = "Copied to Clipboard!";
    setTimeout(() => { btnText.textContent = "Copy Prompt to Clipboard"; }, 2000);
}

// -------------------------------------------------------------
// Local RAG Operations
// -------------------------------------------------------------
async function fetchRAGStats() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/stats`);
        if (!resp.ok) return;
        const data = await resp.json();

        const chunksEl = document.getElementById("modal-rag-chunks");
        const filesEl = document.getElementById("modal-rag-files");
        const vocabEl = document.getElementById("modal-rag-vocab");

        if (chunksEl) chunksEl.textContent = data.total_chunks || 0;
        if (filesEl) filesEl.textContent = data.indexed_files_count || 0;
        if (vocabEl) vocabEl.textContent = data.vocabulary_size || 0;
    } catch (e) {
        console.debug("RAG stats fetch error:", e);
    }
}

async function testRAGSearch() {
    const query = document.getElementById("rag-test-query").value.trim();
    if (!query) return;

    const resultsContainer = document.getElementById("rag-search-results");
    resultsContainer.innerHTML = "<em>Searching index...</em>";

    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, top_k: 3 })
        });
        const data = await resp.json();
        resultsContainer.innerHTML = "";

        if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = "<div class='empty-hint'>No matching snippets found. Try re-indexing.</div>";
            return;
        }

        data.results.forEach(r => {
            const card = document.createElement("div");
            card.className = "result-card";
            card.innerHTML = `
                <div class="result-header">
                    <span>${r.doc_id} (Score: ${r.score})</span>
                    <span>Lines ${r.metadata?.start_line || 1}-${r.metadata?.end_line || "?"}</span>
                </div>
                <div class="result-snippet">${escapeHtml(r.snippet.substring(0, 200))}...</div>
            `;
            resultsContainer.appendChild(card);
        });
    } catch (e) {
        resultsContainer.innerHTML = `<span style="color:red">Search failed: ${e.message}</span>`;
    }
}

async function reindexWorkspace() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/index`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });
        if (resp.ok) {
            const data = await resp.json();
            alert(`Indexed ${data.files_indexed} files into ${data.total_chunks} chunks.`);
            fetchRAGStats();
            fetchHealth();
        }
    } catch (e) {
        alert("Re-index failed: " + e.message);
    }
}

async function clearRAGIndex() {
    if (!confirm("Are you sure you want to clear the RAG knowledge index?")) return;
    try {
        await fetch(`${API_BASE}/api/v1/rag/clear`, { method: "DELETE" });
        fetchRAGStats();
        fetchHealth();
    } catch (e) {
        alert("Failed to clear index: " + e.message);
    }
}

// -------------------------------------------------------------
// Markdown Helpers
// -------------------------------------------------------------
function renderMarkdown(raw) {
    if (!raw) return "";
    let formatted = escapeHtml(raw);

    // Code blocks
    formatted = formatted.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Headings
    formatted = formatted.replace(/^### (.*$)/gim, "<h4>$1</h4>");
    formatted = formatted.replace(/^## (.*$)/gim, "<h3>$1</h3>");
    formatted = formatted.replace(/^# (.*$)/gim, "<h2>$1</h2>");

    // Bold
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Line breaks
    formatted = formatted.replace(/\n\n/g, "<br><br>");
    formatted = formatted.replace(/\n/g, "<br>");

    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// -------------------------------------------------------------
// SECTION 3: ROUTER ANALYTICS
// -------------------------------------------------------------
async function loadRouterAnalytics() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/router/performance`);
        if (!resp.ok) return;
        const data = await resp.json();

        const matrix = data.performance_matrix || {};
        const bench = data.benchmark_status || {};
        const report = data.latest_training_report || {};

        // Summary cards
        setText("an-total-obs", matrix.total_observations || 0);
        setText("an-total-feedback", matrix.total_feedback_entries || 0);
        setText("an-train-ready", matrix.training_ready ? "Yes" : `No (need ${matrix.recommended_retrain_at || 200})`);
        setText("an-classifier-engine", data.classifier_ml_active ? "XGBoost/GBDT" : "Heuristics");
        setText("an-bench-rows", bench.benchmark_dataset_rows || 0);
        setText("an-accuracy", report.accuracy ? `${(report.accuracy * 100).toFixed(1)}%` : "N/A");

        // Benchmark info
        setText("bench-loaded", bench.benchmarks_loaded ? "Yes" : "No");
        setText("bench-dataset-rows", bench.benchmark_dataset_rows || 0);
        setText("bench-last-cal", bench.last_calibrated || "Never");
        setText("bench-sources", (bench.sources || []).join(", ") || "None");

        // Training report
        if (report.engine) {
            setText("tr-engine", report.engine);
            setText("tr-samples", `${report.total_samples} (${report.synthetic_samples} synthetic + ${report.benchmark_samples} benchmark + ${report.observation_samples} live)`);
            setText("tr-accuracy", `${(report.accuracy * 100).toFixed(2)}%`);
            const f1 = report.per_class_f1 || {};
            setText("tr-f1-t1", f1["Tier 1"] || "N/A");
            setText("tr-f1-t2", f1["Tier 2"] || "N/A");
            setText("tr-f1-t3", f1["Tier 3"] || "N/A");
            setText("tr-duration", `${report.duration_seconds}s`);
            setText("tr-trained-at", report.time_iso || "N/A");
        }

        // Model performance table
        const tbody = document.getElementById("model-perf-tbody");
        const models = matrix.models || {};
        const modelKeys = Object.keys(models);

        if (modelKeys.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-hint">No observations yet. Send prompts to start collecting data.</td></tr>';
        } else {
            tbody.innerHTML = modelKeys.map(key => {
                const m = models[key];
                const ratingDisplay = m.avg_user_rating !== null ? `${m.avg_user_rating}/5 (${m.feedback_count})` : "No ratings";
                const totalTok = (m.total_input_tokens + m.total_output_tokens).toLocaleString();
                return `<tr>
                    <td class="model-name-cell">${escapeHtml(key)}</td>
                    <td>${m.total_requests}</td>
                    <td>${m.avg_latency_ms.toFixed(0)}ms</td>
                    <td>${totalTok}</td>
                    <td>$${m.total_cost_usd.toFixed(6)}</td>
                    <td>${(m.error_rate * 100).toFixed(1)}%</td>
                    <td>${ratingDisplay}</td>
                    <td>${m.last_used || "N/A"}</td>
                </tr>`;
            }).join("");
        }
    } catch (e) {
        console.warn("Router analytics fetch failed:", e);
    }
}

async function retrainClassifier() {
    const btn = document.getElementById("btn-retrain");
    if (btn) {
        btn.disabled = true;
        btn.querySelector("span").textContent = "Training...";
    }
    try {
        const resp = await fetch(`${API_BASE}/api/v1/router/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ include_benchmarks: true, include_observations: true, min_obs: 10 })
        });
        const data = await resp.json();
        if (resp.ok) {
            alert(`Retraining complete. Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}% on ${data.metrics.total_samples} samples.`);
            loadRouterAnalytics();
            fetchHealth();
        } else {
            alert("Retraining failed: " + (data.detail || "Unknown error"));
        }
    } catch (e) {
        alert("Retraining request failed: " + e.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.querySelector("span").textContent = "Retrain Classifier";
        }
    }
}

async function submitFeedback(model, provider, rating, feedbackElementId) {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/router/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, provider, rating })
        });
        if (resp.ok) {
            const fbEl = document.getElementById(feedbackElementId);
            if (fbEl) {
                fbEl.innerHTML = `<span class="feedback-label">Rated ${rating}/5. Thank you.</span>`;
            }
        }
    } catch (e) {
        console.warn("Feedback submission failed:", e);
    }
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}
