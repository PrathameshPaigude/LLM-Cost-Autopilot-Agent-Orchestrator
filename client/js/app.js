/**
 * Agent Orchestrator & LLM Cost Autopilot - Client Application Logic (v2.1.0)
 * Supports: Workspace Launcher, Automatic Quota Rollover Chain, Prompt Vault SQLite Archive,
 * Token Compression Engine, LLM Chat, Coding Agent IDE, Router Analytics, and Telemetry.
 */

const API_BASE = window.location.origin;

// State
let chatHistory = [];
let activeFilePath = "server/app/main.py";
let activeJobId = null;
let eventSource = null;
let currentWorkflowState = null;
let vaultDebounceTimer = null;

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    await fetchHealth();
    await fetchTelemetry();
    await fetchRAGStats();
    await checkQuotaChainStatus();
    await loadPromptVault();
    await refreshWorkspaceFiles();
    loadWorkspaceFile(activeFilePath);

    // Periodic telemetry & status updates
    setInterval(fetchTelemetry, 8000);
    setInterval(checkQuotaChainStatus, 15000);
}

// -------------------------------------------------------------
// Section Navigation
// -------------------------------------------------------------
function switchMainSection(section) {
    document.querySelectorAll(".workspace-section").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-tab").forEach(el => el.classList.remove("active"));

    if (section === "home" || section === "overview") {
        document.getElementById("section-home")?.classList.add("active");
        document.getElementById("tab-home")?.classList.add("active");
        loadPromptVault();
    } else if (section === "chat") {
        document.getElementById("section-chat")?.classList.add("active");
        document.getElementById("tab-chat")?.classList.add("active");
    } else if (section === "coding") {
        document.getElementById("section-coding")?.classList.add("active");
        document.getElementById("tab-coding")?.classList.add("active");
    } else if (section === "analytics") {
        document.getElementById("section-analytics")?.classList.add("active");
        document.getElementById("tab-analytics")?.classList.add("active");
        loadRouterAnalytics();
    } else if (section === "vault") {
        document.getElementById("section-vault")?.classList.add("active");
        document.getElementById("tab-vault")?.classList.add("active");
        loadPromptVault();
    }
}

// -------------------------------------------------------------
// SECTION 0: WORKSPACE LAUNCHER & QUICK PROMPT LOGIC
// -------------------------------------------------------------
async function activatePrivacyAndLaunch() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/config/privacy`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ privacy_mode: true })
        });
        if (resp.ok) {
            const privLabel = document.getElementById("privacy-label");
            if (privLabel) privLabel.textContent = "Air-Gapped";
            alert("Complete Air-Gapped Privacy Mode Activated. All cloud API requests are intercepted & blocked. Offline Ollama execution is enforced.");
            switchMainSection("chat");
        }
    } catch (e) {
        console.error("Privacy mode toggle failed:", e);
    }
}

function handleQuickPromptKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleQuickPromptSubmit(event);
    }
}

async function handleQuickPromptSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("quick-prompt-input");
    const prompt = (input?.value || "").trim();
    if (!prompt) return;

    const mode = document.getElementById("quick-mode-select")?.value || "chat";
    const comp = document.getElementById("quick-compress-select")?.value || "balanced";
    const btn = document.getElementById("btn-quick-run");
    if (btn) {
        btn.disabled = true;
        btn.querySelector("span").textContent = "Executing...";
    }

    const resultCard = document.getElementById("quick-result-card");
    const qrBody = document.getElementById("qr-text");
    if (resultCard) resultCard.style.display = "block";
    if (qrBody) qrBody.innerHTML = "<em>Analyzing complexity, applying token compression, and routing across providers...</em>";

    try {
        let data;
        if (mode === "quota-chain") {
            const resp = await fetch(`${API_BASE}/v1/chat/completions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [{ role: "user", content: prompt }],
                    compression: comp
                })
            });
            const resJson = await resp.json();
            data = {
                response: resJson.choices?.[0]?.message?.content || "No response",
                model_name: resJson.model,
                provider: resJson.quota_chain?.provider_used || "Quota Chain",
                tier_used: "Quota Rollover",
                tokens_saved: resJson.quota_chain?.tokens_saved || 0,
                latency_ms: resJson.quota_chain?.latency_ms || 0
            };
        } else if (mode === "coding") {
            switchMainSection("coding");
            const wfInput = document.getElementById("wf-prompt-input");
            if (wfInput) wfInput.value = prompt;
            dispatchWorkflow();
            return;
        } else {
            const resp = await fetch(`${API_BASE}/api/v1/route`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: prompt, agent_name: "QuickLauncher" })
            });
            data = await resp.json();
        }

        setText("qr-provider-model", `${data.provider || "Gateway"} · ${data.model_name || "auto"}`);
        setText("qr-tier", data.tier_used || "Tier 1");
        setText("qr-tokens", `${data.input_tokens || 0} in / ${data.output_tokens || 0} out`);
        setText("qr-savings", data.tokens_saved ? `Saved ${data.tokens_saved} tok` : `Cost: $${(data.actual_cost_usd || 0).toFixed(5)}`);
        if (qrBody) qrBody.innerHTML = renderMarkdown(data.response || (data.error ? `⚠️ Error: ${data.error}` : "No response generated by model."));

        fetchTelemetry();
        loadPromptVault();
    } catch (err) {
        if (qrBody) qrBody.innerHTML = `<span style="color: var(--color-danger);">Execution Error: ${err.message}</span>`;
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.querySelector("span").textContent = "Run Prompt";
        }
    }
}

function closeQuickResult() {
    const el = document.getElementById("quick-result-card");
    if (el) el.style.display = "none";
}

// -------------------------------------------------------------
// SECTION 4: PROMPT VAULT SQLITE LOGIC
// -------------------------------------------------------------
function debounceVaultSearch() {
    clearTimeout(vaultDebounceTimer);
    vaultDebounceTimer = setTimeout(loadPromptVault, 300);
}

async function loadPromptVault() {
    try {
        const statsResp = await fetch(`${API_BASE}/api/v1/vault/stats`);
        if (statsResp.ok) {
            const stats = await statsResp.json();
            setText("vault-total-count", stats.total_prompts || 0);
            setText("vault-orig-tokens", (stats.total_tokens_original || 0).toLocaleString());
            setText("vault-comp-tokens", (stats.total_tokens_compressed || 0).toLocaleString());
            setText("vault-saved-tokens", (stats.total_tokens_saved || 0).toLocaleString());
            const savedPct = stats.avg_compression_ratio ? Math.round((1 - stats.avg_compression_ratio) * 100) : 0;
            setText("vault-avg-ratio", savedPct > 0 ? `${savedPct}% Saved` : "1.0 (Raw)");
            setText("vault-train-count", stats.training_ready_count || 0);

            // Update radar cards on launcher
            setText("radar-vault-count", `${stats.total_prompts || 0} Prompts Stored`);
            setText("radar-tokens-saved", `${(stats.total_tokens_saved || 0).toLocaleString()} Tokens Saved`);
            if (savedPct > 0) {
                setText("radar-comp-pct", `${savedPct}% Saved`);
            }
        }

        const search = (document.getElementById("vault-search-input")?.value || "").trim();
        const tier = document.getElementById("vault-tier-filter")?.value || "";
        let url = `${API_BASE}/api/v1/vault/prompts?limit=50`;
        if (search) url += `&search=${encodeURIComponent(search)}`;
        if (tier) url += `&tier=${encodeURIComponent(tier)}`;

        const listResp = await fetch(url);
        if (!listResp.ok) return;
        const listData = await listResp.json();
        const tbody = document.getElementById("vault-prompts-tbody");
        if (!tbody) return;

        const prompts = listData.prompts || [];
        if (prompts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="empty-hint">No stored prompts yet. Send queries to accumulate training samples.</td></tr>';
        } else {
            tbody.innerHTML = prompts.map(p => {
                const promptSnippet = escapeHtml((p.original_prompt || "").slice(0, 65));
                const compSnippet = escapeHtml((p.compressed_prompt || "").slice(0, 65));
                const ratingStr = p.user_rating ? `★ ${p.user_rating}/5` : '-';
                const created = (p.created_at || "").slice(11, 19);
                return `<tr>
                    <td>${p.id}</td>
                    <td>${created}</td>
                    <td title="${escapeHtml(p.original_prompt)}">${promptSnippet}</td>
                    <td title="${escapeHtml(p.compressed_prompt || '')}">${compSnippet}</td>
                    <td><span class="tag-meta">${escapeHtml(p.tier_used)}</span></td>
                    <td>${escapeHtml(p.model_name)}</td>
                    <td>${p.tokens_in} / ${p.tokens_out}</td>
                    <td>${p.latency_ms ? Math.round(p.latency_ms) + 'ms' : '-'}</td>
                    <td>${ratingStr}</td>
                </tr>`;
            }).join("");
        }
    } catch (e) {
        console.warn("Vault fetch failed:", e);
    }
}

async function exportVaultDataset() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/vault/export`);
        if (!resp.ok) return alert("Failed to export dataset");
        const data = await resp.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `prompt_vault_training_dataset_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        alert("Export failed: " + e.message);
    }
}

// -------------------------------------------------------------
// QUOTA ROLLOVER CHAIN & HEALTH LOGIC
// -------------------------------------------------------------
async function checkQuotaChainStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/quota-chain/status`);
        if (!resp.ok) return;
        const data = await resp.json();
        const badge = document.getElementById("quota-chain-badge") || document.getElementById("omniroute-badge");
        const healthyCount = (data.providers || []).filter(p => p.status === "healthy").length;
        if (badge) {
            badge.title = `Quota Rollover Active: ${healthyCount} healthy providers in rollover chain`;
        }
    } catch (e) {
        // silent
    }
}

async function fetchHealth() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/health`);
        if (!resp.ok) return;
        const data = await resp.json();

        // Update indicators in Settings Modal
        const providers = data.providers || {};
        updateIndicator("ind-gemini", providers.gemini?.configured);
        updateIndicator("ind-groq", providers.groq?.configured);
        updateIndicator("ind-huggingface", providers.huggingface?.configured);
        updateIndicator("ind-openrouter", providers.openrouter?.configured);
        updateIndicator("ind-openai", providers.openai?.configured);

        // Update Privacy Mode label
        const privLabel = document.getElementById("privacy-label");
        if (privLabel) {
            privLabel.textContent = data.privacy_mode ? "Air-Gapped" : "Cloud Active";
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

function updateIndicator(elementId, isOnline) {
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

        // Settings modal telemetry
        setText("settings-stat-saved", `$${(data.estimated_money_saved_usd || 0).toFixed(4)}`);
        setText("settings-stat-tokens", (data.total_tokens_routed || 0).toLocaleString());
        const joules = data.estimated_joules_saved || 0;
        setText("settings-stat-energy", joules / 3600 < 0.1 ? `${joules.toFixed(1)} J` : `${(joules / 3600).toFixed(2)} Wh`);

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

async function clearCache() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/debug/cache`, { method: "DELETE" });
        if (resp.ok) {
            alert("Exact-match cache cleared successfully.");
        }
    } catch (e) {
        alert("Clear cache failed: " + e.message);
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
            <h2>LLM Chat & Cost Autopilot</h2>
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
        const colonIdx = modelSelect.indexOf(":");
        if (colonIdx !== -1) {
            providerOverride = modelSelect.substring(0, colonIdx);
            modelOverride = modelSelect.substring(colonIdx + 1);
        } else {
            providerOverride = modelSelect;
        }
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
            const err = await resp.json().catch(() => ({}));
            appendChatMessage("assistant", `⚠️ Error: ${err.detail || "Route execution failed"}`);
            return;
        }

        const data = await resp.json();
        const responseText = data.response || (data.error ? `⚠️ Error: ${data.error}` : "No response generated by model.");
        chatHistory.push({ role: "user", content: prompt });
        chatHistory.push({ role: "assistant", content: responseText });

        appendChatMessage("assistant", responseText, data);
        fetchTelemetry();
        loadPromptVault();
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
            ${renderMarkdown(text || (auditData?.error ? `⚠️ Error: ${auditData.error}` : "(No response content)"))}
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
        if (!treeList) return;

        const files = data.files || [];
        if (files.length === 0) {
            treeList.innerHTML = '<div class="empty-hint">No files found in workspace.</div>';
            return;
        }

        treeList.innerHTML = files.map(file => {
            const isSelected = file === activeFilePath;
            return `
                <div class="file-tree-item ${isSelected ? 'active' : ''}" onclick="loadWorkspaceFile('${file}')" title="${file}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                    <span>${file}</span>
                </div>
            `;
        }).join("");
    } catch (e) {
        console.warn("Files list failed:", e);
    }
}

async function loadWorkspaceFile(path) {
    if (!path) return;
    activeFilePath = path;
    const tabEl = document.getElementById("active-file-tab");
    if (tabEl) tabEl.textContent = path.split("/").pop();

    try {
        const resp = await fetch(`${API_BASE}/api/v1/workspace/file?path=${encodeURIComponent(path)}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const editor = document.getElementById("code-editor");
        if (editor) editor.value = data.content;

        document.querySelectorAll(".file-tree-item").forEach(item => {
            item.classList.toggle("active", item.getAttribute("title") === path);
        });
    } catch (e) {
        console.warn("File read error:", e);
    }
}

async function saveActiveFile() {
    const editor = document.getElementById("code-editor");
    if (!editor) return;
    const content = editor.value;

    try {
        const resp = await fetch(`${API_BASE}/api/v1/workspace/file`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file_path: activeFilePath, content: content })
        });
        if (resp.ok) {
            alert(`File saved: ${activeFilePath}`);
        } else {
            const err = await resp.json();
            alert(`Save failed: ${err.detail || "Unknown error"}`);
        }
    } catch (e) {
        alert("Save request failed: " + e.message);
    }
}

async function dispatchWorkflow() {
    const input = document.getElementById("wf-prompt-input");
    const prompt = input.value.trim();
    if (!prompt) return;

    const parallel = document.getElementById("wf-parallel-toggle").checked;
    const badge = document.getElementById("wf-status-badge");
    badge.textContent = "RUNNING";
    badge.style.color = "var(--color-info)";

    const taskList = document.getElementById("wf-task-list");
    taskList.innerHTML = '<div class="empty-hint">Decomposing goal and dispatching agents...</div>';

    const consoleOut = document.getElementById("wf-console-output");
    consoleOut.innerHTML = '<em>Workflow started. Streaming progress...</em>';

    try {
        const resp = await fetch(`${API_BASE}/api/v1/workflow/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_prompt: prompt, parallel: parallel })
        });

        if (!resp.ok) {
            badge.textContent = "FAILED";
            badge.style.color = "var(--color-danger)";
            return;
        }

        const data = await resp.json();
        activeJobId = data.job_id;
        streamWorkflowEvents(activeJobId);
    } catch (e) {
        badge.textContent = "ERROR";
        badge.style.color = "var(--color-danger)";
        consoleOut.textContent = `Error: ${e.message}`;
    }
}

function streamWorkflowEvents(jobId) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`${API_BASE}/api/v1/workflow/jobs/${jobId}/events`);

    eventSource.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            handleWorkflowEvent(payload);
        } catch (e) {
            console.warn("Event parse error:", e);
        }
    };

    eventSource.onerror = () => {
        if (eventSource) eventSource.close();
    };
}

function handleWorkflowEvent(payload) {
    const type = payload.type;
    const data = payload.data || {};
    const taskList = document.getElementById("wf-task-list");
    const consoleOut = document.getElementById("wf-console-output");
    const badge = document.getElementById("wf-status-badge");

    if (type === "workflow_started") {
        consoleOut.innerHTML = `<div><strong>Workflow started:</strong> "${escapeHtml(data.prompt)}"</div><div><strong>Plan:</strong> ${escapeHtml(data.plan)}</div>`;
    } else if (type === "task_completed") {
        const taskDiv = document.createElement("div");
        taskDiv.className = "dag-task-card";
        taskDiv.innerHTML = `
            <div class="dag-task-header">
                <span>${escapeHtml(data.agent_name)}</span>
                <span class="audit-tag tier-2">${escapeHtml(data.tier_used || 'Tier 2')}</span>
            </div>
            <div class="dag-task-desc">${escapeHtml(data.task_description)}</div>
        `;
        taskList.appendChild(taskDiv);
        consoleOut.innerHTML += `<div>[${escapeHtml(data.agent_name)}] Completed task.</div>`;
    } else if (type === "hitl_required") {
        document.getElementById("hitl-panel").style.display = "block";
        badge.textContent = "WAITING REVIEW";
        badge.style.color = "var(--color-warning)";
    } else if (type === "workflow_completed") {
        badge.textContent = "FINISHED";
        badge.style.color = "var(--color-success)";
        consoleOut.innerHTML += `
            <hr style="border-color: var(--border-subtle); margin: 8px 0;">
            <div><strong>FINAL SYNTHESIS</strong></div>
            <div class="markdown-content">${renderMarkdown(data.final_synthesis || "Workflow completed.")}</div>
        `;
        if (eventSource) eventSource.close();
        fetchTelemetry();
        loadPromptVault();
    }
}

async function approveHITL() {
    if (!activeJobId) return;
    try {
        await fetch(`${API_BASE}/api/v1/hitl/${activeJobId}/approve`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: "Approved via dashboard" })
        });
        document.getElementById("hitl-panel").style.display = "none";
    } catch (e) {
        alert("Approve error: " + e.message);
    }
}

async function rejectHITL() {
    if (!activeJobId) return;
    try {
        await fetch(`${API_BASE}/api/v1/hitl/${activeJobId}/reject`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: "Rejected via dashboard" })
        });
        document.getElementById("hitl-panel").style.display = "none";
    } catch (e) {
        alert("Reject error: " + e.message);
    }
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

            // Also update radar classifier accuracy on launcher
            setText("radar-classifier-acc", `${(report.accuracy * 100).toFixed(1)}% Accuracy`);
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
                fbEl.innerHTML = `<span class="feedback-label">Rated ${rating}/5. Saved to training database.</span>`;
            }
        }
    } catch (e) {
        console.warn("Feedback submission failed:", e);
    }
}

// -------------------------------------------------------------
// RAG, CONTEXT BRIDGE & MODALS
// -------------------------------------------------------------
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.style.display = "flex";
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.style.display = "none";
}

async function fetchRAGStats() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/stats`);
        if (!resp.ok) return;
        const data = await resp.json();
        setText("modal-rag-chunks", data.total_chunks || 0);
        setText("modal-rag-files", data.indexed_files_count || 0);
        setText("modal-rag-vocab", data.vocab_size || 0);
    } catch (e) {
        console.debug("RAG stats error:", e);
    }
}

async function reindexWorkspace() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/index`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ directory_path: "." })
        });
        const data = await resp.json();
        alert(`Indexing complete. ${data.chunks_indexed} chunks indexed across ${data.files_indexed} files.`);
        fetchRAGStats();
        fetchHealth();
    } catch (e) {
        alert("Indexing failed: " + e.message);
    }
}

async function clearRAGIndex() {
    try {
        await fetch(`${API_BASE}/api/v1/rag/clear`, { method: "DELETE" });
        alert("RAG Index cleared.");
        fetchRAGStats();
        fetchHealth();
    } catch (e) {
        alert("Clear error: " + e.message);
    }
}

async function testRAGSearch() {
    const query = document.getElementById("rag-test-query").value.trim();
    if (!query) return;

    try {
        const resp = await fetch(`${API_BASE}/api/v1/rag/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, top_k: 3 })
        });
        const data = await resp.json();
        const resultsDiv = document.getElementById("rag-search-results");
        const results = data.results || [];

        if (results.length === 0) {
            resultsDiv.innerHTML = '<div class="empty-hint">No matches found.</div>';
            return;
        }

        resultsDiv.innerHTML = results.map(r => `
            <div class="result-card">
                <div class="result-header">
                    <span>${escapeHtml(r.doc_id)} (Score: ${r.score.toFixed(3)})</span>
                </div>
                <div class="result-snippet">${escapeHtml(r.content)}</div>
            </div>
        `).join("");
    } catch (e) {
        alert("Search error: " + e.message);
    }
}

async function generateBridgePrompt() {
    const platform = document.getElementById("bridge-platform").value;
    const editor = document.getElementById("code-editor");
    const activeCode = editor ? editor.value : "";

    try {
        const resp = await fetch(`${API_BASE}/api/v1/bridge/export`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target_platform: platform,
                messages: chatHistory,
                active_code: activeCode
            })
        });
        const data = await resp.json();
        document.getElementById("bridge-output").value = data.transfer_prompt || "";
    } catch (e) {
        console.warn("Bridge export error:", e);
    }
}

function copyBridgePrompt() {
    const out = document.getElementById("bridge-output");
    out.select();
    navigator.clipboard.writeText(out.value);
    const btnText = document.getElementById("copy-btn-text");
    btnText.textContent = "Copied to Clipboard!";
    setTimeout(() => { btnText.textContent = "Copy Prompt to Clipboard"; }, 2000);
}

// -------------------------------------------------------------
// Pareto Chart & Helpers
// -------------------------------------------------------------
function renderParetoChart() {
    const canvas = document.getElementById("pareto-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 1;
    ctx.strokeRect(20, 10, canvas.width - 30, canvas.height - 25);

    const points = [
        { x: 35, y: 95, label: "T1 (0.12)", color: "#10b981" },
        { x: 130, y: 55, label: "T2 (0.61)", color: "#38bdf8" },
        { x: 220, y: 25, label: "T3 (0.95)", color: "#f59e0b" }
    ];

    ctx.beginPath();
    ctx.strokeStyle = "#3f3f46";
    ctx.setLineDash([3, 3]);
    points.forEach((p, idx) => {
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    points.forEach(p => {
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#a1a1aa";
        ctx.font = "9px Inter, sans-serif";
        ctx.fillText(p.label, p.x - 14, p.y - 7);
    });
}

function renderMarkdown(text) {
    if (!text) return "";
    let formatted = escapeHtml(text);

    // Code blocks
    formatted = formatted.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre class="code-block"><code class="lang-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // Bold & italic
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    formatted = formatted.replace(/\*([^*]+)\*/g, "<em>$1</em>");

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

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}
