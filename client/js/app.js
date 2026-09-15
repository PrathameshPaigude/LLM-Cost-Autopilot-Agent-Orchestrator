const API_BASE = "http://localhost:8000/api/v1";

const promptInput = document.getElementById("prompt-input");
const btnRun = document.getElementById("btn-run");
const subtasksList = document.getElementById("subtasks-list");
const outputBox = document.getElementById("output-box");
const planOverviewText = document.getElementById("plan-overview-text");
const currentAgentTag = document.getElementById("current-agent-tag");
const workflowStatus = document.getElementById("workflow-status");
const privacyToggle = document.getElementById("privacy-toggle");
const privacyStatusText = document.getElementById("privacy-status-text");
const privacySubText = document.getElementById("privacy-sub-text");
const modeSelect = document.getElementById("mode-select");

const statSaved = document.getElementById("stat-saved");
const statTokens = document.getElementById("stat-tokens");
const statEnergy = document.getElementById("stat-energy");

// Provider Status Elements
const pillGemini = document.getElementById("pill-gemini");
const pillGroq = document.getElementById("pill-groq");
const pillOpenRouter = document.getElementById("pill-openrouter");
const pillOpenAI = document.getElementById("pill-openai");
const pillOllama = document.getElementById("pill-ollama");

// Set Sample Prompt
function setPrompt(text) {
    promptInput.value = text;
    promptInput.focus();
}

// Fetch Health & Provider Status
async function updateHealthAndProviders() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            
            if (data.providers) {
                updatePill(pillGemini, data.providers.gemini?.configured);
                updatePill(pillGroq, data.providers.groq?.configured);
                updatePill(pillOpenRouter, data.providers.openrouter?.configured);
                updatePill(pillOpenAI, data.providers.openai?.configured);
                updatePill(pillOllama, data.providers.ollama?.online);
            }

            privacyToggle.checked = Boolean(data.privacy_mode);
            updatePrivacyUI(data.privacy_mode);
        }
    } catch (e) {
        [pillGemini, pillGroq, pillOpenRouter, pillOpenAI, pillOllama].forEach(p => p && p.classList.remove("active"));
    }
}

function updatePill(elem, isOnline) {
    if (!elem) return;
    if (isOnline) {
        elem.classList.add("active");
    } else {
        elem.classList.remove("active");
    }
}

function updatePrivacyUI(isPrivate) {
    privacyStatusText.textContent = isPrivate ? "🔒 Offline Air-Gapped" : "🌐 Hybrid Cloud Free";
    privacySubText.textContent = isPrivate ? "100% Local (Ollama)" : "Ollama + Gemini + Groq";
}

// Fetch Real-Time Telemetry
async function updateTelemetry() {
    try {
        const res = await fetch(`${API_BASE}/telemetry`);
        if (res.ok) {
            const data = await res.json();
            statSaved.textContent = `$${data.net_dollars_saved_usd.toFixed(4)}`;
            statTokens.textContent = data.total_tokens_processed.toLocaleString();
            statEnergy.textContent = `${data.energy_saved_watt_hours.toFixed(1)} Wh`;
        }
    } catch (e) {}
}

// Toggle Privacy Mode
privacyToggle.addEventListener("change", async (e) => {
    const isPrivate = e.target.checked;
    updatePrivacyUI(isPrivate);
    try {
        await fetch(`${API_BASE}/config/privacy`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ privacy_mode: isPrivate })
        });
    } catch (err) {
        console.error("Privacy toggle error:", err);
    }
});

// Run Workflow
btnRun.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    workflowStatus.textContent = "PLANNING";
    currentAgentTag.textContent = "Supervisor";
    planOverviewText.textContent = "Analyzing task and decomposing into specialist nodes...";
    
    subtasksList.innerHTML = `
        <div class="empty-state" style="padding: 20px;">
            <div style="font-size: 18px; margin-bottom: 8px;">⏳</div>
            <div>Supervisor is planning DAG...</div>
        </div>
    `;

    outputBox.innerHTML = `
        <div class="placeholder-msg">
            <h3>🔄 Multi-Agent Team in Progress...</h3>
            <p>Supervisor is routing tasks across the Cost Autopilot dynamic execution matrix.</p>
        </div>
    `;
    btnRun.disabled = true;

    // Parse Mode Override
    const modeVal = modeSelect.value;
    let providerOverride = null;
    let modelOverride = null;
    if (modeVal !== "auto") {
        const parts = modeVal.split(":");
        providerOverride = parts[0];
        modelOverride = parts.slice(1).join(":");
    }

    try {
        const response = await fetch(`${API_BASE}/workflow/jobs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_prompt: prompt,
                force_offline: privacyToggle.checked,
                provider_override: providerOverride,
                model_override: modelOverride
            })
        });

        if (!response.ok) throw new Error("Workflow job could not be created");

        const job = await response.json();
        const data = await streamWorkflowJob(job.job_id);
        if (!data) throw new Error("Workflow completed without a result");
        renderWorkflowResult(data);
        await updateTelemetry();

    } catch (err) {
        outputBox.innerHTML = `
            <div style="color: #f87171; padding: 20px;">
                <h3>Execution Error</h3>
                <p>${err.message}</p>
                <p>Ensure backend is running: <code>python -m server.app.main</code></p>
            </div>
        `;
        workflowStatus.textContent = "ERROR";
    } finally {
        btnRun.disabled = false;
    }
});

function streamWorkflowJob(jobId) {
    return new Promise((resolve, reject) => {
        const events = new EventSource(`${API_BASE}/workflow/jobs/${jobId}/events`);
        let result = null;

        events.onmessage = (message) => {
            const event = JSON.parse(message.data);
            renderProgressEvent(event);
            if (event.type === "job_completed") {
                result = event.workflow;
                events.close();
                resolve(result);
            } else if (event.type === "job_failed") {
                events.close();
                reject(new Error(event.error || "Workflow execution failed"));
            }
        };

        events.onerror = () => {
            events.close();
            reject(new Error("Lost connection to workflow progress stream"));
        };
    });
}

function renderProgressEvent(event) {
    if (event.type === "workflow_stage") {
        workflowStatus.textContent = event.stage.toUpperCase();
        planOverviewText.textContent = event.message;
    } else if (event.type === "plan_created") {
        workflowStatus.textContent = "EXECUTING";
        planOverviewText.textContent = event.plan_overview || "Plan created";
        subtasksList.innerHTML = (event.subtasks || []).map(task => `
            <div class="task-card" data-task-id="${escapeHtml(task.task_id)}">
                <div class="task-card-header">
                    <span class="agent-name">${escapeHtml(task.assigned_agent)}</span>
                    <span class="tier-pill">Queued</span>
                </div>
                <div class="task-desc">${escapeHtml(task.description)}</div>
                <div class="task-meta"><span>Waiting</span><span>...</span></div>
            </div>
        `).join("");
    } else if (event.type === "task_started") {
        currentAgentTag.textContent = event.assigned_agent;
        workflowStatus.textContent = `TASK ${event.task_index + 1}/${event.total_tasks}`;
        planOverviewText.textContent = event.description;
    } else if (event.type === "task_completed") {
        planOverviewText.textContent = `${event.assigned_agent} completed: ${event.routed_tier}`;
        const card = document.querySelector(`[data-task-id="${event.task_id}"]`);
        if (card && event.routing_audit) {
            card.querySelector(".task-meta").insertAdjacentHTML("beforeend", renderRoutingAudit(event.routing_audit));
        }
    } else if (event.type === "workflow_finished") {
        workflowStatus.textContent = event.status.toUpperCase();
        currentAgentTag.textContent = "Reviewer";
    }
}

function renderWorkflowResult(data) {
    workflowStatus.textContent = (data.status || "COMPLETED").toUpperCase();
    currentAgentTag.textContent = "Reviewer";
    planOverviewText.textContent = data.plan_overview || "Execution completed";

    // Render Subtasks in Left Sidebar
    subtasksList.innerHTML = "";
    if (data.subtasks && data.subtasks.length > 0) {
        data.subtasks.forEach((task) => {
            const card = document.createElement("div");
            card.className = "task-card";
            card.dataset.taskId = task.task_id;
            card.innerHTML = `
                <div class="task-card-header">
                    <span class="agent-name">${escapeHtml(task.assigned_agent)}</span>
                    <span class="tier-pill">Score: ${task.complexity_score !== null ? task.complexity_score.toFixed(2) : '0.50'}</span>
                </div>
                <div class="task-desc">${escapeHtml(task.description)}</div>
                <div class="task-meta">
                    <span>${escapeHtml(task.routed_tier || 'Tier 2')}</span>
                    <span style="color: #34d399; font-weight: 600;">✓ Done</span>
                </div>
                ${renderRoutingAudit(task.routing_audit)}
            `;
            subtasksList.appendChild(card);
        });
    }

    // Clean any accidental JSON wrapping from raw text
    let cleanText = data.final_output || "No output generated.";
    cleanText = stripJsonArtifacts(cleanText);

    // Render Clean Formatted Markdown in Output Box
    const renderedHtml = parseMarkdown(cleanText);

    outputBox.innerHTML = `
        <div class="result-header">
            <h2>Synthesis & Direct Answer</h2>
            <span style="color: #34d399; font-size: 13px; font-weight: 600;">Quality Confidence Score: ${data.confidence_score}</span>
        </div>
        <div class="markdown-content">
            ${renderedHtml}
        </div>
    `;
}

function renderRoutingAudit(audit) {
    if (!audit || !audit.selected) return "";
    const complexity = audit.complexity || {};
    const selected = audit.selected;
    const alternatives = (audit.alternatives || [])
        .filter(option => option.option !== selected.provider)
        .slice(0, 3)
        .map(option => `<li>${escapeHtml(option.option)}: $${option.estimated_cost_usd.toFixed(6)} / confidence ${(option.estimated_confidence * 100).toFixed(0)}%</li>`)
        .join("");
    return `
        <details class="routing-audit">
            <summary><span class="audit-label">WHY THIS MODEL</span> Score ${(complexity.score ?? 0).toFixed(2)} · Saved $${(audit.estimated_savings_vs_gpt4_class_usd || 0).toFixed(6)}</summary>
            <div class="audit-content">
                <p>${escapeHtml(audit.reason || "Cost-aware routing decision")}</p>
                <p><strong>Selected:</strong> ${escapeHtml(selected.provider)} / ${escapeHtml(selected.model)} · estimated $${selected.estimated_cost_usd.toFixed(6)} · confidence ${(selected.estimated_confidence * 100).toFixed(0)}%</p>
                <p><strong>GPT-4-class comparison:</strong> $${(audit.estimated_gpt4_class_cost_usd || 0).toFixed(6)}</p>
                <strong>Alternatives:</strong><ul>${alternatives}</ul>
                <small>${escapeHtml(audit.cost_basis || "Estimated cost comparison")}</small>
            </div>
        </details>
    `;
}

// Helper to remove any leftover JSON wrapping from over-structured responses
function stripJsonArtifacts(text) {
    if (typeof text !== "string") return String(text);
    
    // Check if it's a JSON string
    const trimmed = text.trim();
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || trimmed.includes("```json")) {
        try {
            let jsonStr = trimmed;
            if (jsonStr.includes("```json")) {
                jsonStr = jsonStr.split("```json")[1].split("```")[0].trim();
            } else if (jsonStr.startsWith("```")) {
                jsonStr = jsonStr.split("```")[1].split("```")[0].trim();
            }
            const obj = JSON.parse(jsonStr);
            return obj.final_synthesis || obj.answer || obj.response || obj.result || text;
        } catch (e) {
            // Regex fallback to extract value of final_synthesis
            const match = trimmed.match(/"final_synthesis"\s*:\s*"((?:[^"\\]|\\.)*)"/s);
            if (match && match[1]) {
                return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
            }
        }
    }
    return text;
}

// Built-in Lightweight Markdown Parser (100% Offline)
function parseMarkdown(md) {
    if (!md) return "";

    // Escape HTML first
    let html = escapeHtml(md);

    // Code blocks with syntax container
    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Bold & Italics
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Horizontal rules
    html = html.replace(/^---$/gim, '<hr>');

    // Unordered lists
    html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gms, '<ul>$1</ul>');
    // Clean nested duplicated ul tags
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Paragraphs (lines separated by double newlines)
    const paragraphs = html.split(/\n\n+/);
    html = paragraphs.map(p => {
        p = p.trim();
        if (!p) return "";
        if (p.startsWith("<h") || p.startsWith("<pre") || p.startsWith("<ul") || p.startsWith("<ol") || p.startsWith("<blockquote") || p.startsWith("<hr")) {
            return p;
        }
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join("\n");

    return html;
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Initial Polling
setInterval(updateHealthAndProviders, 4000);
setInterval(updateTelemetry, 4000);
updateHealthAndProviders();
updateTelemetry();
