# PROJECT CHRONICLE & ARCHITECTURE DECISION LOG (ADR)
**Project Name:** Agent Orchestration Platform with Intelligent LLM Cost Autopilot Gateway  
**Document Reference:** `PROJECT-CHRONICLE-2026-V1`  
**Status:** Living Document (Append-Only Ledger)  
**Standard Compliance:** IEEE Std 830-1998 / ISO/IEC/IEEE 29148  

---

## 1. Executive Summary & Vision

Modern multi-agent architectures (using frameworks like LangGraph, AutoGen, or CrewAI) often face a critical economic and computational bottleneck: **monolithic model invocation**. When every subtask—from simple regex parsing, grammar correction, and UI button creation to complex multi-file backend architecture and database schema design—is routed unconditionally to high-tier frontier models (e.g., Claude 3.5 Sonnet / Opus, GPT-4o), token costs explode and computational latency spikes.

The **Agent Orchestration Platform with Intelligent Cost Autopilot** solves this by establishing a decoupled **Execution Gateway Layer** beneath the Multi-Agent Orchestration State Machine. Every agent interaction is intercepted, sanitized, classified for semantic complexity, and dynamically routed across a **3-Tier Execution Matrix** combining local open-weight models (via Ollama) and generous free-tier cloud APIs (Google Gemini, Groq Cloud).

---

## 2. System Architecture & Component Interactions

```
                                      [ User / Web Dashboard ]
                                                 ▲
                                                 │ HTTP / WebSocket
                                                 ▼
                                     [ FastAPI Gateway Layer ]
                                                 ▲
                                                 │ State Dispatch
                                                 ▼
                       ┌──────────────────────────────────────────────────┐
                       │        LangGraph Multi-Agent State Machine       │
                       ├──────────────────────────────────────────────────┤
                       │  1. Memory Retrieval Node (ChromaDB / SQLite)   │
                       │  2. Supervisor Router (Task Decomposition DAG)   │
                       │  3. Specialist Nodes:                            │
                       │     ├── Research Agent                           │
                       │     ├── Analysis Agent                           │
                       │     ├── Code Agent                               │
                       │     ├── Writing Agent                            │
                       │     └── CV Specialist Agent (YOLOv8)             │
                       │  4. Quality Reviewer Node (Confidence Scoring)   │
                       │  5. HITL Escalation Node (<0.80 Confidence)      │
                       └─────────────────────────┬────────────────────────┘
                                                 │
                                                 │ All Agent LLM Calls
                                                 ▼
                       ┌──────────────────────────────────────────────────┐
                       │    Intelligent Routing Gateway (Cost Autopilot)   │
                       ├──────────────────────────────────────────────────┤
                       │  [0] Budget Guard (X-Max-Cost-Per-Request Cap)   │
                       │  [1] Semantic Cache (Exact Hash & Vector Sim)   │
                       │  [2] PII & Secret Redactor (Presidio / Regex)    │
                       │  [3] ML Complexity Classifier (XGBoost / Trees)  │
                       │      ├── Intent & Verb Extraction (POS)          │
                       │      ├── Instruction vs. Payload Splitting       │
                       │      └── Semantic Embeddings (<5ms CPU)          │
                       └─────────────────────────┬────────────────────────┘
                                                 │
                  ┌──────────────────────────────┼──────────────────────────────┐
                  ▼                              ▼                              ▼
          [ TIER 1: FAST/LIGHT ]        [ TIER 2: BALANCED ]          [ TIER 3: FRONTIER ]
          Score: 0.00 – 0.30            Score: 0.31 – 0.70            Score: 0.71 – 1.00
          • Local: Qwen 2.5 1.5B        • Local: Llama 3.1 8B (4.8GB) • Cloud: Gemini 2.5 Pro
          • Cloud: Groq Llama-3.1 8B    • Cloud: Gemini 2.0 Flash     • Cloud: DeepSeek-R1 / GPT-4o
                  │                              │                              │
                  └──────────────────────────────┼──────────────────────────────┘
                                                 │
                                                 ▼
                                   [ Circuit Breaker & Fallback ]
                                                 │
                                                 ▼
                                   [ Telemetry & Savings Engine ]
```

---

## 3. Foundational Research & Discussion Records

### Topic 1: The Multi-Agent Interception Architecture
* **Discussion Date:** 2026-08-28
* **Context:** Integration of the LLM Cost Autopilot Gateway into a LangGraph multi-agent structure.
* **Consensus:** Individual agent nodes (Supervisor, Research, Code, Writing, Reviewer) should never instantiate static cloud LLM objects. Instead, all agents route payloads through a centralized `gateway.route_and_execute(prompt, agent_name)` function. This enables dynamic runtime model swapping, token budgeting, and offline privacy enforcement without altering agent business logic.

### Topic 2: Hardware Constraints & RAM Sizing (Intel Core i7 + 16 GB RAM)
* **Discussion Date:** 2026-08-28
* **Context:** Assessing the feasibility of running local open-weight models on user hardware (Intel Core i7, 16 GB RAM, no dedicated high-end GPU).
* **Consensus & Findings:**
  * Quantized 4-bit `Llama-3.1-8B-Instruct (Q4_K_M)` or `Qwen-2.5-7B` requires **~4.8 GB of system RAM**.
  * Running *two concurrent 8B models* simultaneously would consume $\approx 9.6\text{ GB} + 4.5\text{ GB (OS)} = 14.1\text{ GB}$, risking RAM paging and stutter.
  * **Solution:** Pair a **Tier 1 lightweight model (`Qwen 2.5 1.5B`, ~1.2 GB RAM)** with a **Tier 2 balanced model (`Llama 3.1 8B`, ~4.8 GB RAM)**. Total peak RAM is $\approx 6.0\text{ GB} + 4.5\text{ GB} = 10.5\text{ GB}$, leaving over **5.5 GB of free RAM** for OS stability and vector storage.

### Topic 3: 100% Free-of-Cost Strategy
* **Discussion Date:** 2026-08-28
* **Context:** Strategy to build and run the entire platform at zero dollar cost without paid OpenAI/Anthropic subscriptions.
* **Adopted Stack:**
  1. **Tier 1 (Light / Fast):** Groq Cloud API (`llama-3.1-8b-instant`, 30 RPM free, 500+ tokens/sec) or Local `Qwen 2.5 1.5B`.
  2. **Tier 2 (Mid / Code / Reasoning):** Google AI Studio (`gemini-2.0-flash`, 15 RPM / 1M TPM free) or Local `Llama 3.1 8B`.
  3. **Tier 3 (Frontier / Judge / Architecture):** Google AI Studio (`gemini-2.5-pro` / `gemini-2.5-flash-thinking`) or Free OpenRouter/GitHub Models (`DeepSeek-R1`, `GPT-4o`).

### Topic 4: The "Intent vs. Payload" NLP Classification Challenge
* **Discussion Date:** 2026-08-28
* **Context:** Preventing false-positive complexity escalation when a simple text prompt contains complex technical words.
  * *Example A:* `"Write a complex code to train xgboost model with cross-validation"` $\rightarrow$ Coding generation (**Tier 3: Score 0.85**).
  * *Example B:* `"Correct the grammar in the sentence: Write a complex code to train xgboost model"` $\rightarrow$ Grammar editing (**Tier 1: Score 0.12**).
* **Technical Resolution:**
  1. **Instruction vs. Payload Splitting:** Parse quoted substrings and target strings as passive data payloads.
  2. **Root Action-Verb Extraction:** POS tag the governing verb (`Correct/Fix` vs `Write/Implement/Train`).
  3. **Semantic Anchors:** Evaluate cosine similarity against linguistic proofreading vs software engineering vector clusters.
  4. **Decision Tree Priority Branching:** In XGBoost, an active editing intent flag evaluates first, immediately classifying prompt complexity $\le 0.15$ regardless of technical keywords in the payload.

### Topic 5: 100% Offline Air-Gapped Scoping & Privacy Mode
* **Discussion Date:** 2026-08-28
* **Context:** Running the entire requirements gathering and code generation pipeline offline with zero internet connectivity.
* **Consensus:** Ollama runs completely offline once weights are pulled. When `PRIVACY_MODE=True`, all cloud API requests are intercepted by the Gateway and redirected to local Ollama models with zero outgoing network packets.

### Topic 6: Unified Multi-Provider Matrix & Free-Tier Model Roster
* **Discussion Date:** 2026-08-28
* **Context:** Expanding from local-only to a hybrid execution matrix supporting Google Gemini, OpenAI GPT, Groq Cloud, OpenRouter free models, and local Ollama.
* **Consensus & Architecture:**
  * **Zero-Dependency Resilient Connectors:** Every cloud provider (OpenAI, Gemini, Groq, OpenRouter) supports dual dispatch: official SDK when available, plus pure-Python `requests` REST fallbacks.
  * **Dynamic Tier Resolution:**
    * *Tier 1 (Fast / Light):* Groq (`llama-3.1-8b-instant`), Gemini (`gemini-2.0-flash-lite`), OpenAI (`gpt-4o-mini`), OpenRouter (`meta-llama/llama-3.3-70b-instruct:free`), Ollama (`qwen2.5:1.5b`).
    * *Tier 2 (Balanced / Code):* Gemini (`gemini-2.0-flash`), Groq (`llama-3.3-70b-versatile`), OpenRouter (`qwen/qwen-2.5-coder-32b-instruct:free`), OpenAI (`gpt-4o-mini`), Ollama (`llama3.1:8b`).
    * *Tier 3 (Frontier / Deep Reasoning):* Gemini (`gemini-2.5-pro`), OpenAI (`gpt-4o`, `o3-mini`), Groq (`deepseek-r1-distill-llama-70b`), OpenRouter (`deepseek/deepseek-r1:free`), Ollama (`llama3.1:8b`).

---

### Topic 7: Comprehensive Failure Post-Mortem & Root-Cause Engineering Log
* **Discussion Date:** 2026-08-28
* **Context:** Systematic analysis of runtime failure modes, edge cases, quota bottlenecks, and UI layout issues encountered during multi-provider integration and end-to-end multi-agent orchestration.

| # | Failure Mode Encountered | Symptoms & Error Codes | Root Cause Analysis | Architectural Solution Implemented | Status |
| :- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Gemini Model Deprecation & 503 Spike** | `404 NOT_FOUND: models/gemini-2.0-flash is no longer available` & `503 UNAVAILABLE: Model experiencing high demand` | Google AI Studio updated model identifiers to `gemini-flash-latest` and `gemini-2.5-flash`. Free tier endpoints experience transient traffic spikes. | Built an intelligent model alias dictionary (`GEMINI_MODEL_MAP`) that dynamically remaps deprecated IDs to active aliases, and set low connect timeouts (`3s`) with automatic failover to Groq/Ollama. | **Resolved** |
| **2** | **Groq Decommissioned IDs & 429 TPM Caps** | `404: The model llama-3.1-8b-instant does not exist` & `429: Rate limit reached on tokens per minute (TPM: 8000)` | Groq decommissioned older Llama 3.1 endpoints in favor of `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`. Multi-agent workflows firing 4 concurrent tasks burst through the 8k TPM free-tier ceiling. | Implemented an active **Model Pool Rotation** mechanism in `groq_provider.py` (`openai/gpt-oss-20b` $\rightarrow$ `qwen/qwen3.8-27b` $\rightarrow$ `groq/compound-mini`). If one model returns 429, the provider instantly rotates to the next model pool with zero user interruption. | **Resolved** |
| **3** | **Circuit Breaker Shared-State Collision** | System returned placeholder string `[Offline Fallback] Generated via local recovery...` instead of real AI output. | The circuit breaker failure counter was shared across both cloud and fallback calls. When a cloud call tripped the breaker, the subsequent fallback attempt to local Ollama evaluated the breaker as `OPEN`, immediately returning the fallback string instead of executing the local Ollama engine. | Decoupled local Ollama execution from the cloud circuit breaker; cloud errors now directly trigger `_try_ollama(...)` to generate real, complete multi-page model responses. | **Resolved** |
| **4** | **JSON Envelope & Meta Commentary Leakage** | Final synthesis in the UI displayed raw JSON strings (`{"confidence_score": 0.95, "final_synthesis": "## What is..."}`) with escaped newlines (`\n`). | Over-structured Reviewer prompts requested strict JSON envelopes. When model outputs included markdown with unescaped quotes or newlines, standard `json.loads` threw an error, falling back to printing the raw JSON string. | Rewrote `ReviewerAgent` system instructions to produce pure, direct answers in clean GitHub markdown. Added a dual-layer regex/JSON scrubber in both backend (`_extract_clean_text`) and frontend (`stripJsonArtifacts`) to guarantee zero JSON leakage. | **Resolved** |
| **5** | **Frontend Sidebar Flexbox Compression** | Workflow graph sidebar collapsed to a ~20px sliver on the left edge of the dashboard. | Flexbox parent container lacked fixed dimension constraints on child elements without explicit `flex-shrink: 0`. The center workspace expanding with large outputs caused the flex engine to shrink `.sidebar` to near-zero. | Added `width: 370px; min-width: 370px; max-width: 370px; flex-shrink: 0;` to `.sidebar` in `styles.css`, locking the sidebar layout across all screen sizes. | **Resolved** |
| **6** | **Windows Console Unicode Encoding Crash** | Python CLI scripts crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u274c'`. | Windows command prompts default to `cp1252` encoding, which cannot serialize certain UTF-8 emoji characters directly to stdout without explicit terminal reconfiguration. | Standardized all terminal CLI utilities (`verify_apis.py`) to use cross-platform ASCII indicators (`[OK]`, `[-]`, `[FAIL]`, `[WARN]`). | **Resolved** |

---

## 4. Technology Evaluation Matrix: Adopted vs. Rejected

| Technology Component | Options Considered | Decision | Why Selected / Why Rejected |
| :--- | :--- | :--- | :--- |
| **Orchestration Framework** | LangGraph, AutoGen, CrewAI | **Adopted: LangGraph** | Provides deterministic state graphs, cyclical routing, native Human-in-the-Loop breakpoint pauses, and ACID checkpoint persistence. |
| **API Server Layer** | FastAPI, Flask, Django | **Adopted: FastAPI** | Asynchronous request handling, native Pydantic data validation, OpenAPI docs, and low-overhead background task execution. |
| **Local LLM Runtime** | Ollama, vLLM, llama.cpp raw | **Adopted: Ollama** | Easy installation on Windows, built-in model unloading (`keep_alive`), standardized REST API (`/api/generate`), excellent CPU optimization for Core i7. |
| **Model Invocations** | Hardcoded OpenAI / Anthropic APIs | **Rejected** | Prohibitively expensive for high-volume agent iterations; locked to proprietary clouds. |
| **Execution Gateway** | Dynamic 3-Tier Cost Autopilot | **Adopted** | Yields up to 60-80% cost reduction by routing simple tasks to Tier 1 and mid tasks to Tier 2. |
| **Semantic Cache** | Redis / In-Memory Dictionary | **Adopted: In-Memory / Redis Hybrid** | Instant $\approx 0\text{ ms}$ response on repeated queries; eliminates redundant compute. |
| **PII & Data Redaction** | Microsoft Presidio / Regex Sanitizer | **Adopted** | Runs 100% locally on CPU to redact emails, API keys, and phone numbers before routing. |
| **Vector Database** | ChromaDB (Embedded), Pinecone, Weaviate | **Adopted: ChromaDB (Local SQLite/DuckDB)** | Free, local, zero-setup, in-process vector search without cloud fees. |

---

## 5. Changelog & Project Evolution Ledger

*All changes must be appended here chronologically.*

### [v1.1.1] - 2026-08-28
* **Failure Resolution:** Solved all 6 runtime failure modes documented in Topic 7 post-mortem log.
* **UI Fixes:** Fixed sidebar layout bug with `min-width: 370px; flex-shrink: 0;` to prevent flexbox squishing.
* **Output Sanitization:** Updated `ReviewerAgent` to output pure direct answers and added a client-side JSON artifact scrubber.
* **Markdown Rendering:** Added a rich built-in offline markdown parser rendering headings, syntax code blocks, blockquotes, and lists cleanly.
* **Model Pool Rotation:** Implemented rate-limit failover across Groq model pool (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `groq/compound-mini`).

### [v1.1.0] - 2026-08-28
* **Provider Connectors:** Added `openai_provider.py` (GPT-4o, GPT-4o-mini, o3-mini) and `openrouter_provider.py` (100% Free models e.g., DeepSeek-R1:free, Llama-3.3-70b:free).
* **Resilient REST:** Enhanced `gemini_provider.py` and `groq_provider.py` with zero-dependency direct HTTPS REST fallbacks.
* **Router Matrix:** Extended `CostAutopilotRouter` to dynamically dispatch across all 5 providers with manual override support.
* **API Endpoints:** Added `/api/v1/models` and enriched `/api/v1/health` with live provider connectivity reporting.
* **Client UI:** Added live Provider status badges (Gemini, Groq, OpenRouter, OpenAI, Ollama) and an execution mode dropdown selector in `index.html` and `app.js`.

### [v1.0.0] - 2026-08-28
* **Init:** Initialized clean project repository structure at `C:\Users\hp\OneDrive\Desktop\Agent-Orchestrator`.
* **Architecture:** Established Client-Server separation (`server/` and `client/`).
* **Gateway:** Created `gateway_autopilot.py` with intent-aware complexity scoring, local Ollama integration, and savings telemetry tracker.
* **Documentation:** Created this `PROJECT_CHRONICLE.md` living ledger to preserve full conversation context, architecture designs, and rationale.


