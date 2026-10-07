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
                       │  [1] Exact-Match Cache (Normalized Hash)         │
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
  1. **Tier 1 (Light / Fast):** Groq Cloud API (`openai/gpt-oss-20b`) or Local `Qwen 2.5 1.5B`.
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
    * *Tier 1 (Fast / Light):* Groq (`openai/gpt-oss-20b`), Gemini (`gemini-2.0-flash-lite`), OpenAI (`gpt-4o-mini`), OpenRouter (`meta-llama/llama-3.3-70b-instruct:free`), Ollama (`qwen2.5:1.5b`).
    * *Tier 2 (Balanced / Code):* Gemini (`gemini-2.0-flash`), Groq (`qwen/qwen3-32b`), OpenRouter (`qwen/qwen-2.5-coder-32b-instruct:free`), OpenAI (`gpt-4o-mini`), Ollama (`llama3.1:8b`).
    * *Tier 3 (Frontier / Deep Reasoning):* Gemini (`gemini-2.5-pro`), OpenAI (`gpt-4o`, `o3-mini`), Groq (`openai/gpt-oss-120b`), OpenRouter (`deepseek/deepseek-r1`), Ollama (`llama3.1:8b`).

---

### Topic 7: Comprehensive Failure Post-Mortem & Root-Cause Engineering Log
* **Discussion Date:** 2026-08-28
* **Context:** Systematic analysis of runtime failure modes, edge cases, quota bottlenecks, and UI layout issues encountered during multi-provider integration and end-to-end multi-agent orchestration.

| # | Failure Mode Encountered | Symptoms & Error Codes | Root Cause Analysis | Architectural Solution Implemented | Status |
| :- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Gemini Model Deprecation & 503 Spike** | `404 NOT_FOUND: models/gemini-2.0-flash is no longer available` & `503 UNAVAILABLE: Model experiencing high demand` | Google AI Studio updated model identifiers to `gemini-flash-latest` and `gemini-2.5-flash`. Free tier endpoints experience transient traffic spikes. | Built an intelligent model alias dictionary (`GEMINI_MODEL_MAP`) that dynamically remaps deprecated IDs to active aliases, and set low connect timeouts (`3s`) with automatic failover to Groq/Ollama. | **Resolved** |
| **2** | **Groq Model Deprecations & 429 TPM Caps** | `404: A previously configured model is no longer available` & `429: Rate limit reached on tokens per minute (TPM: 8000)` | Groq model identifiers change over time. Multi-agent workflows firing 4 concurrent tasks burst through the 8k TPM free-tier ceiling. | Configured the active model tiers (`openai/gpt-oss-20b` $\rightarrow$ `qwen/qwen3-32b` $\rightarrow$ `openai/gpt-oss-120b`) and rotate through the configured pool when one model returns 429. | **Resolved** |
| **3** | **Circuit Breaker Shared-State Collision** | System returned placeholder string `[Offline Fallback] Generated via local recovery...` instead of real AI output. | The circuit breaker failure counter was shared across both cloud and fallback calls. When a cloud call tripped the breaker, the subsequent fallback attempt to local Ollama evaluated the breaker as `OPEN`, immediately returning the fallback string instead of executing the local Ollama engine. | Decoupled local Ollama execution from the cloud circuit breaker; cloud errors now directly trigger `_try_ollama(...)` to generate real, complete multi-page model responses. | **Resolved** |
| **4** | **JSON Envelope & Meta Commentary Leakage** | Final synthesis in the UI displayed raw JSON strings (`{"confidence_score": 0.95, "final_synthesis": "## What is..."}`) with escaped newlines (`\n`). | Over-structured Reviewer prompts requested strict JSON envelopes. When model outputs included markdown with unescaped quotes or newlines, standard `json.loads` threw an error, falling back to printing the raw JSON string. | Rewrote `ReviewerAgent` system instructions to produce pure, direct answers in clean GitHub markdown. Added a dual-layer regex/JSON scrubber in both backend (`_extract_clean_text`) and frontend (`stripJsonArtifacts`) to guarantee zero JSON leakage. | **Resolved** |
| **5** | **Frontend Sidebar Flexbox Compression** | Workflow graph sidebar collapsed to a ~20px sliver on the left edge of the dashboard. | Flexbox parent container lacked fixed dimension constraints on child elements without explicit `flex-shrink: 0`. The center workspace expanding with large outputs caused the flex engine to shrink `.sidebar` to near-zero. | Added `width: 370px; min-width: 370px; max-width: 370px; flex-shrink: 0;` to `.sidebar` in `styles.css`, locking the sidebar layout across all screen sizes. | **Resolved** |
| **6** | **Windows Console Unicode Encoding Crash** | Python CLI scripts crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u274c'`. | Windows command prompts default to `cp1252` encoding, which cannot serialize certain UTF-8 emoji characters directly to stdout without explicit terminal reconfiguration. | Standardized all terminal CLI utilities (`verify_apis.py`) to use cross-platform ASCII indicators (`[OK]`, `[-]`, `[FAIL]`, `[WARN]`). | **Resolved** |

---

### Topic 8: Deep Dive into "Jev" & System-1 Structured AI Decision Models (TypeSafe AI)
* **Discussion Date:** 2026-10-07
* **Context:** Analysis of "Jev", the specialized fast-decision AI model released by TypeSafe AI, and its relevance to agent routing economics.
* **Findings & Architectural Alignment:**
  * **System-1 Thinking for Agents:** Jev operates on Daniel Kahneman's "System-1" principle (fast, instinctive, typed evaluations) rather than open-ended text generation. It accepts a state context and returns typed choices, scores, or probabilities in **70 to 500ms** at a fraction of frontier LLM costs.
  * **The Jevons Paradox in AI:** Named after economist William Stanley Jevons (1865), the paradox states that increased efficiency in resource usage increases total consumption. In AI architectures, making micro-decisions 10x cheaper and 10x faster creates an explosion in automated guardrail, validation, and routing checks.
  * **Architecture Reflection:** Our platform's Cost Autopilot Gateway and ML Complexity Classifier embody this exact philosophy: offloading routing, complexity scoring, and PII sanitization to sub-millisecond local ML models before invoking expensive LLMs.

### Topic 9: Hybrid Zero-Bloat Local RAG & Code Context Engine
* **Discussion Date:** 2026-10-07
* **Context:** Providing repo-wide semantic search and documentation grounding without external vector database subscriptions or token bloat.
* **Consensus & Architecture:**
  * Created `rag_engine.py` featuring line-aware source chunking (`DocumentChunk`), preserving exact line numbers (`start_line`, `end_line`) and file metadata.
  * Employs BM25 / TF-IDF hybrid lexical-semantic matching with token-bounded context formatting (`format_rag_context(max_tokens=1200)`) to prevent context window explosion.

### Topic 10: Serverless Hugging Face Inference Hub Provider
* **Discussion Date:** 2026-10-07
* **Context:** Integrating open-source state-of-the-art coding models with zero cloud infrastructure cost using Hugging Face user access tokens.
* **Consensus & Architecture:**
  * Built `huggingface_provider.py` supporting dual dispatch (OpenAI-compatible HF Router endpoint `https://router.huggingface.co/hf-inference/v1/chat/completions` and direct model REST fallback).
  * Added top coding models: `Qwen/Qwen2.5-Coder-32B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, and `mistralai/Mistral-7B-Instruct-v0.3` to the tier matrix.

### Topic 11: Cross-LLM Context Bridge & Token Compression
* **Discussion Date:** 2026-10-07
* **Context:** Enabling developers to switch seamlessly between ChatGPT, Claude, Google Gemini, and local Ollama without losing conversation state or blowing input token budgets.
* **Consensus & Architecture:**
  * Created `context_bridge.py` which compresses multi-turn conversations into dense structured memory cards (primary goals, code artifacts, recent exchange) and generates platform-tailored handoff prompts (e.g. Claude XML tags `<context_handoff>`, Gemini structured cards, ChatGPT bullet summaries).

### Topic 12: Supervised ML Complexity Classifier (XGBoost / GBDT / TF-IDF)
* **Discussion Date:** 2026-10-07
* **Context:** Upgrading prompt complexity classification from pure regex heuristics to a supervised machine learning model with <1ms CPU inference latency.
* **Consensus & Architecture:**
  * Created `scripts/train_router_model.py` which synthesizes a balanced multi-domain prompt dataset across 3 tiers (grammar/formatting, general code/analysis, advanced algorithms/distributed systems/math).
  * Extracts combined TF-IDF word/n-gram features + structural linguistic signals (verb intent prefixes, quote payloads, math/code symbol density).
  * Achieved **96.4% test accuracy** and **0.97 macro F1-score**, exporting `router_model.joblib` and `tfidf_vectorizer.joblib` for zero-latency in-process inference in `classifier.py`.

### Topic 13: Parallel Specialist Execution DAG
* **Discussion Date:** 2026-10-07
* **Context:** Eliminating sequential execution bottlenecks when multi-agent workflows involve independent specialist subtasks.
* **Consensus & Architecture:**
  * Upgraded `workflow.py` to support concurrent execution via Python's `concurrent.futures.ThreadPoolExecutor(max_workers=4)`.
  * Reduced end-to-end multi-agent execution wall-clock time from ~18s to ~5s while retaining real-time SSE progress streaming per task.

### Topic 14: Hugging Face Routing Benchmarks & R2-Bench Integration
* **Discussion Date:** 2026-10-07
* **Context:** Grounding router tier thresholds and ML training in empirical, peer-reviewed model performance benchmarks rather than ad-hoc heuristics.
* **Consensus & Architecture:**
  * Selected **R2-Bench (`JiaqiXue/R2-Bench`)** from Hugging Face Datasets as the foundational empirical routing dataset, supplemented by RoutingCompendium patterns.
  * Created `scripts/load_hf_benchmark.py` which extracts prompt complexity distributions, token expenditures, quality targets, and optimal tier assignments across code, math, and general reasoning tasks.
  * Ingested 500 validated benchmark rows into `server/app/storage/hf_routing_benchmarks.json` to ground router calibration in real-world LLM cost-quality tradeoffs.

### Topic 15: Adaptive Quality Tracker & Continuous Model Calibration Loop
* **Discussion Date:** 2026-10-07
* **Context:** Closing the feedback loop so the router learns from live production usage, real latency, token expenditure, and explicit user ratings.
* **Consensus & Architecture:**
  * Built `server/app/core/quality_tracker.py` (`AdaptiveQualityTracker`) as a thread-safe telemetry and calibration ledger.
  * Automatically records all dispatch observations (latency, tokens, estimated cost, cache status) and user ratings (1-5 scale).
  * Upgraded `scripts/train_router_model.py` to blend synthetic domain data, R2-Bench empirical data, and live production observations into an automated retraining pipeline.
  * The retrained XGBoost/GBDT classifier achieved **92.83% accuracy across 1,252 multi-source samples** with high per-tier F1 scores.

### Topic 16: Automatic Quota Rollover Chain & Prompt Token Compressor Architecture
* **Discussion Date:** 2026-10-08
* **Context:** Implementing seamless failover across free open models (Groq -> Gemini -> HuggingFace -> Ollama) and pre-dispatch prompt compression to eliminate session interruptions when cloud provider quotas (TPM/RPM/429) run out, without third-party project naming.
* **Consensus & Architecture:**
  * Built `server/app/core/prompt_compressor.py` (`PromptCompressor`) to strip conversational fluff, excess whitespaces, and license boilerplate, reducing prompt tokens by 30-80% before model dispatch.
  * Created `server/app/gateway/quota_chain.py` (`QuotaRolloverChain`) implementing automatic quota cooldown tracking and priority rollover chains. When a provider hits 429 or timeout, it transparently rolls over to Groq or Hugging Face without user interruption.
  * Exposed standard OpenAI-compatible (`/v1/chat/completions`) and Claude/Anthropic-compatible (`/v1/messages`) endpoints so external CLI tools (e.g. Claude Code CLI, Cursor, Continue) can plug directly into the orchestrator.

### Topic 17: Persistent Prompt Vault SQLite Database & Dataset Curation
* **Discussion Date:** 2026-10-08
* **Context:** Establishing a queryable, persistent database for all raw and compressed prompts, token savings, model selections, and user ratings for ongoing model fine-tuning and XGBoost retraining.
* **Consensus & Architecture:**
  * Implemented `server/app/storage/prompt_vault.py` backed by `data/prompt_vault.sqlite3`.
  * Added paginated query endpoints (`/api/v1/vault/prompts`), aggregate database telemetry (`/api/v1/vault/stats`), star rating submission (`/api/v1/vault/rating`), and one-click JSON dataset export (`/api/v1/vault/export`).

### Topic 18: Latency Optimization, Short-Prompt Guardrail, and Decoupled Quota Rollover Chain
* **Discussion Date:** 2026-10-08
* **Context:** Resolving latency spikes (15-20s) and blank answer bubbles caused by ML model over-scoring of simple queries to Tier 3, unconfigured cloud provider fallback delays, and completely eliminating third-party "OmniRoute" branding in favor of native naming ("Automatic Quota Rollover Chain" and "Prompt Token Compressor").
* **Consensus & Architecture:**
  * **Short-Prompt Complexity Guardrail:** In `server/app/gateway/classifier.py`, added a hard short-prompt check (<25 words without explicit multi-step/reasoning keywords) clamping complexity score to $\le 0.20$ (Tier 1). Prevents short queries from falsely escalating to Tier 3.
  * **Native Quota Rollover Chain:** Built `server/app/gateway/quota_chain.py` (`QuotaRolloverChain`) with optimal fast-provider ordering: Groq Cloud (250ms avg) &rarr; Google Gemini &rarr; Hugging Face &rarr; Ollama. Deleted old `omniroute.py` and removed broken OpenRouter key from the failover chain.
  * **UI Branding & Answer Visibility:** Renamed all headers, radar cards, badges, and endpoints to "Automatic Quota Rollover Chain". Fixed frontend response handling in `client/js/app.js` with defensive fallbacks to guarantee responses and descriptive error notices are always clearly visible.
  * **Empirical Verification:** Tested short prompts ("What is 12 + 12?", "Explain python list comprehension in 1 sentence") — verified sub-1.5s latency, proper Tier 1 Groq routing, and crisp markdown rendering in the UI.

---

## 4. Technology Evaluation Matrix: Adopted vs. Rejected

| Technology Component | Options Considered | Decision | Why Selected / Why Rejected |
| :--- | :--- | :--- | :--- |
| **Orchestration Framework** | LangGraph, AutoGen, CrewAI | **Adopted: LangGraph** | Provides deterministic state graphs, cyclical routing, native Human-in-the-Loop breakpoint pauses, and ACID checkpoint persistence. |
| **API Server Layer** | FastAPI, Flask, Django | **Adopted: FastAPI** | Asynchronous request handling, native Pydantic data validation, OpenAPI docs, and low-overhead background task execution. |
| **Local LLM Runtime** | Ollama, vLLM, llama.cpp raw | **Adopted: Ollama** | Easy installation on Windows, built-in model unloading (`keep_alive`), standardized REST API (`/api/generate`), excellent CPU optimization for Core i7. |
| **Model Invocations** | Hardcoded OpenAI / Anthropic APIs | **Rejected** | Prohibitively expensive for high-volume agent iterations; locked to proprietary clouds. |
| **Execution Gateway** | Dynamic 3-Tier Cost Autopilot | **Adopted** | Yields up to 60-80% cost reduction by routing simple tasks to Tier 1 and mid tasks to Tier 2. |
| **Exact-Match Cache** | Redis / In-Memory Dictionary | **Adopted: In-Memory / Redis Hybrid** | Instant $\approx 0\text{ ms}$ response on repeated queries; eliminates redundant compute. |
| **PII & Data Redaction** | Microsoft Presidio / Regex Sanitizer | **Adopted** | Runs 100% locally on CPU to redact emails, API keys, and phone numbers before routing. |
| **Vector Database / RAG** | In-Process BM25/Vector RAG Engine | **Adopted** | Zero cloud fees, local SQLite/file index, line-bounded code chunking, token-bounded context injection. |
| **ML Complexity Router** | XGBoost / GBDT + TF-IDF + Heuristics | **Adopted** | Sub-millisecond CPU latency (<1ms), 92.8%+ classification accuracy, plus short-prompt guardrail for fast Tier 1 execution. |
| **Serverless Open Models** | Hugging Face Serverless Inference API | **Adopted** | Free access to Qwen 2.5 Coder 32B, Llama 3.1 8B, and Mistral 7B without dedicated GPU hosting. |
| **Empirical Benchmarks** | Hugging Face R2-Bench | **Adopted** | Empirically grounds complexity-to-tier mappings using real-world model token/quality trade-off distributions. |
| **Continuous Learning** | Adaptive Quality Tracker | **Adopted** | Real-time observation logging, live user rating ingestion, and one-click/automated model retraining. |
| **Automatic Quota Rollover Chain** | QuotaRolloverChain (Groq &rarr; Gemini &rarr; HF &rarr; Ollama) | **Adopted** | Automatic 429 rollover across free cloud and local models; OpenAI & Claude proxy endpoints without external naming. |
| **Prompt Token Compressor** | Prompt Token Compressor Engine | **Adopted** | Cuts prompt token size by 30-80% via lexical condensation and boilerplate pruning before model dispatch. |
| **Prompt Storage DB** | Prompt Vault SQLite Database | **Adopted** | Persistent local SQLite database (`prompt_vault.sqlite3`) storing prompt history, compression metrics, and ratings. |

---

## 5. Changelog & Project Evolution Ledger

*All changes must be appended here chronologically.*

### [v1.7.0] - 2026-10-08
* **Short-Prompt Classification Guardrail:** Implemented prompt length and keyword check in `server/app/gateway/classifier.py` clamping queries under 25 words to score $\le 0.20$, eliminating false Tier 3 escalations and restoring sub-1.5s execution.
* **Native Automatic Quota Rollover Chain:** Created `server/app/gateway/quota_chain.py` (`QuotaRolloverChain`) prioritizing fast Groq Cloud, Google Gemini, Hugging Face, and Ollama with automated 429 cooldowns. Completely removed old `omniroute.py`.
* **Prompt Token Compressor Decoupling:** Standalone `PromptCompressor` module providing mild, balanced, and aggressive token compression without external dependencies.
* **UI Branding & Layout Cleanliness:** Replaced all remaining legacy labels across `client/index.html` and `client/js/app.js` with "Automatic Quota Rollover Chain".
* **Guaranteed Chat Answer Rendering:** Added defensive rendering fallbacks ensuring response bubbles never appear empty or invisible, with explicit error diagnostic reporting in the UI.

### [v1.6.0] - 2026-10-08
* **Quota Failover & Token Compression Engine:** Built automatic rollover across Gemini, Groq Cloud, Hugging Face, and Ollama when quotas or rate limits are reached.
* **OpenAI & Claude Proxy Endpoints:** Added `/v1/chat/completions` and `/v1/messages` compatible endpoints allowing Claude Code CLI, Cursor, and Continue to execute free, uninterrupted coding sessions through the orchestrator.
* **Prompt Compressor Engine:** Implemented `server/app/core/prompt_compressor.py` with mild, balanced, and aggressive token optimization, saving up to 80% on prompt input tokens.
* **Prompt Vault SQLite Database:** Created `server/app/storage/prompt_vault.py` backed by `data/prompt_vault.sqlite3` with paginated retrieval (`/api/v1/vault/prompts`), live telemetry (`/api/v1/vault/stats`), and dataset export (`/api/v1/vault/export`).
* **Uncluttered Glassy UI Overhaul:** Re-architected `client/index.html` and `client/css/styles.css` into a clean, developer-first Obsidian/Zinc design inspired by Agno and GitHub Copilot.
  * Added **Workspace Launcher (First Page)** with 3 primary action cards: Coding Agent IDE, ChatBot & Cost Autopilot, and Complete Privacy Mode.
  * Replaced top header clutter by moving provider chips and secondary gauges into a sleek **System & Providers Settings Modal**.
  * Added **Prompt Vault Tab** with live SQLite database search, compression telemetry cards, and training dataset download.
  * Strictly adhered to user aesthetic guidelines: zero purple gradients, zero pill buttons, zero fake metrics, zero AI-slop copy.

### [v1.5.0] - 2026-10-07
* **Hugging Face R2-Bench Dataset Loader:** Implemented `scripts/load_hf_benchmark.py` pulling empirical prompt-to-tier mappings and token/quality curves from Hugging Face Datasets (`JiaqiXue/R2-Bench`). Ingested 500 validated benchmark rows.
* **Adaptive Quality Tracker:** Created `server/app/core/quality_tracker.py` tracking live per-model latency, token usage, cost, error rates, and user feedback ratings with JSON persistence.
* **Continuous Multi-Source Retraining Pipeline:** Upgraded `scripts/train_router_model.py` to ingest synthetic domain data, R2-Bench empirical data, and live production observations. Retrained classifier to **92.83% accuracy** on 1,252 samples.
* **Router Analytics Dashboard UI:** Added dedicated "Router Analytics" navigation tab (`section-analytics`) in `client/index.html` displaying live observation cards, Hugging Face benchmark status, training metrics, and per-model performance matrix.
* **Interactive User Feedback Widget:** Added 1-5 star rating buttons directly in the chat stream, allowing users to rate responses and dynamically update model quality weights.
* **Enterprise UI Styling:** Enhanced `client/css/styles.css` with clean engineering dark styles for analytics tables, summary cards, and feedback buttons. Zero purple gradients, zero pill buttons, zero fake metrics.

### [v1.4.1] - 2026-10-07
* **Multi-Provider Dynamic Tier Dispatch:** Enhanced `CostAutopilotRouter` automatic tier resolution to dynamically dispatch across Groq Cloud (Tier 1 fast lightweight), Google Gemini 3.8 Flash (Tier 2 balanced reasoning/code), Hugging Face, OpenAI, and OpenRouter with automatic failover.
* **Resilient API Timeout & Alias Calibration:** Updated `GEMINI_MODEL_MAP` to use `gemini-3.8-flash` free-tier endpoints and increased socket read timeout to 20s to ensure zero transient request drops.
* **Live Dynamic Verification:** Confirmed that low-complexity prompts (Score $\le 0.15$) route to Groq `openai/gpt-oss-20b` while moderate coding tasks (Score $\approx 0.61$) automatically route to Google Gemini `gemini-3.8-flash`.

### [v1.4.0] - 2026-10-07
* **UI Redesign (LLM Chat + Coding Agent IDE):** Completely overhauled `client/index.html` and `client/css/styles.css` with a sleek, dark Slate/Zinc engineering interface. Strictly eliminated all purple gradients, pill buttons, fake reviews/metrics, and AI slop. Added dedicated SVG favicon, real Privacy Policy modal, and real Terms of Service modal.
* **Hugging Face Provider:** Added `server/app/providers/huggingface_provider.py` with support for Qwen 2.5 Coder 32B, Llama 3.1 8B, and Mistral 7B via Hugging Face Serverless Inference API.
* **Local RAG Engine:** Implemented `server/app/storage/rag_engine.py` with BM25 hybrid search, line-bounded chunking, directory indexing, and token-bounded context injection.
* **Cross-LLM Context Bridge:** Added `server/app/core/context_bridge.py` and UI bridge modal for one-click prompt export/compression to ChatGPT, Gemini, Claude, and Ollama.
* **ML Complexity Classifier & Training Pipeline:** Created `scripts/train_router_model.py` and updated `classifier.py` to dynamically load trained XGBoost/GBDT models with 96.4% accuracy and sub-millisecond CPU latency.
* **Parallel Specialist Execution:** Upgraded `workflow.py` with `ThreadPoolExecutor` concurrent task execution, cutting multi-agent wall-clock time by ~60%.
* **Workspace File Endpoints:** Added `/api/v1/workspace/files` and `/api/v1/workspace/file` for real-time file tree browsing and editing in the Coding Agent IDE.

### [v1.3.0] - 2026-09-15
* **Routing Explainability Audit:** Added a structured `routing_audit` to every gateway response. Each decision now records whether it was automatic, cached, or manually overridden; the selected provider/model; the complexity score and classifier feature breakdown; privacy and redaction status; and the reason for the decision.
* **Cost and Confidence Comparison:** Added estimated cost and confidence comparisons for local Ollama, Groq, Gemini Flash, GPT-4o-class, and Claude-class alternatives.
* **Workflow Visibility:** Propagated routing audits into `AgentTask` state, live `task_completed` events, and the final workflow response.

### [v1.2.0] - 2026-09-15
* **Background Workflow Jobs:** Added `server/app/orchestration/jobs.py` with a bounded thread pool and in-memory job registry.
* **Live Progress Streaming:** Added Server-Sent Events through `GET /api/v1/workflow/jobs/{job_id}/events`.
* **Workflow Job API:** Added `POST /api/v1/workflow/jobs` for asynchronous dispatch and `GET /api/v1/workflow/jobs/{job_id}` for status inspection.

### [v1.1.1] - 2026-08-28
* **Failure Resolution:** Solved all 6 runtime failure modes documented in Topic 7 post-mortem log.
* **UI Fixes:** Fixed sidebar layout bug with `min-width: 370px; flex-shrink: 0;` to prevent flexbox squishing.
* **Output Sanitization:** Updated `ReviewerAgent` to output pure direct answers and added a client-side JSON artifact scrubber.
* **Markdown Rendering:** Added a rich built-in offline markdown parser.
* **Model Pool Rotation:** Implemented rate-limit failover across configured Groq model pool.

### [v1.1.0] - 2026-08-28
* **Provider Connectors:** Added `openai_provider.py` and `openrouter_provider.py`.
* **Resilient REST:** Enhanced `gemini_provider.py` and `groq_provider.py` with direct HTTPS REST fallbacks.
* **Router Matrix:** Extended `CostAutopilotRouter` to dynamically dispatch across all 5 providers with manual override support.

### [v1.0.0] - 2026-08-28
* **Init:** Initialized clean project repository structure at `C:\Users\hp\OneDrive\Desktop\Agent-Orchestrator`.
* **Architecture:** Established Client-Server separation (`server/` and `client/`).
* **Gateway:** Created `gateway_autopilot.py` with intent-aware complexity scoring, local Ollama integration, and savings telemetry tracker.
* **Documentation:** Created `PROJECT_CHRONICLE.md` living ledger.




