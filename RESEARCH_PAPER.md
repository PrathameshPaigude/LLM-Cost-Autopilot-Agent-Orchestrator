# Agent Orchestration Platform with an Intelligent LLM Cost Autopilot Gateway

## Abstract

Large language model (LLM) applications frequently route every request to the same expensive or computationally heavy model, even when many requests are simple editing, formatting, or factual tasks. This paper presents an implemented agent orchestration platform that introduces a cost-aware execution gateway between multi-agent workflow logic and model providers. The gateway sanitizes prompts, detects repeated requests, estimates semantic complexity with an interpretable rule-based classifier, selects an execution tier, supports local and cloud providers, and records routing and savings telemetry. The platform also provides a stateful specialist workflow, background job execution, Server-Sent Events (SSE) progress updates, reviewer-based confidence assessment, human-in-the-loop escalation, and a tamper-evident SQLite event ledger. The implementation is designed to support zero-cloud-cost and air-gapped operation through Ollama while retaining optional access to Gemini, Groq, OpenAI, and OpenRouter. This manuscript documents the system design, implementation, feature set, interfaces, verification scope, limitations, and planned extensions.

**Keywords:** large language models, multi-agent systems, model routing, cost optimization, local inference, privacy-preserving AI, FastAPI, workflow orchestration

## 1. Introduction

LLM-based software increasingly combines several model calls to plan a task, generate intermediate artifacts, inspect those artifacts, and produce a final answer. A uniform model policy creates two operational problems. First, simple requests consume more model capacity than necessary. Second, applications become dependent on a single provider, network connection, pricing model, and availability profile.

The project described in this paper addresses these problems through a centralized LLM Cost Autopilot Gateway. Agent implementations submit prompts to the gateway rather than directly constructing provider-specific clients. The gateway makes an execution decision using request mode, privacy constraints, user overrides, cache state, prompt features, and provider availability. This separation allows the orchestration layer to remain focused on task decomposition and synthesis while the gateway manages execution policy.

The project has two related goals:

1. Reduce unnecessary cloud usage and enable local execution for requests that do not require frontier-scale models.
2. Provide an observable and resilient multi-agent development platform that can continue operating across provider failures, rate limits, and offline environments.

This paper describes the system as implemented in the repository at the time of writing. It distinguishes delivered functionality from future architecture ideas recorded in historical project notes.

## 2. Problem Definition and Objectives

The system is designed for workflows in which a user submits a broad goal and multiple specialized agents contribute to a final response. The central problem is to choose an execution path for each model request while preserving workflow correctness, privacy, and operational visibility.

The implementation objectives are:

- classify prompts into interpretable complexity tiers;
- prioritize editing intent over complex terminology contained in passive text;
- avoid repeated generation through normalized exact-response caching;
- remove common personal data and secrets before provider dispatch;
- support automatic, offline, cloud, and manual-provider execution modes;
- provide local execution through Ollama;
- provide optional cloud execution through Gemini, Groq, OpenAI, and OpenRouter;
- keep provider failures from terminating the complete workflow when a fallback is available;
- coordinate supervisor, specialist, reviewer, and human-review stages;
- expose progress and routing explanations to the dashboard;
- collect estimated cost, token, latency, cache, energy, quality, and routing telemetry;
- preserve workflow events in a SQLite ledger with hash-chain verification.

## 3. System Scope

The implemented system contains a Python FastAPI backend, a browser-based client, provider adapters, an orchestration package, gateway components, a SQLite ledger, unit tests, and a benchmark driver. The current execution path is sequential within a workflow: the supervisor creates subtasks, specialists execute one after another, and the reviewer synthesizes the outputs.

The system supports two operational deployment profiles:

- **Hybrid profile:** automatic or cloud execution can use configured cloud providers and fall back to local Ollama where applicable.
- **Offline privacy profile:** privacy mode or an explicit offline request forces local Ollama execution and prevents cloud fallback.

The current implementation does not claim persistent distributed job storage, token-by-token streaming, vector similarity caching, production-grade distributed scheduling, or automatic use of every configured provider in the automatic route.

## 4. System Architecture

### 4.1 High-Level Data Flow

The request flow is:

1. A user submits a direct prompt or a complete workflow request through the dashboard or HTTP API.
2. FastAPI validates the request and forwards it to the gateway or workflow engine.
3. The gateway sanitizes the prompt and calculates a normalized SHA-256 cache key.
4. A cache hit returns the previous response without a new model call.
5. On a cache miss, the classifier extracts instruction, domain, length, and agent features.
6. The router resolves the execution mode and selected tier.
7. The selected provider is called through its provider adapter.
8. Provider failures invoke the applicable fallback path, subject to privacy and mode constraints.
9. The gateway records a routing audit, telemetry observation, and optional workflow ledger events.
10. The workflow engine stores the specialist result and continues to the next task.
11. The reviewer synthesizes the collected outputs and calculates a confidence score.
12. Low-confidence workflows enter the in-memory human-review queue.

### 4.2 Backend Components

The backend is organized into the following layers:

| Layer | Implemented responsibility |
|---|---|
| FastAPI application | HTTP endpoints, request models, health reporting, static client serving, and SSE responses |
| Orchestration | Supervisor planning, specialist dispatch, review, confidence evaluation, and human-review registration |
| Gateway | Prompt redaction, exact cache, complexity scoring, mode resolution, provider selection, fallback, and routing audit generation |
| Providers | Provider-specific REST or SDK integration for Ollama, Groq, Gemini, OpenAI, and OpenRouter |
| Core services | Configuration, telemetry, SQLite event ledger, and reusable circuit-breaker implementation |
| Client | Dashboard controls, provider status, workflow progress, telemetry, markdown rendering, and routing explanations |

## 5. Intelligent Cost Autopilot Gateway

### 5.1 Execution Modes

The gateway accepts three explicit execution modes:

- **Auto:** selects a route based on complexity, availability, and fallback policy.
- **Offline:** forces local Ollama execution and disables cloud fallback.
- **Cloud:** prefers configured cloud providers and uses cloud fallback behavior.

A global privacy setting and a per-request offline flag take precedence over normal automatic routing. Provider and model overrides are also accepted for controlled experiments and operator choice.

### 5.2 Complexity Classification

The current classifier is interpretable heuristic logic rather than a trained machine-learning model. It extracts the following features:

- editing intent;
- coding intent;
- architecture keywords;
- strong architecture keywords;
- algorithm keywords;
- advanced-domain keywords;
- quoted or backtick-delimited payload;
- prompt word count;
- agent-persona bias.

The classifier returns a score in the interval $[0,1]$ and assigns tiers using the configured boundaries:

- **Tier 1:** score below 0.30;
- **Tier 2:** score from 0.30 through 0.70;
- **Tier 3:** score above 0.70.

Editing intent is evaluated first. For example, a request to correct grammar in a sentence containing terms such as `xgboost` or `pipeline` is treated as an editing request rather than as a model-building request. Architecture and advanced-domain signals increase the score, while algorithmic and coding signals generally select the balanced tier.

The classifier can return a diagnostic explanation containing the score, tier, matched signals, feature values, instruction-versus-payload indicators, and contribution categories. This explanation is included in the routing audit when debug information is requested and is also surfaced by the dashboard.

### 5.3 Exact Cache

The gateway maintains an in-memory cache for completed responses. Cache keys are generated after redaction and normalization using:

- trimming;
- lower-casing;
- collapsing repeated whitespace;
- SHA-256 hashing.

A cache hit is represented as Tier 0 and includes the cache key, cache status, latency, and routing audit. Provider or model overrides bypass the cache so that explicit operator requests are honored. The current cache is exact-match and process-local; it is not a vector or persistent cache.

### 5.4 Redaction

The redaction component runs before hashing, classification, and provider dispatch. It detects and replaces common sensitive values, including:

- API-key-like strings;
- selected AWS and GitHub token patterns;
- quoted secrets;
- email addresses;
- telephone numbers;
- credit-card-like numbers.

The gateway returns a boolean redaction indicator and records a hash of the sanitized prompt in workflow ledger events. The original prompt is not sent through the provider path after sanitization.

### 5.5 Provider Selection and Fallback

The provider roster consists of:

- **Ollama:** local, offline-capable inference;
- **Groq:** cloud execution with a model-pool retry strategy;
- **Google Gemini:** REST-based generation with model aliases;
- **OpenAI:** optional SDK integration with REST fallback;
- **OpenRouter:** REST-based access to a multi-model roster.

Automatic cloud-oriented routing currently attempts Groq first and then Gemini, followed by local execution in automatic mode when cloud providers are unavailable. OpenAI and OpenRouter are available through explicit provider selection. Manual overrides select a provider and optionally a model regardless of the automatic tier choice.

Local routing selects a lightweight Ollama model for Tier 1 and a larger local model for Tier 2 and Tier 3. Timeouts are tier-dependent. In offline mode, a timeout produces an explicit error rather than silently transmitting the prompt to a cloud service.

### 5.6 Routing Audit

Each non-error route produces a structured audit containing:

- decision type: automatic, manual override, or cache hit;
- decision reason;
- complexity score and tier;
- extracted classifier features;
- selected provider and model;
- estimated selected cost;
- estimated confidence;
- alternative model-class cost and confidence estimates;
- estimated savings against a GPT-4-class baseline;
- privacy and redaction status;
- cost-estimation disclaimer.

The cost values are illustrative application estimates based on token approximations and configured reference rates. They are not provider billing records.

## 6. Multi-Agent Orchestration

### 6.1 Supervisor Planning

The supervisor receives the user goal and asks a model to produce a structured workflow plan. The plan contains a workflow identifier, an overview, and assigned subtasks. If model output cannot be parsed, the supervisor provides a heuristic fallback plan so that the workflow can continue.

### 6.2 Specialist Agents

The platform includes five specialist adapters:

- **Research Agent:** gathers and structures research-oriented material;
- **Analysis Agent:** performs systems and technical analysis;
- **Code Agent:** handles implementation and programming tasks;
- **Writing Agent:** produces structured technical or editorial content;
- **Computer Vision Specialist:** handles computer-vision-oriented tasks, including YOLO-related prompts.

Specialists are intentionally thin adapters. Their business prompts are routed through the shared gateway, which keeps provider policy, privacy handling, telemetry, and fallback behavior centralized.

### 6.3 Reviewer and Confidence Assessment

After specialist execution, the reviewer receives the accumulated task outputs and produces a final synthesis. The reviewer also extracts clean answer text from model output and calculates a heuristic confidence score. The workflow is marked for human review when confidence is below the configured threshold of 0.80.

### 6.4 Human-in-the-Loop Workflow

Low-confidence workflows are registered with an in-memory HITL manager. The backend exposes operations to inspect pending reviews and to approve, reject, or apply human edits. Approval and rejection actions are recorded in the event ledger. The current browser dashboard includes the review surface, but client-side approve/reject interaction remains incomplete and should be treated as an API-supported feature.

### 6.5 Background Jobs and Progress Events

Long-running workflows can be submitted as background jobs. A bounded thread pool executes up to four workflow jobs concurrently, while an in-memory job registry tracks status, timestamps, workflow state, and errors.

The job API publishes ordered lifecycle events, including:

- `job_queued`;
- `job_started`;
- planning stage;
- plan creation;
- specialist task start;
- specialist task completion;
- review stage;
- workflow completion;
- job failure.

The SSE endpoint streams these events to the dashboard and emits keep-alive comments during idle periods. This is workflow progress streaming, not token-level model streaming. Job state is lost when the backend process restarts.

## 7. Provider and Model Matrix

The dashboard and model endpoint expose a model roster organized by provider and tier. The supported categories are:

| Provider | Execution type | Primary role in the implementation |
|---|---|---|
| Ollama | Local and offline | Privacy mode, local Tier 1, Tier 2, and Tier 3 execution |
| Groq | Cloud | Fast automatic cloud route and model-pool retry behavior |
| Google Gemini | Cloud | Automatic cloud fallback and explicit provider selection |
| OpenAI | Cloud | Explicit commercial-provider override with SDK/REST support |
| OpenRouter | Cloud | Explicit access to free and multi-model routes |

Model identifiers are configurable through environment settings and request overrides. Provider availability and configuration are reported by the health endpoint.

## 8. API Surface

The primary HTTP interface includes:

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Reports application status, privacy mode, provider configuration, and Ollama availability |
| `GET /api/v1/models` | Returns the provider and model roster |
| `POST /api/v1/route` | Executes one prompt through the gateway |
| `GET /api/v1/debug/cache` | Inspects cache rules and optionally tests a prompt hit |
| `DELETE /api/v1/debug/cache` | Clears the process-local cache |
| `POST /api/v1/workflow/run` | Runs a complete workflow synchronously |
| `POST /api/v1/workflow/jobs` | Starts a background workflow and returns a job identifier |
| `GET /api/v1/workflow/jobs/{job_id}` | Reads background job status |
| `GET /api/v1/workflow/jobs/{job_id}/events` | Streams job lifecycle events through SSE |
| `GET /api/v1/telemetry` | Returns aggregate routing, token, cost, cache, energy, and quality telemetry |
| `GET /api/v1/telemetry/pareto` | Returns recent routing and reviewed-quality observations |
| Workflow ledger endpoints | Retrieve and verify recorded workflow events |
| HITL endpoints | Inspect and act on pending human reviews |

FastAPI and Pydantic provide request validation and generated OpenAPI documentation.

## 9. Observability, Accounting, and Auditability

### 9.1 Telemetry

The telemetry service tracks request counts, tier usage, cache hits, estimated tokens, illustrative costs, estimated savings, energy estimates, latency-related observations, and reviewed-quality observations. It also maintains recent routing tradeoffs used by the dashboard's Pareto view.

The application clearly labels cost values as estimates. This is important because free-tier quotas, changing provider prices, local energy costs, and actual billing data are not available to the application in a uniform form.

### 9.2 SQLite Event Ledger

Workflow events are written asynchronously to SQLite. Events may include workflow start and completion, task creation and completion, routing decisions, provider calls and responses, review events, HITL events, and errors. Parent event identifiers connect related records. A global hash chain supports tamper detection and verification.

Sensitive prompt content is not stored directly in provider-call records. Instead, the ledger stores sanitized prompt hashes, response hashes, lengths, routing metadata, and redaction status.

## 10. Web Dashboard

The client is a static browser dashboard composed of HTML, CSS, and JavaScript. Its delivered features include:

- provider health badges for Gemini, Groq, OpenRouter, OpenAI, and Ollama;
- privacy-mode control;
- execution-mode selection;
- provider and model selection;
- direct prompt execution;
- synchronous and background workflow submission;
- live SSE workflow progress;
- active specialist and queued-subtask visibility;
- task-level routing-audit disclosure;
- complexity, confidence, provider, model, and savings information;
- telemetry polling;
- Pareto-style routing visualization;
- offline markdown rendering for headings, lists, quotes, and code blocks;
- client-side removal of residual JSON artifacts from displayed model output;
- resilient layout constraints for the workflow sidebar.

The dashboard is designed to make the model decision inspectable rather than presenting the selected provider as an opaque implementation detail.

## 11. Reliability and Security Features

The project includes the following reliability and security behaviors:

- provider-specific fallback paths;
- Groq model-pool rotation for rate-limit and availability failures;
- Gemini model alias handling for changing model identifiers;
- explicit local timeout handling;
- privacy-mode cloud blocking;
- prompt redaction before routing;
- cache normalization and debug inspection;
- structured errors for invalid execution modes and missing jobs;
- background worker exception reporting through `job_failed` events;
- hash-chain verification for ledger records;
- cross-platform ASCII console status markers for Windows compatibility;
- client-side layout constraints to prevent sidebar collapse;
- output cleanup to reduce raw JSON and escaped-newline leakage in the UI.

A reusable circuit-breaker module is present in the core package, but it is not currently integrated into the active router path. This is documented as an implementation boundary rather than presented as an active request-protection feature.

## 12. Verification and Engineering Artifacts

The repository contains focused unit tests for:

- editing-versus-payload classification;
- classifier tier and signal behavior;
- API-key and email redaction;
- normalized cache storage and retrieval;
- cache debug metadata;
- ledger ordering and parent identifiers;
- ledger hash-chain verification and tamper detection;
- route-mode validation and timeout ordering.

The benchmark driver exercises offline, cloud, and automatic modes, duplicate requests, expected tier assignment, latency collection, cache behavior, and illustrative savings accounting. Benchmark output is stored separately as an application-generated telemetry snapshot and should not be interpreted as provider billing evidence or as a general performance guarantee.

The project also provides an API verification script and Python compilation checks as part of the operational development workflow.

## 13. Current Limitations

The following limitations define the current implementation boundary:

1. The classifier uses regular expressions and weighted heuristics; it is not an ML or XGBoost classifier.
2. The cache is exact-match, in-memory, and process-local; it does not provide vector similarity or persistence.
3. Workflow specialists execute sequentially rather than as parallel branches of a distributed DAG.
4. Automatic cloud routing currently prioritizes Groq and Gemini; OpenAI and OpenRouter are primarily explicit-override providers.
5. Background jobs and HITL state are lost after process restart.
6. SSE reports workflow lifecycle events, not token-level output streaming.
7. The dashboard's HITL controls are not fully wired to approval and rejection actions.
8. Cost and savings values are illustrative estimates, not billing records.
9. The redactor uses pattern matching and cannot guarantee detection of every secret or personal-data format.
10. The active router does not currently use the reusable circuit-breaker module.
11. The ledger hash chain is global to the SQLite event stream, so workflow-level verification must account for the shared chain.
12. The current tests do not comprehensively cover provider integrations, background jobs, HITL transitions, telemetry, or browser behavior.

## 14. Future Work

Future engineering work should prioritize:

- persistent job and HITL storage;
- cancellation and retry controls for background workflows;
- token-level streaming where provider APIs support it;
- integration of the circuit breaker into provider execution;
- provider-aware automatic selection across all configured providers;
- learned or embedding-assisted complexity classification with an interpretable fallback;
- persistent semantic caching with privacy-preserving keys;
- parallel specialist execution with dependency-aware scheduling;
- complete dashboard HITL actions;
- broader integration and end-to-end test coverage;
- authenticated API access, origin restriction, and deployment-grade secret management;
- calibration of confidence estimates against labeled human-review outcomes;
- optional durable cost accounting using provider usage metadata.

## 15. Conclusion

This project implements an observable multi-agent LLM platform in which model execution is controlled by a centralized cost and privacy gateway. Its delivered capabilities include interpretable complexity routing, exact-response caching, local Ollama operation, optional multi-provider cloud access, prompt redaction, routing explanations, fallback behavior, specialist orchestration, reviewer confidence assessment, HITL registration, background jobs, SSE progress, telemetry, and a tamper-evident event ledger.

The principal architectural contribution is the separation of workflow intelligence from model execution policy. Because all specialist calls pass through the gateway, privacy enforcement, provider selection, cost estimation, and audit data can evolve without rewriting each agent. The current system is a functional foundation for cost-aware and privacy-conscious multi-agent applications, with clearly identified boundaries for persistence, parallelism, learned classification, production resilience, and complete UI workflow support.

## References

1. Project source code and tests, `LLM-Cost-Autopilot-Agent-Orchestrator` repository.
2. FastAPI documentation, https://fastapi.tiangolo.com/
3. Ollama documentation, https://ollama.com/
4. Google Gemini API documentation, https://ai.google.dev/
5. Groq API documentation, https://console.groq.com/docs
6. OpenAI API documentation, https://platform.openai.com/docs/
7. OpenRouter API documentation, https://openrouter.ai/docs/
8. Python `sqlite3` documentation, https://docs.python.org/3/library/sqlite3.html
9. W3C Server-Sent Events specification, https://html.spec.whatwg.org/multipage/server-sent-events.html

## Appendix A. Repository Feature Map

| Repository area | Feature coverage |
|---|---|
| `server/app/main.py` | FastAPI application and HTTP API |
| `server/app/gateway/` | Classifier, cache, redaction, and routing gateway |
| `server/app/orchestration/` | Workflow, agents, review, HITL, and jobs |
| `server/app/providers/` | Gemini, Groq, OpenAI, OpenRouter, and Ollama adapters |
| `server/app/core/` | Configuration, telemetry, ledger, and circuit-breaker utility |
| `client/` | Dashboard, styling, progress, audit display, and markdown rendering |
| `server/tests/` | Gateway and ledger unit tests |
| `benchmark_cost_autopilot.py` | Mode, tier, cache, latency, and estimate benchmark driver |
| `scripts/verify_apis.py` | Provider and API verification utility |
| `PROJECT_CHRONICLE.md` | Architecture decisions, historical failures, and evolution log |
