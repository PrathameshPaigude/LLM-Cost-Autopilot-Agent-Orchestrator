# ⚡ Agent Orchestration Platform with LLM Cost Autopilot Gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Offline%20Ready-black.svg)](https://ollama.com)
[![Zero Cost Strategy](https://img.shields.io/badge/Compute%20Cost-%240.00%20Free%20Tier-success.svg)](#-zero-cost-execution-matrix)

An enterprise-grade, stateful Multi-Agent AI Orchestration Engine equipped with an **Intelligent Routing Gateway (LLM Cost Autopilot)**, dual-mode resilient execution (Cloud APIs + 100% Offline Local Ollama), and real-time compute/cost savings telemetry.

---

## 🌟 Key Architectural Features

* **3-Tier Cost Autopilot Gateway**: Intercepts every subtask, scores semantic complexity from $0.00$ to $1.00$, and dynamically routes to the most cost-effective tier.
* **100% Free & Offline-Ready**: Supports local execution via Ollama (`Qwen 2.5 1.5B` & `Llama 3.1 8B`) alongside generous free-tier cloud APIs (`Groq Cloud`, `Google Gemini`, `OpenRouter`).
* **Air-Gapped Privacy Mode**: Enforces strict local-only routing with zero external network transmission when enabled.
* **Intent-Aware Classification**: Distinguishes active generation commands from passive text editing payloads to prevent false-positive cost escalation.
* **Multi-Agent Specialist Team**:
  * **Supervisor Planner**: Hierarchically decomposes projects into sequential subtasks.
  * **Specialists**: Research Agent, Systems Analysis Agent, Code Agent, Technical Writing Agent, and Computer Vision (YOLOv8) Specialist.
  * **Lead Synthesizer & Reviewer**: Quality audit node synthesizing authoritative final delivery.
* **Real-Time Savings Telemetry**: Live tracking of cloud money saved (USD), tokens processed, and watt-hours of compute energy conserved.

---

## 🏗️ System Architecture & Flow

```
                                  [ User / Web Dashboard ]
                                             ▲
                                             │ HTTP / REST
                                             ▼
                                  [ FastAPI Gateway Layer ]
                                             ▲
                                             │ State Dispatch
                                             ▼
                   ┌──────────────────────────────────────────────────┐
                   │        Multi-Agent Orchestration State Machine   │
                   ├──────────────────────────────────────────────────┤
                   │  1. Supervisor Router (Task Decomposition)      │
                   │  2. Specialist Nodes (Research, Code, Vision...) │
                   │  3. Quality Audit & Final Synthesizer Node       │
                   └─────────────────────────┬────────────────────────┘
                                             │ All Agent LLM Calls
                                             ▼
                   ┌──────────────────────────────────────────────────┐
                   │    Intelligent Routing Gateway (Cost Autopilot)   │
                   ├──────────────────────────────────────────────────┤
                   │  [0] Semantic Cache (SHA-256 In-Memory / ~0ms)   │
                   │  [1] PII & Secret Redactor (Presidio / Regex)    │
                   │  [2] ML Complexity Classifier (0.0 to 1.0)       │
                   └─────────────────────────┬────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
      [ TIER 1: FAST/LIGHT ]        [ TIER 2: BALANCED ]          [ TIER 3: FRONTIER ]
      Score: 0.00 – 0.30            Score: 0.31 – 0.70            Score: 0.71 – 1.00
      • Groq: gpt-oss-20b (500+ t/s)• Groq: gpt-oss-120b          • Gemini: gemini-pro-latest
      • Gemini: flash-lite-latest   • Gemini: flash-latest        • OpenAI: gpt-4o / o3-mini
      • Local: Qwen 2.5 1.5B        • Local: Llama 3.1 8B         • OpenRouter: DeepSeek-R1:free
```

---

## ⚡ Zero-Cost Execution Matrix

| Tier | Complexity | Providers & Models (Free & Paid) | Purpose |
| :--- | :--- | :--- | :--- |
| **Tier 0** | 0.00 | **Semantic Cache** (SHA-256 In-Memory) | Instant $\approx 0\text{ ms}$ response on repeated queries ($0.00 cost) |
| **Tier 1** | 0.00 – 0.30 | • **Groq (Free):** `openai/gpt-oss-20b` (500+ tok/s)<br>• **Gemini (Free):** `gemini-flash-lite-latest`<br>• **OpenRouter (Free):** `mistralai/mistral-7b:free`<br>• **OpenAI:** `gpt-4o-mini`<br>• **Local Ollama:** `qwen2.5:1.5b` (~1.2 GB RAM) | Fast parsing, grammar correction, formatting, quick micro-tasks |
| **Tier 2** | 0.31 – 0.70 | • **Groq (Free):** `openai/gpt-oss-120b`<br>• **Gemini (Free):** `gemini-flash-latest`<br>• **OpenRouter (Free):** `qwen/qwen-2.5-coder-32b:free`<br>• **OpenAI:** `gpt-4o-mini`<br>• **Local Ollama:** `llama3.1:8b` (~4.8 GB RAM) | Software development, code refactoring, system analysis, technical writing |
| **Tier 3** | 0.71 – 1.00 | • **Gemini (Free):** `gemini-pro-latest`<br>• **OpenAI:** `gpt-4o`, `o3-mini`<br>• **Groq (Free):** `qwen/qwen3.8-27b`<br>• **OpenRouter (Free):** `deepseek/deepseek-r1:free`<br>• **Local Ollama:** `llama3.1:8b` | Multi-agent synthesis, system architecture, deep reasoning, quality review |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com) installed and running locally
* (Optional) Free API keys from [Google AI Studio](https://aistudio.google.com/app/apikey), [Groq Cloud](https://console.groq.com), or [OpenRouter](https://openrouter.ai/keys).

### 2. Clone & Install Dependencies
```powershell
git clone https://github.com/PrathameshPaigude/LLM-Cost-Autopilot-Agent-Orchestrator.git
cd LLM-Cost-Autopilot-Agent-Orchestrator
pip install -r requirements.txt
```

### 3. Pull Free Local Models
```powershell
ollama pull qwen2.5:1.5b
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 4. Configure Environment
Copy `.env.example` to `.env` and add your API keys:
```powershell
cp .env.example .env
```

### 5. Launch the Server
```powershell
python -m server.app.main
```
The FastAPI backend will be live at `http://localhost:8000`.

### 6. Launch the Client Dashboard
Open `client/index.html` in any web browser.

### 7. Verify All Connected APIs
```powershell
python scripts/verify_apis.py
```

---

## 📖 Architecture Chronicle & Decision Log
For the complete engineering ledger, mathematical hardware sizing, intent vs. payload NLP classification proof, failure post-mortems, and technology trade-off evaluations, refer to:
👉 **[`PROJECT_CHRONICLE.md`](PROJECT_CHRONICLE.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
