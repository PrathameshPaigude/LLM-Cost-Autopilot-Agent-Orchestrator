import os
import json
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .core.config import settings
from .core.telemetry import telemetry
from .core.ledger import ledger
from .gateway.router import router
from .gateway.cache import cache
from .orchestration.workflow import engine
from .orchestration.jobs import jobs
from .orchestration.hitl import hitl_manager
from .providers.gemini_provider import gemini_provider
from .providers.openai_provider import openai_provider
from .providers.groq_provider import groq_provider
from .providers.openrouter_provider import openrouter_provider
from .providers.ollama_provider import ollama_provider

app = FastAPI(
    title="Agent Orchestration Platform & LLM Cost Autopilot",
    version="1.1.0",
    description="Multi-Agent AI Platform with Intelligent LLM Routing Gateway, Zero-Cost Free Tiers, and Real-Time Savings Telemetry."
)

# Enable CORS for web UI client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class DirectRouteRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    prompt: str
    agent_name: Optional[str] = "General"
    force_offline: Optional[bool] = None
    provider_override: Optional[str] = None
    model_override: Optional[str] = None
    execution_mode: Optional[str] = None
    mode: Optional[str] = None
    force_cloud: Optional[bool] = None
    classifier_debug: bool = False
    debug: bool = False

class WorkflowRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    user_prompt: str
    force_offline: Optional[bool] = None
    provider_override: Optional[str] = None
    model_override: Optional[str] = None


class WorkflowJobRequest(WorkflowRequest):
    pass


class HITLActionRequest(BaseModel):
    human_edits: Optional[str] = None
    reason: Optional[str] = "Approved by human reviewer"

class PrivacyToggleRequest(BaseModel):
    privacy_mode: bool

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------

@app.get("/api/v1/health")
def health_check():
    ollama_online = ollama_provider.is_available()
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "privacy_mode": settings.PRIVACY_MODE,
        "providers": {
            "gemini": {
                "configured": gemini_provider.is_configured(),
                "tier": "Free & Frontier",
                "default_model": "gemini-2.0-flash"
            },
            "groq": {
                "configured": groq_provider.is_configured(),
                "tier": "Free Tier (Fast)",
                "default_model": "llama-3.3-70b-versatile"
            },
            "openai": {
                "configured": openai_provider.is_configured(),
                "tier": "Commercial Frontier",
                "default_model": "gpt-4o-mini"
            },
            "openrouter": {
                "configured": openrouter_provider.is_configured(),
                "tier": "100% Free Roster",
                "default_model": "deepseek/deepseek-r1:free"
            },
            "ollama": {
                "configured": True,
                "online": ollama_online,
                "tier": "Local 100% Offline (Free)",
                "tier1_model": settings.LOCAL_TIER1_MODEL,
                "tier2_model": settings.LOCAL_TIER2_MODEL
            }
        }
    }

@app.get("/api/v1/models")
def list_available_models():
    """Returns the matrix of available models across all providers."""
    local_installed = ollama_provider.list_installed_models() if ollama_provider.is_available() else []
    
    return {
        "active_privacy_mode": settings.PRIVACY_MODE,
        "roster": [
            {
                "provider": "Google Gemini",
                "id": "gemini",
                "type": "Cloud (Free & Paid)",
                "configured": gemini_provider.is_configured(),
                "models": [
                    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Fast / Multimodal - Free)", "tier": "Tier 2"},
                    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (Ultra Fast - Free)", "tier": "Tier 1"},
                    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (2M Context - Free)", "tier": "Tier 3"},
                    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro (Frontier Reasoning)", "tier": "Tier 3"}
                ]
            },
            {
                "provider": "OpenAI",
                "id": "openai",
                "type": "Cloud (Commercial)",
                "configured": openai_provider.is_configured(),
                "models": [
                    {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Cost Efficient)", "tier": "Tier 1/2"},
                    {"id": "gpt-4o", "name": "GPT-4o (Omni Frontier)", "tier": "Tier 3"},
                    {"id": "o3-mini", "name": "o3-mini (Deep STEM Reasoning)", "tier": "Tier 3"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo (Legacy Fast)", "tier": "Tier 1"}
                ]
            },
            {
                "provider": "Groq Cloud",
                "id": "groq",
                "type": "Cloud (Ultra-Fast Free Tier)",
                "configured": groq_provider.is_configured(),
                "models": [
                    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile (Free Tier)", "tier": "Tier 2/3"},
                    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant (500+ tok/s - Free)", "tier": "Tier 1"},
                    {"id": "deepseek-r1-distill-llama-70b", "name": "DeepSeek R1 Distill 70B (Free)", "tier": "Tier 3"},
                    {"id": "gemma2-9b-it", "name": "Gemma 2 9B (Free)", "tier": "Tier 1"},
                    {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B MoE (Free)", "tier": "Tier 2"}
                ]
            },
            {
                "provider": "OpenRouter",
                "id": "openrouter",
                "type": "Cloud (100% Free Tier Hub)",
                "configured": openrouter_provider.is_configured(),
                "models": [
                    {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1 (100% Free)", "tier": "Tier 3"},
                    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Meta Llama 3.3 70B (100% Free)", "tier": "Tier 2/3"},
                    {"id": "qwen/qwen-2.5-coder-32b-instruct:free", "name": "Qwen 2.5 Coder 32B (100% Free)", "tier": "Tier 2"},
                    {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B Instruct (100% Free)", "tier": "Tier 1"}
                ]
            },
            {
                "provider": "Local Ollama",
                "id": "ollama",
                "type": "Local Offline (Zero Cloud Cost)",
                "configured": True,
                "online": ollama_provider.is_available(),
                "installed_models": local_installed,
                "models": [
                    {"id": "qwen2.5:1.5b", "name": "Qwen 2.5 1.5B (Tier 1 Lightweight - 1.2 GB RAM)", "tier": "Tier 1"},
                    {"id": "llama3.1:8b-instruct-q4_K_M", "name": "Llama 3.1 8B Q4 (Tier 2 Balanced - 4.8 GB RAM)", "tier": "Tier 2"},
                    {"id": "deepseek-r1:8b", "name": "DeepSeek R1 8B (Local Reasoning)", "tier": "Tier 3"}
                ]
            }
        ]
    }

@app.post("/api/v1/route")
def direct_gateway_route(req: DirectRouteRequest, x_max_cost: Optional[float] = Header(None)):
    """Directly invokes the LLM Cost Autopilot Gateway."""
    return router.route_and_execute(
        prompt=req.prompt,
        agent_name=req.agent_name or "General",
        force_offline=req.force_offline,
        provider_override=req.provider_override,
        model_override=req.model_override,
        execution_mode=req.execution_mode or req.mode,
        force_cloud=bool(req.force_cloud),
        classifier_debug=req.classifier_debug or req.debug,
    )

@app.get("/api/v1/debug/cache")
def debug_cache(prompt: Optional[str] = None):
    """Inspect cache key rules and optionally test whether a prompt is cached."""
    return cache.debug_info(prompt)

@app.delete("/api/v1/debug/cache")
def clear_debug_cache():
    """Clear the in-memory cache before an isolated benchmark mode run."""
    cache.clear()
    return cache.debug_info()

@app.post("/api/v1/workflow/run")
def run_orchestration_workflow(req: WorkflowRequest):
    """Executes full Multi-Agent Orchestration workflow (Supervisor -> Specialists -> Reviewer)."""
    state = engine.execute_workflow(
        user_prompt=req.user_prompt,
        force_offline=req.force_offline,
        provider_override=req.provider_override,
        model_override=req.model_override
    )
    return state.dict()


@app.post("/api/v1/workflow/jobs", status_code=202)
def create_workflow_job(req: WorkflowJobRequest):
    """Starts a workflow in the background and returns a trackable job ID."""
    return jobs.create_job(
        user_prompt=req.user_prompt,
        force_offline=req.force_offline,
        provider_override=req.provider_override,
        model_override=req.model_override,
    )


@app.get("/api/v1/workflow/jobs/{job_id}")
def get_workflow_job(job_id: str):
    """Returns the latest state of a background workflow job."""
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Workflow job not found.")
    return job


@app.get("/api/v1/workflow/jobs/{job_id}/events")
def stream_workflow_job(job_id: str):
    """Streams workflow progress as Server-Sent Events until the job finishes."""
    event_queue = jobs.subscribe(job_id)
    if not event_queue:
        raise HTTPException(status_code=404, detail="Workflow job not found.")

    def event_stream():
        while True:
            try:
                event = event_queue.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in {"job_completed", "job_failed"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/v1/telemetry")
def get_telemetry():
    """Retrieves real-time token, dollar, and compute energy savings."""
    return telemetry.get_summary()

@app.get("/api/v1/telemetry/pareto")
def get_telemetry_pareto():
    """Returns recent routing tradeoffs and reviewed quality outcomes."""
    return telemetry.get_pareto()

@app.get("/api/v1/workflow/{workflow_id}/ledger")
def get_workflow_ledger(workflow_id: str):
    """Returns the ordered evidence ledger and its tamper check for a workflow."""
    entries = ledger.get_workflow_entries(workflow_id)
    return {
        "workflow_id": workflow_id,
        "entries": [entry.to_dict() for entry in entries],
        "verify_chain": ledger.verify_chain(workflow_id),
    }

@app.get("/api/v1/hitl/pending")
def list_pending_reviews():
    """Lists all workflows awaiting human approval."""
    return hitl_manager.list_all_pending()

@app.post("/api/v1/hitl/{workflow_id}/approve")
def approve_workflow(workflow_id: str, req: HITLActionRequest):
    """Approves and finalizes a paused workflow."""
    try:
        approved_state = hitl_manager.approve(workflow_id, human_edits=req.human_edits)
        return approved_state.dict()
    except KeyError:
        raise HTTPException(status_code=404, detail="Workflow ID not found in pending review queue.")

@app.post("/api/v1/hitl/{workflow_id}/reject")
def reject_workflow(workflow_id: str, req: HITLActionRequest):
    """Rejects a workflow."""
    try:
        rejected_state = hitl_manager.reject(workflow_id, reason=req.reason or "Rejected by reviewer")
        return rejected_state.dict()
    except KeyError:
        raise HTTPException(status_code=404, detail="Workflow ID not found in pending review queue.")

@app.post("/api/v1/config/privacy")
def set_privacy_mode(req: PrivacyToggleRequest):
    """Toggles offline privacy mode globally."""
    settings.PRIVACY_MODE = req.privacy_mode
    return {"privacy_mode": settings.PRIVACY_MODE, "message": "Updated global privacy enforcement mode."}

# Serve static client UI — must be mounted AFTER all API routes
_client_dir = Path(__file__).resolve().parents[2] / "client"
if _client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_client_dir)), name="static")
    app.mount("/css", StaticFiles(directory=str(_client_dir / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(_client_dir / "js")), name="js")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(_client_dir / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
