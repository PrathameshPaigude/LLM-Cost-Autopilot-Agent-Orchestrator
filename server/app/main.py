import os
import json
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .core.config import settings
from .core.telemetry import telemetry
from .core.ledger import ledger
from .core.context_bridge import context_bridge
from .core.quality_tracker import quality_tracker
from .gateway.router import router
from .gateway.cache import cache
from .gateway.classifier import classifier
from .orchestration.workflow import engine
from .orchestration.jobs import jobs
from .orchestration.hitl import hitl_manager
from .providers.gemini_provider import gemini_provider
from .providers.openai_provider import openai_provider
from .providers.groq_provider import groq_provider
from .providers.openrouter_provider import openrouter_provider
from .providers.huggingface_provider import huggingface_provider
from .providers.ollama_provider import ollama_provider
from .storage.rag_engine import rag_engine

app = FastAPI(
    title="Agent Orchestration Platform & LLM Cost Autopilot",
    version="1.5.0",
    description="Multi-Agent AI Platform with Intelligent LLM Routing Gateway, Zero-Cost Free Tiers, HuggingFace Benchmark Ingestion, Adaptive Performance Quality Tracking, and RAG Context Indexing."
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
    rag_context: Optional[str] = None

class WorkflowRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    user_prompt: str
    force_offline: Optional[bool] = None
    provider_override: Optional[str] = None
    model_override: Optional[str] = None
    parallel: Optional[bool] = True

class WorkflowJobRequest(WorkflowRequest):
    pass

class HITLActionRequest(BaseModel):
    human_edits: Optional[str] = None
    reason: Optional[str] = "Approved by human reviewer"

class PrivacyToggleRequest(BaseModel):
    privacy_mode: bool

class RAGIndexRequest(BaseModel):
    doc_id: Optional[str] = "custom_doc"
    content: Optional[str] = None
    directory_path: Optional[str] = None
    extensions: Optional[List[str]] = None

class RAGSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4

class BridgeExportRequest(BaseModel):
    target_platform: str  # chatgpt, gemini, claude, ollama
    messages: List[Dict[str, str]]
    active_code: Optional[str] = None

class SaveFileRequest(BaseModel):
    file_path: str
    content: str

class RouterFeedbackRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model: str
    provider: str
    rating: int
    prompt_snippet: Optional[str] = None
    comment: Optional[str] = None

class TrainRouterRequest(BaseModel):
    include_benchmarks: Optional[bool] = True
    include_observations: Optional[bool] = True
    min_obs: Optional[int] = 10

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
        "ml_classifier_active": classifier.is_ml_active(),
        "rag_chunks_indexed": rag_engine.total_chunks,
        "providers": {
            "gemini": {
                "configured": gemini_provider.is_configured(),
                "tier": "Free & Frontier",
                "default_model": "gemini-2.0-flash"
            },
            "groq": {
                "configured": groq_provider.is_configured(),
                "tier": "Free Tier (Fast)",
                "default_model": "qwen/qwen3-32b"
            },
            "huggingface": {
                "configured": huggingface_provider.is_configured(),
                "tier": "Serverless Free Hub",
                "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct"
            },
            "openai": {
                "configured": openai_provider.is_configured(),
                "tier": "Commercial Frontier",
                "default_model": "gpt-4o-mini"
            },
            "openrouter": {
                "configured": openrouter_provider.is_configured(),
                "tier": "100% Free Roster",
                "default_model": "deepseek/deepseek-r1"
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
        "ml_classifier_active": classifier.is_ml_active(),
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
                "provider": "Hugging Face",
                "id": "huggingface",
                "type": "Cloud (Serverless Free Hub)",
                "configured": huggingface_provider.is_configured(),
                "models": [
                    {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "name": "Qwen 2.5 Coder 32B (Top Coding - Free)", "tier": "Tier 2"},
                    {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B Instruct (Free)", "tier": "Tier 1"},
                    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B v0.3 (Free)", "tier": "Tier 1"},
                    {"id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "name": "DeepSeek Coder V2 Lite (Free)", "tier": "Tier 2"}
                ]
            },
            {
                "provider": "Groq Cloud",
                "id": "groq",
                "type": "Cloud (Ultra-Fast Free Tier)",
                "configured": groq_provider.is_configured(),
                "models": [
                    {"id": "qwen/qwen3-32b", "name": "Qwen3 32B", "tier": "Tier 2"},
                    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "tier": "Tier 1"},
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
                    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1 ($0.70/$2.50 per 1M tokens)", "tier": "Tier 3"},
                    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Meta Llama 3.3 70B (100% Free)", "tier": "Tier 2/3"},
                    {"id": "qwen/qwen-2.5-coder-32b-instruct:free", "name": "Qwen 2.5 Coder 32B (100% Free)", "tier": "Tier 2"},
                    {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B Instruct (100% Free)", "tier": "Tier 1"}
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
                    {"id": "o3-mini", "name": "o3-mini (Deep STEM Reasoning)", "tier": "Tier 3"}
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
        rag_context=req.rag_context,
    )

# -------------------------------------------------------------
# RAG Endpoints
# -------------------------------------------------------------

@app.post("/api/v1/rag/index")
def index_rag_content(req: RAGIndexRequest):
    """Indexes text or directory contents into the hybrid RAG engine."""
    if req.directory_path:
        return rag_engine.index_directory(req.directory_path, extensions=req.extensions)
    elif req.content:
        chunks = rag_engine.index_text(req.doc_id or "custom_doc", req.content)
        return {"status": "success", "chunks_indexed": chunks, "total_chunks": rag_engine.total_chunks}
    else:
        # Default index current repository root
        repo_root = str(Path(__file__).resolve().parents[2])
        return rag_engine.index_directory(repo_root)

@app.post("/api/v1/rag/search")
def search_rag(req: RAGSearchRequest):
    """Searches the indexed RAG knowledge base for top matching snippets."""
    results = rag_engine.search(req.query, top_k=req.top_k or 4)
    formatted = rag_engine.format_rag_context(req.query, top_k=req.top_k or 3)
    return {
        "results": results,
        "formatted_context": formatted,
        "count": len(results)
    }

@app.get("/api/v1/rag/stats")
def get_rag_stats():
    """Returns indexed knowledge chunks and file statistics."""
    return rag_engine.get_stats()

@app.delete("/api/v1/rag/clear")
def clear_rag_index():
    """Clears the active RAG index."""
    rag_engine.clear()
    return {"status": "cleared", "total_chunks": 0}

# -------------------------------------------------------------
# Context Bridge Endpoints
# -------------------------------------------------------------

@app.post("/api/v1/bridge/export")
def export_context_bridge(req: BridgeExportRequest):
    """Generates an optimized prompt to transfer context cleanly to ChatGPT, Gemini, or Claude."""
    handoff_prompt = context_bridge.generate_handoff_prompt(
        target_platform=req.target_platform,
        messages=req.messages,
        active_code=req.active_code
    )
    compressed = context_bridge.compress_history(req.messages)
    return {
        "handoff_prompt": handoff_prompt,
        "compressed_summary": compressed,
        "target_platform": req.target_platform
    }

# -------------------------------------------------------------
# Workspace Files API (for Coding Agent IDE)
# -------------------------------------------------------------

@app.get("/api/v1/workspace/files")
def list_workspace_files():
    """Lists workspace files for the Coding Agent file tree."""
    root_dir = Path(__file__).resolve().parents[2]
    file_list = []
    ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
    for p in root_dir.rglob("*"):
        if any(part in ignore_dirs for part in p.parts):
            continue
        if p.is_file():
            rel_path = str(p.relative_to(root_dir)).replace("\\", "/")
            file_list.append({
                "path": rel_path,
                "name": p.name,
                "size": p.stat().st_size,
                "extension": p.suffix
            })
    return {"workspace_root": str(root_dir), "files": file_list[:200]}

@app.get("/api/v1/workspace/file")
def get_workspace_file(path: str = Query(..., description="Relative path to file")):
    """Reads content of a workspace file."""
    root_dir = Path(__file__).resolve().parents[2]
    target_path = (root_dir / path).resolve()
    if not str(target_path).startswith(str(root_dir)):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        content = target_path.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workspace/file")
def save_workspace_file(req: SaveFileRequest):
    """Saves updated content to a workspace file."""
    root_dir = Path(__file__).resolve().parents[2]
    target_path = (root_dir / req.file_path).resolve()
    if not str(target_path).startswith(str(root_dir)):
        raise HTTPException(status_code=403, detail="Access denied.")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(req.content, encoding="utf-8")
        return {"status": "saved", "path": req.file_path, "bytes": len(req.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------
# Workflow & Telemetry Endpoints
# -------------------------------------------------------------

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
        model_override=req.model_override,
        parallel_execution=bool(req.parallel)
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

# -------------------------------------------------------------
# Router Performance, Feedback & HuggingFace Benchmark Endpoints
# -------------------------------------------------------------

@app.get("/api/v1/router/performance")
def get_router_performance():
    """Returns aggregated performance matrix, live observation stats, and benchmark calibrations."""
    matrix = quality_tracker.get_performance_matrix()
    benchmark_status = classifier.get_benchmark_status()
    
    training_report = None
    report_file = Path(__file__).resolve().parents[2] / "data" / "latest_training_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                training_report = json.load(f)
        except Exception:
            pass

    return {
        "performance_matrix": matrix,
        "benchmark_status": benchmark_status,
        "latest_training_report": training_report,
        "classifier_ml_active": classifier.is_ml_active(),
    }

@app.post("/api/v1/router/feedback")
def submit_router_feedback(req: RouterFeedbackRequest):
    """Submits user quality feedback rating (1-5) for a routed model response."""
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")
    quality_tracker.record_feedback(
        model=req.model,
        provider=req.provider,
        rating=req.rating,
        prompt_snippet=req.prompt_snippet,
        comment=req.comment
    )
    return {
        "status": "success",
        "message": f"Feedback ({req.rating}/5) recorded for {req.model}.",
        "total_feedback_entries": quality_tracker.feedback_count
    }

@app.get("/api/v1/router/benchmarks")
def get_benchmark_info():
    """Returns status and distribution of loaded routing benchmarks (R2-Bench, etc.)."""
    return classifier.get_benchmark_status()

@app.post("/api/v1/router/train")
def retrain_router_classifier(req: Optional[TrainRouterRequest] = None):
    """Triggers retraining of the XGBoost router complexity classifier with benchmark & observation data."""
    try:
        from scripts.train_router_model import train_and_export
        gateway_dir = str(Path(__file__).resolve().parent / "gateway")
        inc_bench = req.include_benchmarks if req else True
        inc_obs = req.include_observations if req else True
        min_obs = req.min_obs if req else 10
        summary = train_and_export(
            export_dir=gateway_dir,
            include_benchmarks=inc_bench,
            include_observations=inc_obs,
            min_obs=min_obs
        )
        # Hot-reload ML model in active classifier instance
        classifier._load_trained_ml_model()
        return {
            "status": "success",
            "message": "Router model retrained and hot-reloaded successfully.",
            "metrics": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


# Serve static client UI — must be mounted AFTER all API routes
_client_dir = Path(__file__).resolve().parents[2] / "client"
if _client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_client_dir)), name="static")
    app.mount("/css", StaticFiles(directory=str(_client_dir / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(_client_dir / "js")), name="js")

    @app.get("/favicon.ico", include_in_schema=False)
    def serve_favicon():
        fav_path = _client_dir / "favicon.svg"
        if fav_path.exists():
            return FileResponse(str(fav_path), media_type="image/svg+xml")
        return FileResponse(str(_client_dir / "index.html"))

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(_client_dir / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
