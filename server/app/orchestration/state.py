from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentTask(BaseModel):
    task_id: str
    description: str
    assigned_agent: str  # Research, Analysis, Code, Writing, CVSpecialist
    complexity_score: Optional[float] = None
    routed_tier: Optional[str] = None
    output: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed

class WorkflowState(BaseModel):
    workflow_id: str
    user_prompt: str
    plan_overview: Optional[str] = None
    subtasks: List[AgentTask] = Field(default_factory=list)
    current_step: int = 0
    final_output: Optional[str] = None
    confidence_score: float = 0.0
    status: str = "initialized"  # initialized, planning, executing, review, pending_hitl, completed, rejected
    telemetry_summary: Dict[str, Any] = Field(default_factory=dict)
