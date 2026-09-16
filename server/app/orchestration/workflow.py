import logging
import uuid
from typing import Any, Callable, Optional
from .state import WorkflowState
from .supervisor import supervisor
from .reviewer import reviewer
from .hitl import hitl_manager
from .specialists.code_agent import code_agent
from .specialists.writing_agent import writing_agent
from .specialists.analysis_agent import analysis_agent
from .specialists.research_agent import research_agent
from .specialists.cv_agent import cv_agent
from ..core.telemetry import telemetry
from ..core.ledger import EventType, append_entry
from ..gateway.redactor import redactor

logger = logging.getLogger(__name__)

SPECIALIST_MAP = {
    "CodeAgent": code_agent,
    "WritingAgent": writing_agent,
    "AnalysisAgent": analysis_agent,
    "ResearchAgent": research_agent,
    "CVSpecialist": cv_agent
}

class OrchestrationEngine:
    def execute_workflow(
        self, 
        user_prompt: str, 
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> WorkflowState:
        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        workflow_start_entry = append_entry(
            workflow_id, EventType.WORKFLOW_STARTED,
            {"prompt_length": len(user_prompt), "force_offline": force_offline},
        )
        def emit(event_type: str, **payload: Any) -> None:
            if event_callback:
                event_callback({"type": event_type, **payload})

        emit("workflow_stage", stage="planning", message="Supervisor is decomposing the goal")
        # Step 1: Supervisor decomposes prompt into DAG of subtasks
        state = supervisor.plan_workflow(
            user_prompt=user_prompt, 
            workflow_id=workflow_id,
            force_offline=force_offline, 
            provider_override=provider_override, 
            model_override=model_override
        )
        state.status = "executing"
        emit(
            "plan_created",
            workflow_id=state.workflow_id,
            plan_overview=state.plan_overview,
            subtasks=[task.model_dump() for task in state.subtasks],
        )
        for task in state.subtasks:
            safe_description, _ = redactor.sanitize(task.description)
            append_entry(
                state.workflow_id, EventType.TASK_CREATED,
                {"task_id": task.task_id, "assigned_agent": task.assigned_agent,
                 "description_summary": safe_description[:160]},
                parent_entry_id=workflow_start_entry,
            )

        accumulated_context = f"Project Goal: {user_prompt}\nPlan: {state.plan_overview}\n"

        # Step 2: Execute specialist nodes sequentially (intercepted by Gateway)
        for index, task in enumerate(state.subtasks):
            agent = SPECIALIST_MAP.get(task.assigned_agent, code_agent)
            task.status = "in_progress"
            state.current_step = index + 1
            emit(
                "task_started",
                workflow_id=state.workflow_id,
                task_id=task.task_id,
                task_index=index,
                total_tasks=len(state.subtasks),
                assigned_agent=task.assigned_agent,
                description=task.description,
            )
            
            result = agent.execute(
                task_description=task.description, 
                context=accumulated_context,
                force_offline=force_offline,
                provider_override=provider_override,
                model_override=model_override,
                workflow_id=state.workflow_id,
            )
            task.output = result.get("response", "")
            task.complexity_score = result.get("complexity_score", 0.5)
            task.routed_tier = result.get("tier_used", "Tier 2")
            task.routing_audit = result.get("routing_audit", {})
            task.status = "completed"
            emit(
                "task_completed",
                workflow_id=state.workflow_id,
                task_id=task.task_id,
                task_index=index,
                assigned_agent=task.assigned_agent,
                complexity_score=task.complexity_score,
                routed_tier=task.routed_tier,
                routing_audit=task.routing_audit,
            )
            append_entry(
                state.workflow_id, EventType.TASK_COMPLETED,
                {"task_id": task.task_id, "assigned_agent": task.assigned_agent,
                 "complexity_score": task.complexity_score, "routed_tier": task.routed_tier},
            )

            accumulated_context += f"\nOutput from {task.assigned_agent} on '{task.description}':\n{task.output}\n"

        # Step 3: Reviewer evaluates confidence and synthesizes final output
        state.status = "review"
        emit("workflow_stage", stage="review", message="Reviewer is checking the specialist outputs")
        append_entry(state.workflow_id, EventType.REVIEW_STARTED, {"task_count": len(state.subtasks)})
        reviewed_state = reviewer.review_and_synthesize(
            state=state,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override,
            workflow_id=state.workflow_id,
        )
        reviewed_cost = sum(
            task.routing_audit.get("selected", {}).get("estimated_cost_usd", 0.0)
            for task in state.subtasks
        )
        telemetry.record_quality_observation(reviewed_state.confidence_score, reviewed_cost)

        # Step 4: Handle HITL Escalation if confidence < 0.80
        if reviewed_state.status == "pending_hitl":
            hitl_manager.register_for_review(reviewed_state)
            append_entry(
                reviewed_state.workflow_id, EventType.HITL_TRIGGERED,
                {"confidence_score": reviewed_state.confidence_score},
            )
            logger.warning(f"Workflow {reviewed_state.workflow_id} paused for Human Review (Confidence: {reviewed_state.confidence_score}).")

        reviewed_state.telemetry_summary = telemetry.get_summary()
        emit(
            "workflow_finished",
            workflow_id=reviewed_state.workflow_id,
            status=reviewed_state.status,
            confidence_score=reviewed_state.confidence_score,
        )
        append_entry(
            reviewed_state.workflow_id, EventType.WORKFLOW_COMPLETED,
            {"status": reviewed_state.status, "confidence_score": reviewed_state.confidence_score},
        )
        return reviewed_state

engine = OrchestrationEngine()
