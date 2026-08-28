import logging
from typing import Optional
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
        model_override: Optional[str] = None
    ) -> WorkflowState:
        # Step 1: Supervisor decomposes prompt into DAG of subtasks
        state = supervisor.plan_workflow(
            user_prompt=user_prompt, 
            force_offline=force_offline, 
            provider_override=provider_override, 
            model_override=model_override
        )
        state.status = "executing"

        accumulated_context = f"Project Goal: {user_prompt}\nPlan: {state.plan_overview}\n"

        # Step 2: Execute specialist nodes sequentially (intercepted by Gateway)
        for task in state.subtasks:
            agent = SPECIALIST_MAP.get(task.assigned_agent, code_agent)
            task.status = "in_progress"
            
            result = agent.execute(
                task_description=task.description, 
                context=accumulated_context,
                force_offline=force_offline,
                provider_override=provider_override,
                model_override=model_override
            )
            task.output = result.get("response", "")
            task.complexity_score = result.get("complexity_score", 0.5)
            task.routed_tier = result.get("tier_used", "Tier 2")
            task.status = "completed"

            accumulated_context += f"\nOutput from {task.assigned_agent} on '{task.description}':\n{task.output}\n"

        # Step 3: Reviewer evaluates confidence and synthesizes final output
        state.status = "review"
        reviewed_state = reviewer.review_and_synthesize(
            state=state,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override
        )

        # Step 4: Handle HITL Escalation if confidence < 0.80
        if reviewed_state.status == "pending_hitl":
            hitl_manager.register_for_review(reviewed_state)
            logger.warning(f"Workflow {reviewed_state.workflow_id} paused for Human Review (Confidence: {reviewed_state.confidence_score}).")

        reviewed_state.telemetry_summary = telemetry.get_summary()
        return reviewed_state

engine = OrchestrationEngine()
