import json
import uuid
from typing import List, Optional
from .state import AgentTask, WorkflowState
from ..gateway.router import router

class SupervisorAgent:
    def __init__(self):
        self.system_prompt = (
            "You are the Supervisor Planner of a Multi-Agent system. "
            "Your job is to decompose the user's high-level goal into 2 to 4 concrete, sequential subtasks. "
            "Assign each task to one of the following specialist agents: "
            "['ResearchAgent', 'AnalysisAgent', 'CodeAgent', 'WritingAgent', 'CVSpecialist']. "
            "Output your plan strictly in JSON format as follows: "
            '{"plan_overview": "Brief summary", "subtasks": [{"description": "task detail", "assigned_agent": "AgentName"}]}'
        )

    def plan_workflow(
        self, 
        user_prompt: str, 
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> WorkflowState:
        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        state = WorkflowState(workflow_id=workflow_id, user_prompt=user_prompt)

        # Call Gateway for Planning
        router_res = router.route_and_execute(
            prompt=f"Goal: {user_prompt}\nDeconstruct this into subtasks.",
            agent_name="Supervisor",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override
        )

        try:
            # Parse structured JSON plan from model response
            content = router_res["response"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            state.plan_overview = parsed.get("plan_overview", "Standard decomposition plan")
            for idx, task_data in enumerate(parsed.get("subtasks", [])):
                task = AgentTask(
                    task_id=f"task-{idx+1}",
                    description=task_data.get("description", ""),
                    assigned_agent=task_data.get("assigned_agent", "CodeAgent")
                )
                state.subtasks.append(task)
        except Exception:
            # Heuristic fallback decomposition if model returns free text
            state.plan_overview = "Automated Task Decomposition"
            state.subtasks = [
                AgentTask(task_id="task-1", description=f"Analyze requirements & architecture for: {user_prompt}", assigned_agent="AnalysisAgent"),
                AgentTask(task_id="task-2", description=f"Implement core logic / code for: {user_prompt}", assigned_agent="CodeAgent"),
                AgentTask(task_id="task-3", description=f"Format documentation and delivery for: {user_prompt}", assigned_agent="WritingAgent")
            ]

        state.status = "planning"
        return state

supervisor = SupervisorAgent()
