from typing import Optional
from ...gateway.router import router

class AnalysisAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a Senior Systems Analyst and Architect. Break down technical systems, evaluate trade-offs, "
            "and create rigorous architectural plans and data flow diagrams."
        )

    def execute(
        self, 
        task_description: str, 
        context: str = "", 
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> dict:
        prompt = f"Context: {context}\n\nTask: {task_description}" if context else task_description
        return router.route_and_execute(
            prompt=prompt,
            agent_name="AnalysisAgent",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override,
            workflow_id=workflow_id,
        )

analysis_agent = AnalysisAgent()
