from typing import Optional
from ...gateway.router import router

class ResearchAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a Research Specialist. Conduct deep domain research, synthesize facts, "
            "cite sources and technical standards where applicable."
        )

    def execute(
        self, 
        task_description: str, 
        context: str = "", 
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> dict:
        prompt = f"Context: {context}\n\nTask: {task_description}" if context else task_description
        return router.route_and_execute(
            prompt=prompt,
            agent_name="ResearchAgent",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override
        )

research_agent = ResearchAgent()
