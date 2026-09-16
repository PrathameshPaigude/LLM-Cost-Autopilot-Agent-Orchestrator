from typing import Optional
from ...gateway.router import router

class WritingAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert Technical Writer and Communications Specialist. "
            "Synthesize technical inputs into clear, professional, well-formatted documentation or responses."
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
            agent_name="WritingAgent",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override,
            workflow_id=workflow_id,
        )

writing_agent = WritingAgent()
