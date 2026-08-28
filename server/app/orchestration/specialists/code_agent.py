from typing import Optional
from ...gateway.router import router

class CodeAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert Senior Software Engineer. Provide clean, robust, well-commented code. "
            "Include error handling, clear type annotations, and explain your technical choices concisely."
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
            agent_name="CodeAgent",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override
        )

code_agent = CodeAgent()
