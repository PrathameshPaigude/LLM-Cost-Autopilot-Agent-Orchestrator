from typing import Optional
from ...gateway.router import router

class CodeAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert Senior Software Engineer. Solve the user's exact programming task, "
            "preserving every requested language, version, file name, API, feature, edge case, and "
            "output format. When code is requested, provide complete self-contained files that compile "
            "as written; never use placeholders such as 'add the rest here'. Include all required "
            "imports, declarations, and a runnable test or usage example when requested. Check the "
            "algorithm for correctness, boundary conditions, integer overflow, and stated complexity "
            "before answering. Explain technical choices concisely after the code."
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
