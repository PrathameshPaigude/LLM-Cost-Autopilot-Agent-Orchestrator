import json
import re
from typing import Optional
from .state import WorkflowState
from ..gateway.router import router
from ..core.config import settings

class ReviewerAgent:
    def __init__(self):
        self.system_prompt = (
            "You are the Lead Synthesizer and Quality Audit Specialist. "
            "Your job is to evaluate the specialist outputs and produce the complete, high-quality, comprehensive final answer for the user. "
            "IMPORTANT RULES:\n"
            "1. Output ONLY the polished, final answer directly in clean GitHub markdown.\n"
            "2. DO NOT output JSON wrapper objects, review notes, or meta commentary.\n"
            "3. DO NOT include phrases like 'Here is the synthesis' or 'Review Notes:'.\n"
            "4. Provide thorough, well-structured, directly usable content with clear headings, explanations, and code examples if applicable."
        )

    def review_and_synthesize(
        self, 
        state: WorkflowState,
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> WorkflowState:
        # Aggregate outputs from all completed subtasks
        context_parts = []
        for task in state.subtasks:
            context_parts.append(f"### Specialist ({task.assigned_agent}) on '{task.description}':\n{task.output}\n")
        
        aggregated_context = "\n".join(context_parts)
        prompt = (
            f"User Goal: {state.user_prompt}\n\n"
            f"Specialist Findings & Logic:\n{aggregated_context}\n\n"
            "Task: Synthesize a single, cohesive, authoritative, and direct final answer for the user in clean Markdown format."
        )

        res = router.route_and_execute(
            prompt=prompt,
            agent_name="Reviewer",
            system_prompt=self.system_prompt,
            force_offline=force_offline,
            provider_override=provider_override,
            model_override=model_override
        )

        raw_response = res.get("response", "").strip()

        # Clean any accidental JSON wrapping from over-obedient models
        clean_output = self._extract_clean_text(raw_response)
        
        state.confidence_score = 0.95
        state.final_output = clean_output
        state.status = "completed"

        return state

    def _extract_clean_text(self, text: str) -> str:
        if not text:
            return "No output generated."

        # If model returned a raw JSON object string
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                if "final_synthesis" in data:
                    return data["final_synthesis"].strip()
                if "answer" in data:
                    return data["answer"].strip()
                if "response" in data:
                    return data["response"].strip()
            except Exception:
                pass

        # If wrapped in ```json ```
        if "```json" in text:
            try:
                json_str = text.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                if "final_synthesis" in data:
                    return data["final_synthesis"].strip()
            except Exception:
                pass

        # Remove any leading meta commentary artifacts
        text = re.sub(r'^(?:Here is the (?:final )?synthesis:?|Review Notes:?|Final Synthesis:?)\s*', '', text, flags=re.IGNORECASE).strip()
        return text

reviewer = ReviewerAgent()
