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
            "4. Provide thorough, well-structured, directly usable content with clear headings, explanations, and code examples if applicable.\n"
            "5. Preserve every explicit requirement from the User Goal.\n"
            "6. For programming tasks, verify that every requested file is present, code blocks are complete, "
            "imports and declarations are included, examples match the requested language/version, and compile "
            "commands are accurate. Do not silently simplify or change the requested deliverable.\n"
            "7. If specialist findings are incomplete, use the User Goal to fill the gap and return a complete answer."
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
        
        state.confidence_score = self._estimate_confidence(state, clean_output)
        state.final_output = clean_output
        state.status = "completed" if state.confidence_score >= settings.CONFIDENCE_THRESHOLD else "pending_hitl"

        return state

    def _estimate_confidence(self, state: WorkflowState, output: str) -> float:
        if not output or output == "No output generated.":
            return 0.0

        score = 0.55
        completed_tasks = sum(task.status == "completed" for task in state.subtasks)
        if state.subtasks and completed_tasks == len(state.subtasks):
            score += 0.20
        if len(output.strip()) >= 200:
            score += 0.10
        if "```" in output:
            score += 0.05
        if "error" in output.lower() and "without error" not in output.lower():
            score -= 0.10

        return round(max(0.0, min(0.95, score)), 2)

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
