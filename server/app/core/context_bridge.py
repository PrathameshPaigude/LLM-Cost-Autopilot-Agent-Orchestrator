import re
import json
import time
from typing import Dict, List, Any, Optional

class ContextBridgeEngine:
    """
    Context Bridge & Cross-LLM Transmission Engine.
    Enables seamless context export, compression, and handoff across LLMs
    (e.g., ChatGPT -> Gemini -> Claude -> Local Ollama) without losing state or blowing token budgets.
    """

    def compress_history(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> Dict[str, Any]:
        """
        Compresses a multi-turn conversation into a dense, structured context block.
        """
        user_goals = []
        key_decisions = []
        code_snippets = []
        recent_exchanges = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()

            if role == "user":
                # Extract core goals
                if any(kw in content.lower() for kw in ["build", "create", "fix", "implement", "design", "write", "how"]):
                    user_goals.append(content[:150])
            elif role == "assistant":
                # Extract code blocks
                codes = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", content)
                for code in codes:
                    if len(code.strip()) > 20:
                        first_lines = "\n".join(code.strip().splitlines()[:6])
                        code_snippets.append(first_lines + ("\n..." if len(code.splitlines()) > 6 else ""))

        # Keep last 2-3 raw turns
        recent_turns = messages[-3:] if len(messages) >= 3 else messages
        for m in recent_turns:
            recent_exchanges.append(f"**{m.get('role', '').capitalize()}:** {m.get('content', '')[:250]}")

        compressed_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "primary_goals": user_goals[:4],
            "code_artifacts_summary": len(code_snippets),
            "recent_context": recent_exchanges,
            "total_raw_turns": len(messages)
        }
        return compressed_summary

    def generate_handoff_prompt(
        self, 
        target_platform: str, 
        messages: List[Dict[str, str]], 
        active_code: Optional[str] = None,
        workflow_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates an optimized, ready-to-paste prompt for transferring work
        directly to ChatGPT, Gemini, Claude, or Local Ollama.
        """
        target = target_platform.lower()
        compressed = self.compress_history(messages)

        goals_text = "\n".join(f"- {g}" for g in compressed["primary_goals"]) or "- Continue project workflow"
        recent_text = "\n".join(compressed["recent_context"]) or "No previous turns."

        if "claude" in target:
            return (
                "Here is my active project session context transferred from Agent Orchestrator. "
                "Please resume working from this exact state.\n\n"
                "<context_handoff>\n"
                f"<primary_goals>\n{goals_text}\n</primary_goals>\n\n"
                f"<recent_conversation>\n{recent_text}\n</recent_conversation>\n"
                + (f"\n<active_code>\n{active_code}\n</active_code>\n" if active_code else "")
                + "</context_handoff>\n\n"
                "Please confirm you have this context and tell me the next recommended step."
            )
        elif "gemini" in target:
            return (
                "**[AGENT ORCHESTRATOR CONTEXT BRIDGE]**\n\n"
                "### Context & Project State Transfer:\n"
                f"**Goals & Focus:**\n{goals_text}\n\n"
                f"**Recent Conversation Summary:**\n{recent_text}\n\n"
                + (f"**Active Code in Workspace:**\n```python\n{active_code}\n```\n\n" if active_code else "")
                + "Please acknowledge this context state and let me know how you'd like to proceed!"
            )
        elif "chatgpt" in target or "openai" in target:
            return (
                "I am continuing a coding and problem-solving session transferred from an external orchestrator. "
                "Here is the summary of what we have done so far:\n\n"
                f"**Objectives:**\n{goals_text}\n\n"
                f"**Previous Exchange:**\n{recent_text}\n\n"
                + (f"**Working Code / Spec:**\n```\n{active_code}\n```\n\n" if active_code else "")
                + "Please resume seamlessly without repeating what was already accomplished."
            )
        else:
            # Universal format
            return (
                "=== CONTEXT BRIDGE HANDOFF ===\n"
                f"Goals:\n{goals_text}\n\n"
                f"Recent Turns:\n{recent_text}\n"
                + (f"\nCode:\n{active_code}\n" if active_code else "")
                + "\n=== RESUME WORK ==="
            )

    def export_session_json(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None) -> str:
        data = {
            "format": "agent_orchestrator_bridge_v1",
            "exported_at": time.time(),
            "messages": messages,
            "metadata": metadata or {}
        }
        return json.dumps(data, indent=2)

    def import_session_json(self, raw_json: str) -> List[Dict[str, str]]:
        data = json.loads(raw_json)
        if isinstance(data, dict) and "messages" in data:
            return data["messages"]
        elif isinstance(data, list):
            return data
        raise ValueError("Invalid session JSON format.")

context_bridge = ContextBridgeEngine()
