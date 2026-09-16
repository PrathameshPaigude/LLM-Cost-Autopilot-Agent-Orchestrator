from typing import Dict, Any
from .state import WorkflowState
from ..core.ledger import EventType, append_entry
from ..gateway.redactor import redactor

class HITLManager:
    def __init__(self):
        self._pending_workflows: Dict[str, WorkflowState] = {}

    def register_for_review(self, state: WorkflowState):
        self._pending_workflows[state.workflow_id] = state

    def get_pending(self, workflow_id: str) -> WorkflowState:
        return self._pending_workflows.get(workflow_id)

    def list_all_pending(self) -> Dict[str, Any]:
        return {
            wf_id: {
                "user_prompt": s.user_prompt,
                "confidence_score": s.confidence_score,
                "final_output_preview": (s.final_output or "")[:120] + "..."
            }
            for wf_id, s in self._pending_workflows.items()
        }

    def approve(self, workflow_id: str, human_edits: str = None) -> WorkflowState:
        state = self._pending_workflows.pop(workflow_id, None)
        if not state:
            raise KeyError(f"Workflow {workflow_id} not found in pending review queue.")
        
        if human_edits:
            state.final_output = human_edits
        state.status = "completed"
        append_entry(
            workflow_id, EventType.HITL_RESOLVED,
            {"resolution": "approved", "human_edits": bool(human_edits)},
        )
        return state

    def reject(self, workflow_id: str, reason: str = "Rejected by human reviewer") -> WorkflowState:
        state = self._pending_workflows.pop(workflow_id, None)
        if not state:
            raise KeyError(f"Workflow {workflow_id} not found in pending review queue.")
        
        state.status = "rejected"
        state.final_output = f"[REJECTED]: {reason}"
        safe_reason, _ = redactor.sanitize(reason)
        append_entry(
            workflow_id, EventType.HITL_RESOLVED,
            {"resolution": "rejected", "reason_summary": safe_reason[:160]},
        )
        return state

hitl_manager = HITLManager()
