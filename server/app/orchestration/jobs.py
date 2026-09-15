import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .workflow import engine


class WorkflowJobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="workflow")
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        user_prompt: str,
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        event_queue: queue.Queue = queue.Queue()
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": self._now(),
            "updated_at": self._now(),
            "workflow": None,
            "error": None,
            "events": event_queue,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._publish(job_id, {"type": "job_queued", "job_id": job_id, "status": "queued"})
        self._executor.submit(
            self._run_job,
            job_id,
            user_prompt,
            force_offline,
            provider_override,
            model_override,
        )
        return self.public_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if job_id not in self._jobs:
                return None
        return self.public_job(job_id)

    def subscribe(self, job_id: str) -> Optional[queue.Queue]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job["events"] if job else None

    def public_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            return {key: value for key, value in job.items() if key != "events"}

    def _run_job(
        self,
        job_id: str,
        user_prompt: str,
        force_offline: Optional[bool],
        provider_override: Optional[str],
        model_override: Optional[str],
    ) -> None:
        self._update(job_id, status="running")
        self._publish(job_id, {"type": "job_started", "job_id": job_id, "status": "running"})
        try:
            state = engine.execute_workflow(
                user_prompt=user_prompt,
                force_offline=force_offline,
                provider_override=provider_override,
                model_override=model_override,
                event_callback=lambda event: self._publish(job_id, event),
            )
            workflow = state.model_dump()
            self._update(job_id, status=state.status, workflow=workflow)
            self._publish(job_id, {"type": "job_completed", "job_id": job_id, "status": state.status, "workflow": workflow})
        except Exception as exc:
            self._update(job_id, status="failed", error=str(exc))
            self._publish(job_id, {"type": "job_failed", "job_id": job_id, "status": "failed", "error": str(exc)})

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes, updated_at=self._now())

    def _publish(self, job_id: str, event: Dict[str, Any]) -> None:
        self._update(job_id, last_event=event)
        self._jobs[job_id]["events"].put(event)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


jobs = WorkflowJobManager()
