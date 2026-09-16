import hashlib
import json
import logging
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    WORKFLOW_STARTED = "workflow_started"
    TASK_CREATED = "task_created"
    TASK_ROUTED = "task_routed"
    PROVIDER_CALLED = "provider_called"
    PROVIDER_RESPONSE = "provider_response"
    TASK_COMPLETED = "task_completed"
    REVIEW_STARTED = "review_started"
    REVIEW_COMPLETED = "review_completed"
    HITL_TRIGGERED = "hitl_triggered"
    HITL_RESOLVED = "hitl_resolved"
    WORKFLOW_COMPLETED = "workflow_completed"


@dataclass
class LedgerEntry:
    entry_id: uuid.UUID
    workflow_id: str
    timestamp: str
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    parent_entry_id: Optional[uuid.UUID] = None
    previous_entry_hash: Optional[str] = None

    @classmethod
    def create(
        cls,
        workflow_id: str,
        event_type: EventType,
        payload: Optional[dict[str, Any]] = None,
        parent_entry_id: Optional[str] = None,
    ) -> "LedgerEntry":
        return cls(
            entry_id=uuid.uuid4(),
            workflow_id=workflow_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            payload=payload or {},
            parent_entry_id=uuid.UUID(parent_entry_id) if parent_entry_id else None,
        )

    def content(self) -> dict[str, Any]:
        return {
            "entry_id": str(self.entry_id),
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "parent_entry_id": str(self.parent_entry_id) if self.parent_entry_id else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.content(), "previous_entry_hash": self.previous_entry_hash}


class EvidenceLedger:
    def __init__(self, database_path: Optional[str] = None) -> None:
        default_path = Path(__file__).resolve().parents[3] / "data" / "ledger.sqlite3"
        self.database_path = Path(database_path or default_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_queue: queue.Queue[Optional[LedgerEntry]] = queue.Queue()
        self._initialize_database()
        self._writer = threading.Thread(target=self._write_loop, name="evidence-ledger", daemon=True)
        self._writer.start()

    def append_entry(self, entry: LedgerEntry) -> str:
        """Queue an insert and return immediately so ledger I/O never blocks a workflow."""
        try:
            self._write_queue.put_nowait(entry)
        except Exception as exc:
            logger.warning("Evidence ledger enqueue failed: %s", exc)
        return str(entry.entry_id)

    def get_workflow_entries(self, workflow_id: str) -> list[LedgerEntry]:
        self._write_queue.join()
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT entry_id, workflow_id, timestamp, event_type, payload_json,
                       parent_entry_id, previous_entry_hash
                FROM ledger_entries
                WHERE workflow_id = ?
                ORDER BY sequence_number ASC
                """,
                (workflow_id,),
            ).fetchall()
        finally:
            connection.close()
        return [self._entry_from_row(row) for row in rows]

    def verify_chain(self, workflow_id: str) -> bool:
        self._write_queue.join()
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT entry_id, workflow_id, timestamp, event_type, payload_json,
                       parent_entry_id, previous_entry_hash, entry_hash
                FROM ledger_entries
                ORDER BY sequence_number ASC
                """,
            ).fetchall()
        finally:
            connection.close()

        previous_hash = ""
        workflow_seen = False
        for row in rows:
            entry = self._entry_from_row(row)
            if entry.previous_entry_hash != previous_hash:
                return False
            if self._hash_entry(entry, previous_hash) != row[7]:
                return False
            previous_hash = row[7]
            workflow_seen = workflow_seen or row[1] == workflow_id
        return workflow_seen

    def _write_loop(self) -> None:
        while True:
            entry = self._write_queue.get()
            if entry is None:
                return
            try:
                self._insert_entry(entry)
            except Exception:
                logger.exception("Evidence ledger insert failed")
            finally:
                self._write_queue.task_done()

    def _insert_entry(self, entry: LedgerEntry) -> None:
        connection = self._connect()
        try:
            previous_row = connection.execute(
                "SELECT entry_hash FROM ledger_entries ORDER BY sequence_number DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous_row[0] if previous_row else ""
            entry.previous_entry_hash = previous_hash
            entry_hash = self._hash_entry(entry, previous_hash)
            connection.execute(
                """
                INSERT INTO ledger_entries
                (entry_id, workflow_id, timestamp, event_type, payload_json,
                 parent_entry_id, previous_entry_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(entry.entry_id), entry.workflow_id, entry.timestamp,
                    entry.event_type.value, json.dumps(entry.payload, sort_keys=True),
                    str(entry.parent_entry_id) if entry.parent_entry_id else None,
                    previous_hash, entry_hash,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    sequence_number INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL UNIQUE,
                    workflow_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    parent_entry_id TEXT,
                    previous_entry_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=5)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _hash_entry(entry: LedgerEntry, previous_hash: str) -> str:
        content = json.dumps(entry.content(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{content}{previous_hash}".encode("utf-8")).hexdigest()

    @staticmethod
    def _entry_from_row(row: tuple[Any, ...]) -> LedgerEntry:
        return LedgerEntry(
            entry_id=uuid.UUID(row[0]),
            workflow_id=row[1],
            timestamp=row[2],
            event_type=EventType(row[3]),
            payload=json.loads(row[4]),
            parent_entry_id=uuid.UUID(row[5]) if row[5] else None,
            previous_entry_hash=row[6],
        )


ledger = EvidenceLedger()


def append_entry(
    workflow_id: str,
    event_type: EventType,
    payload: Optional[dict[str, Any]] = None,
    parent_entry_id: Optional[str] = None,
) -> str:
    return ledger.append_entry(LedgerEntry.create(workflow_id, event_type, payload, parent_entry_id))