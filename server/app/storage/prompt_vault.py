"""
Prompt Vault - Persistent SQLite Store for Training & Telemetry
==============================================================
Thread-safe SQLite database storing all prompts, compressed variants,
complexity scores, model routing decisions, tokens, costs, and feedback ratings.
Used for continuous model retraining, dataset export, and audit analysis.
"""

import sqlite3
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "prompt_vault.sqlite3"


class PromptVault:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prompts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        session_id TEXT DEFAULT 'default',
                        agent_name TEXT DEFAULT 'General',
                        original_prompt TEXT NOT NULL,
                        compressed_prompt TEXT,
                        tokens_original INTEGER DEFAULT 0,
                        tokens_compressed INTEGER DEFAULT 0,
                        tokens_saved INTEGER DEFAULT 0,
                        compression_ratio REAL DEFAULT 1.0,
                        complexity_score REAL DEFAULT 0.0,
                        tier_used TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        response_snippet TEXT,
                        tokens_in INTEGER DEFAULT 0,
                        tokens_out INTEGER DEFAULT 0,
                        cost_usd REAL DEFAULT 0.0,
                        latency_ms REAL DEFAULT 0.0,
                        is_cached INTEGER DEFAULT 0,
                        user_rating INTEGER,
                        for_training INTEGER DEFAULT 1
                    );
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prompts_tier ON prompts(tier_used);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prompts_model ON prompts(model_name);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prompts_created ON prompts(created_at);
                """)
                conn.commit()

    def store_prompt(
        self,
        original_prompt: str,
        tier_used: str,
        provider: str,
        model_name: str,
        compressed_prompt: Optional[str] = None,
        tokens_original: int = 0,
        tokens_compressed: int = 0,
        compression_ratio: float = 1.0,
        complexity_score: float = 0.0,
        response_snippet: Optional[str] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        is_cached: bool = False,
        session_id: str = "default",
        agent_name: str = "General",
        user_rating: Optional[int] = None,
        for_training: bool = True
    ) -> int:
        """Stores a prompt interaction in SQLite and returns the row id."""
        tokens_saved = max(0, tokens_original - tokens_compressed)
        snippet = (response_snippet or "")[:500]
        created_at = datetime.utcnow().isoformat() + "Z"

        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO prompts (
                            created_at, session_id, agent_name, original_prompt, compressed_prompt,
                            tokens_original, tokens_compressed, tokens_saved, compression_ratio,
                            complexity_score, tier_used, provider, model_name, response_snippet,
                            tokens_in, tokens_out, cost_usd, latency_ms, is_cached,
                            user_rating, for_training
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        created_at, session_id, agent_name, original_prompt, compressed_prompt or original_prompt,
                        tokens_original, tokens_compressed or tokens_original, tokens_saved, compression_ratio,
                        complexity_score, tier_used, provider, model_name, snippet,
                        tokens_in, tokens_out, cost_usd, latency_ms, 1 if is_cached else 0,
                        user_rating, 1 if for_training else 0
                    ))
                    conn.commit()
                    return cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to store prompt in PromptVault: {e}")
                return -1

    def update_rating(self, prompt_id: int, rating: int) -> bool:
        """Updates user rating for a specific stored prompt."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE prompts SET user_rating = ? WHERE id = ?;", (rating, prompt_id))
                    conn.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to update rating in PromptVault: {e}")
                return False

    def get_recent_prompts(
        self,
        limit: int = 50,
        offset: int = 0,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves paginated prompts with optional filtering."""
        query = "SELECT * FROM prompts WHERE 1=1"
        params = []

        if tier:
            query += " AND tier_used LIKE ?"
            params.append(f"%{tier}%")
        if model:
            query += " AND model_name LIKE ?"
            params.append(f"%{model}%")
        if search:
            query += " AND (original_prompt LIKE ? OR response_snippet LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY id DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Calculates aggregate telemetry across all stored prompts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    COALESCE(SUM(tokens_original), 0) as total_tokens_orig,
                    COALESCE(SUM(tokens_compressed), 0) as total_tokens_comp,
                    COALESCE(SUM(tokens_saved), 0) as total_tokens_saved,
                    COALESCE(SUM(cost_usd), 0.0) as total_cost,
                    COALESCE(AVG(complexity_score), 0.0) as avg_complexity,
                    COALESCE(AVG(latency_ms), 0.0) as avg_latency,
                    COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END) as rated_count,
                    COUNT(CASE WHEN for_training = 1 THEN 1 END) as training_ready_count
                FROM prompts;
            """)
            row = cursor.fetchone()
            total_count = row["total_count"] if row else 0

            # Get tier breakdown
            cursor.execute("SELECT tier_used, COUNT(*) as cnt FROM prompts GROUP BY tier_used;")
            tier_dist = {r["tier_used"]: r["cnt"] for r in cursor.fetchall()}

            # Get model breakdown
            cursor.execute("SELECT model_name, COUNT(*) as cnt FROM prompts GROUP BY model_name;")
            model_dist = {r["model_name"]: r["cnt"] for r in cursor.fetchall()}

            avg_compression = 1.0
            if row and row["total_tokens_orig"] > 0:
                avg_compression = round(row["total_tokens_comp"] / row["total_tokens_orig"], 3)

            return {
                "total_prompts": total_count,
                "total_tokens_original": row["total_tokens_orig"] if row else 0,
                "total_tokens_compressed": row["total_tokens_comp"] if row else 0,
                "total_tokens_saved": row["total_tokens_saved"] if row else 0,
                "total_cost_usd": round(row["total_cost"] if row else 0.0, 6),
                "avg_complexity": round(row["avg_complexity"] if row else 0.0, 2),
                "avg_latency_ms": round(row["avg_latency"] if row else 0.0, 1),
                "avg_compression_ratio": avg_compression,
                "rated_count": row["rated_count"] if row else 0,
                "training_ready_count": row["training_ready_count"] if row else 0,
                "tier_distribution": tier_dist,
                "model_distribution": model_dist,
                "database_path": str(self.db_path)
            }

    def export_dataset_for_training(self) -> List[Dict[str, Any]]:
        """Exports prompts formatted as training rows for classifier retraining."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT original_prompt, tier_used, complexity_score, user_rating
                FROM prompts
                WHERE for_training = 1
                ORDER BY id ASC;
            """)
            rows = cursor.fetchall()
            return [
                {
                    "prompt": r["original_prompt"],
                    "tier": r["tier_used"],
                    "complexity_score": r["complexity_score"],
                    "rating": r["user_rating"]
                }
                for r in rows
            ]


prompt_vault = PromptVault()
