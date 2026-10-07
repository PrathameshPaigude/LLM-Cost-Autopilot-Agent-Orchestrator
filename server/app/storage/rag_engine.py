import os
import re
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class DocumentChunk:
    def __init__(self, chunk_id: str, doc_id: str, content: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.tokens = self._tokenize(content)
        self.tf = Counter(self.tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Split on non-alphanumeric while preserving snake_case and camelCase tokens
        words = re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())
        return words

class RAGEngine:
    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.doc_counts: Counter = Counter()  # Term -> number of chunks containing term
        self.total_chunks: int = 0
        self.indexed_files: List[str] = []

    def clear(self):
        self.chunks.clear()
        self.doc_counts.clear()
        self.total_chunks = 0
        self.indexed_files.clear()

    def index_text(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, chunk_size: int = 350, overlap: int = 50) -> int:
        metadata = metadata or {}
        lines = content.splitlines()
        chunks_created = 0
        
        # Line-aware chunking for code and documentation
        current_chunk_lines = []
        current_word_count = 0
        start_line = 1

        for idx, line in enumerate(lines, 1):
            current_chunk_lines.append(line)
            current_word_count += len(line.split())

            if current_word_count >= chunk_size or idx == len(lines):
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunk_id = f"{doc_id}#chunk_{chunks_created + 1}"
                    chunk_meta = {
                        **metadata,
                        "doc_id": doc_id,
                        "start_line": start_line,
                        "end_line": idx,
                        "line_count": len(current_chunk_lines)
                    }
                    chunk = DocumentChunk(chunk_id, doc_id, chunk_text, chunk_meta)
                    self.chunks[chunk_id] = chunk
                    
                    # Update document frequencies for BM25/TF-IDF
                    unique_terms = set(chunk.tokens)
                    for t in unique_terms:
                        self.doc_counts[t] += 1
                    
                    chunks_created += 1

                # Overlap retention
                if overlap > 0 and len(current_chunk_lines) > 4:
                    keep_lines = current_chunk_lines[-max(2, len(current_chunk_lines) // 4):]
                    current_chunk_lines = list(keep_lines)
                    current_word_count = sum(len(l.split()) for l in current_chunk_lines)
                    start_line = idx - len(current_chunk_lines) + 1
                else:
                    current_chunk_lines = []
                    current_word_count = 0
                    start_line = idx + 1

        self.total_chunks = len(self.chunks)
        if doc_id not in self.indexed_files:
            self.indexed_files.append(doc_id)
        return chunks_created

    def index_directory(self, root_dir: str, extensions: Optional[List[str]] = None, max_files: int = 100) -> Dict[str, Any]:
        extensions = extensions or [".py", ".md", ".json", ".js", ".html", ".css", ".txt", ".yaml", ".yml", ".sql"]
        root_path = Path(root_dir).resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"Directory {root_dir} does not exist or is not a directory.")

        indexed_count = 0
        total_chunks = 0
        skipped = []

        ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".idea", ".vscode"}

        for p in root_path.rglob("*"):
            if any(part in ignore_dirs for part in p.parts):
                continue
            if p.is_file() and p.suffix.lower() in extensions:
                if indexed_count >= max_files:
                    break
                try:
                    rel_path = str(p.relative_to(root_path)).replace("\\", "/")
                    content = p.read_text(encoding="utf-8", errors="replace")
                    chunks = self.index_text(rel_path, content, metadata={"file_path": rel_path, "file_name": p.name})
                    total_chunks += chunks
                    indexed_count += 1
                except Exception as e:
                    skipped.append(f"{p.name}: {str(e)}")

        return {
            "files_indexed": indexed_count,
            "total_chunks": total_chunks,
            "total_knowledge_chunks": self.total_chunks,
            "skipped": skipped
        }

    def search(self, query: str, top_k: int = 4, score_threshold: float = 0.05) -> List[Dict[str, Any]]:
        if not self.chunks or not query.strip():
            return []

        query_tokens = DocumentChunk._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}
        avg_chunk_len = sum(len(c.tokens) for c in self.chunks.values()) / max(1, len(self.chunks))
        k1 = 1.5
        b = 0.75

        # BM25 Scoring
        for chunk_id, chunk in self.chunks.items():
            score = 0.0
            doc_len = len(chunk.tokens)
            
            for q_term in query_tokens:
                if q_term in chunk.tf:
                    freq = chunk.tf[q_term]
                    doc_freq = self.doc_counts.get(q_term, 1)
                    # IDF formula with smoothing
                    idf = math.log(1 + (self.total_chunks - doc_freq + 0.5) / (doc_freq + 0.5))
                    
                    # BM25 term weighting
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * (doc_len / avg_chunk_len))
                    score += idf * (numerator / denominator)

            # Metadata matching boost (e.g. filename in query)
            file_name = chunk.metadata.get("file_name", "").lower()
            if file_name and file_name in query.lower():
                score += 2.0

            if score > score_threshold:
                scores[chunk_id] = score

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, sc in sorted_chunks:
            chunk = self.chunks[chunk_id]
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "score": round(sc, 4),
                "metadata": chunk.metadata,
                "snippet": chunk.content
            })

        return results

    def format_rag_context(self, query: str, top_k: int = 3, max_tokens: int = 1200) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        context_parts = ["--- RELEVANT REPOSITORY CONTEXT (RAG) ---"]
        current_token_est = 0

        for r in results:
            header = f"[File: {r['doc_id']} (Lines {r['metadata'].get('start_line', 1)}-{r['metadata'].get('end_line', '?')}) | Relevance: {r['score']}]"
            snippet = r["snippet"]
            est_tokens = len(snippet.split()) * 4 // 3
            if current_token_est + est_tokens > max_tokens:
                # Truncate snippet to fit remaining token budget
                words = snippet.split()[:max(10, (max_tokens - current_token_est) * 3 // 4)]
                snippet = " ".join(words) + "\n...[truncated to fit token budget]"
                context_parts.append(f"{header}\n{snippet}")
                break
            context_parts.append(f"{header}\n{snippet}")
            current_token_est += est_tokens

        context_parts.append("--- END OF RAG CONTEXT ---")
        return "\n\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "indexed_files_count": len(self.indexed_files),
            "indexed_files": self.indexed_files[:30],
            "vocabulary_size": len(self.doc_counts)
        }

rag_engine = RAGEngine()
