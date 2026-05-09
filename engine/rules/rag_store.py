"""
CAFE-u RAG Store — Vector memory for past signals and successful adaptations.

When a frustration signal comes in, we query past similar signals
to find what adaptation worked before. This is the "retrieval" in RAG.

Uses FAISS for fast vector search. Falls back to keyword match if FAISS unavailable.
"""

import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("cafeu.rag")

# Import FAISS with graceful fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available — RAG will use keyword fallback")

# Import embedding model with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("sentence-transformers not available — RAG will use basic embeddings")


class EmbeddingEngine:
    """Convert signal descriptions into vector embeddings for similarity search."""

    def __init__(self):
        self.model = None
        if EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Embedding model loaded: all-MiniLM-L6-v2 (384-dim)")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")

    def encode(self, text: str) -> list[float]:
        """Convert text to embedding vector."""
        if self.model:
            return self.model.encode(text).tolist()
        # Fallback: simple character-level hash-based embedding (384-dim)
        return self._basic_embed(text)

    def _basic_embed(self, text: str, dim: int = 384) -> list[float]:
        """Simple deterministic embedding when no ML model available."""
        vec = [0.0] * dim
        for i, ch in enumerate(text):
            vec[i % dim] += ord(ch) / 255.0
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def is_loaded(self) -> bool:
        return self.model is not None


class SignalMemory:
    """
    A single stored memory: a past signal, the adaptation that worked,
    and whether the user found it helpful.
    """

    def __init__(self, signal: dict, adaptation: dict, effective: bool = True):
        self.signal = signal
        self.adaptation = adaptation
        self.effective = effective
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.id = hashlib.md5(
            json.dumps(signal, sort_keys=True).encode()
        ).hexdigest()[:12]

    def to_text(self) -> str:
        """Convert to text for embedding."""
        s = self.signal
        return (
            f"Signal: {s.get('type', 'unknown')} on element '{s.get('element', 'unknown')}' "
            f"count={s.get('count', 1)} frequency={s.get('frequency', 1)} "
            f"duration_ms={s.get('duration_ms', 0)} "
            f"Adaptation: {self.adaptation.get('action', 'none')} "
            f"params={self.adaptation}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "signal": self.signal,
            "adaptation": self.adaptation,
            "effective": self.effective,
            "timestamp": self.timestamp,
        }


class RAGStore:
    """
    Retrieval-Augmented Generation store.
    
    Stores past signal→adaptation pairs as vectors.
    On retrieval, returns top-k most similar past cases with their outcomes.
    """

    CACHE_PATH = Path(__file__).parent / "rag_cache"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.embedder = EmbeddingEngine()
        self.memories: list[SignalMemory] = []
        self.index = None
        self.index_dim = 384
        self.cache_dir = cache_dir or self.CACHE_PATH
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load cached index
        self._load()

    def add(self, signal: dict, adaptation: dict, effective: bool = True):
        """Store a signal→adaptation pair."""
        memory = SignalMemory(signal, adaptation, effective)
        self.memories.append(memory)

        # Build/update FAISS index
        if FAISS_AVAILABLE and len(self.memories) >= 1:
            text = memory.to_text()
            vec = np.array([self.embedder.encode(text)], dtype=np.float32)
            
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.index_dim)
            
            self.index.add(vec)

        # Auto-save every 10 memories
        if len(self.memories) % 10 == 0:
            self._save()

        logger.debug(f"Stored memory #{len(self.memories)}: {memory.id}")

    def retrieve(self, signal: dict, k: int = 3) -> list[dict]:
        """
        Retrieve top-k most similar past signals + their adaptations.
        
        Returns list of {memory, signal, adaptation, similarity_score}.
        """
        if not self.memories:
            return []

        query_text = self._signal_to_text(signal)

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            return self._vector_search(query_text, k)
        else:
            return self._keyword_search(signal, k)

    def _vector_search(self, query_text: str, k: int) -> list[dict]:
        """FAISS-based similarity search."""
        query_vec = np.array([self.embedder.encode(query_text)], dtype=np.float32)
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories):
                mem = self.memories[idx]
                results.append({
                    "memory": mem.to_dict(),
                    "signal": mem.signal,
                    "adaptation": mem.adaptation,
                    "similarity_score": float(1.0 / (1.0 + distances[0][i])),
                })

        return results

    def _keyword_search(self, signal: dict, k: int) -> list[dict]:
        """Fallback keyword-based similarity when FAISS unavailable."""
        signal_type = signal.get("type", "")
        element = signal.get("element", "")

        scored = []
        for mem in self.memories:
            score = 0.0
            # Same signal type = high score
            if mem.signal.get("type") == signal_type:
                score += 0.5
            # Same or similar element
            if mem.signal.get("element") == element:
                score += 0.3
            elif element and mem.signal.get("element", "") and (
                element in mem.signal["element"] or mem.signal["element"] in element
            ):
                score += 0.15
            # Effective adaptations weighted higher
            if mem.effective:
                score += 0.2
            scored.append((score, mem))

        scored.sort(key=lambda x: -x[0])
        return [
            {
                "memory": mem.to_dict(),
                "signal": mem.signal,
                "adaptation": mem.adaptation,
                "similarity_score": score,
            }
            for score, mem in scored[:k]
            if score > 0
        ]

    def _signal_to_text(self, signal: dict) -> str:
        """Convert a signal dict to search text."""
        return (
            f"Signal: {signal.get('type', 'unknown')} on element '{signal.get('element', 'unknown')}' "
            f"count={signal.get('count', 1)} frequency={signal.get('frequency', 1)} "
            f"duration_ms={signal.get('duration_ms', 0)}"
        )

    def report_feedback(self, signal: dict, adaptation: dict, was_helpful: bool):
        """
        Report whether an adaptation was helpful.
        Updates the stored memory for better future retrieval.
        """
        for mem in self.memories:
            if (mem.signal.get("element") == signal.get("element")
                    and mem.adaptation.get("action") == adaptation.get("action")):
                mem.effective = was_helpful
                logger.info(f"Feedback: {mem.id} → {'helpful' if was_helpful else 'not helpful'}")
                break
        self._save()

    def _save(self):
        """Persist memories and index to disk."""
        try:
            # Save memories
            mem_path = self.cache_dir / "memories.json"
            with open(mem_path, "w") as f:
                json.dump([m.to_dict() for m in self.memories], f)

            # Save FAISS index
            if FAISS_AVAILABLE and self.index is not None:
                idx_path = self.cache_dir / "faiss.index"
                faiss.write_index(self.index, str(idx_path))

            logger.debug(f"RAG cache saved ({len(self.memories)} memories)")
        except Exception as e:
            logger.warning(f"Failed to save RAG cache: {e}")

    def _load(self):
        """Load cached memories and index from disk."""
        try:
            mem_path = self.cache_dir / "memories.json"
            if mem_path.exists():
                with open(mem_path) as f:
                    data = json.load(f)
                for d in data:
                    mem = SignalMemory(d["signal"], d["adaptation"], d.get("effective", True))
                    mem.id = d.get("id", mem.id)
                    mem.timestamp = d.get("timestamp", mem.timestamp)
                    self.memories.append(mem)
                logger.info(f"Loaded {len(self.memories)} memories from cache")

            # Load FAISS index
            if FAISS_AVAILABLE:
                idx_path = self.cache_dir / "faiss.index"
                if idx_path.exists():
                    self.index = faiss.read_index(str(idx_path))
                    logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors)")
        except Exception as e:
            logger.warning(f"Failed to load RAG cache: {e}")
            self.memories = []
            self.index = None

    def stats(self) -> dict:
        return {
            "total_memories": len(self.memories),
            "faiss_available": FAISS_AVAILABLE and self.index is not None,
            "embedding_model": "all-MiniLM-L6-v2" if self.embedder.is_loaded() else "basic (fallback)",
            "index_vectors": self.index.ntotal if self.index else 0,
            "effective_rate": sum(1 for m in self.memories if m.effective) / max(len(self.memories), 1),
        }
