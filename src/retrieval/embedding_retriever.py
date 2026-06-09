# src/retrieval/embedding_retriever.py
"""Embedding-based top-k skill retrieval using cosine similarity.

Uses sentence-transformers to embed skills and the issue, then selects the
top-k most similar skills. Adapted from scripts/sample_top_skills_with_embs.py.
"""

import hashlib
import threading
from pathlib import Path

import numpy as np
from ace import Skillbook
from ace.core.skillbook import Skill
from loguru import logger

from .base import SkillRetrieverBase, extract_issue_info

# Module-level shared SentenceTransformer model and lock.
# Concurrent threads (iterate_repos) must share one model to avoid
# "Cannot copy out of meta tensor" errors from simultaneous model loading.
_shared_model = None
_shared_model_name = None
_shared_model_lock = threading.Lock()


def _get_shared_st_model(model_name: str, device: str):
    """Get or create the shared SentenceTransformer model (thread-safe)."""
    global _shared_model, _shared_model_name
    if _shared_model is not None and _shared_model_name == model_name:
        return _shared_model
    with _shared_model_lock:
        if _shared_model is not None and _shared_model_name == model_name:
            return _shared_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingRetriever. "
                "Install with: pip install sentence-transformers"
            )
        _shared_model = SentenceTransformer(
            model_name, device=device,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
        _shared_model_name = model_name
    return _shared_model


def _skill_text(skill: Skill, *, include_section: bool = False) -> str:
    """Build embedding text from a Skill object."""
    parts = []
    if include_section:
        parts.append(skill.section)
    parts.append(skill.content)
    return " ".join(parts)


def _skillbook_hash(skills: list[Skill], include_section: bool) -> str:
    """Deterministic hash of skillbook contents for cache keying."""
    h = hashlib.sha256()
    for s in sorted(skills, key=lambda sk: sk.id):
        h.update(s.id.encode())
        h.update(s.content.encode())
        if include_section:
            h.update(s.section.encode())
    return h.hexdigest()[:12]


def _query_hash(model_name: str, text: str) -> str:
    """Hash of model + query text for cache keying."""
    return hashlib.sha256(f"{model_name}\n{text}".encode()).hexdigest()[:12]


class EmbeddingRetriever(SkillRetrieverBase):
    """Embedding-based retriever using cosine similarity.

    Embeds skills and the issue with a sentence-transformers model,
    then selects top-k by dot-product similarity (equivalent to cosine
    on L2-normalized vectors).

    The SentenceTransformer model is loaded lazily on first ``retrieve()``
    call and shared across threads (module-level singleton).

    Args:
        model_name: Sentence-transformers model identifier.
        device: Torch device (``"cuda"`` or ``"cpu"``).
        top_k: Number of skills to retrieve.
        skip_threshold: Skip retrieval if skillbook has ≤ this many skills.
        include_section: Prepend skill section to embedding text.
        batch_size: Batch size for skill embedding.
        cache_dir: Directory for embedding cache files (None = ``data/.retrieval_cache/``).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "cuda",
        top_k: int = 5,
        skip_threshold: int = 10,
        include_section: bool = False,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        self.model = model_name
        self.top_k = top_k
        self.skip_threshold = skip_threshold
        self._device = device
        self._include_section = include_section
        self._batch_size = batch_size
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/.retrieval_cache")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, skillbook: Skillbook, instance: dict) -> list[Skill]:
        """Return top-k skills by cosine similarity to the issue.

        Returns all skills if count ≤ skip_threshold (no embedding computation).
        """
        skills = skillbook.skills()
        if not skills:
            return []

        if len(skills) <= self.skip_threshold:
            logger.debug(
                f"[EmbeddingRetriever] {len(skills)} skills ≤ "
                f"skip_threshold={self.skip_threshold}, skipping retrieval"
            )
            return list(skills)

        # Embed skills (with cache)
        skill_ids, skill_embs = self._embed_skills(skills)
        id_to_skill = {s.id: s for s in skills}

        # Embed query
        query_vec = self._embed_query(instance)

        # Cosine similarity via dot product (vectors are L2-normalized)
        scores = skill_embs @ query_vec  # (num_skills,)
        k = min(self.top_k, len(skills))
        top_indices = np.argsort(scores)[::-1][:k]

        selected = [id_to_skill[skill_ids[i]] for i in top_indices]
        logger.debug(
            f"[EmbeddingRetriever] Selected {len(selected)}/{len(skills)} skills "
            f"(scores: {', '.join(f'{scores[i]:.3f}' for i in top_indices)})"
        )
        return selected

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_skills(self, skills: list[Skill]) -> tuple[list[str], np.ndarray]:
        """Return (skill_ids, embeddings_matrix), using cache if available."""
        book_hash = _skillbook_hash(skills, self._include_section)
        cache = self._skill_cache_path(book_hash)

        if cache.exists():
            logger.debug(f"[EmbeddingRetriever] Loading cached skill embeddings from {cache}")
            data = np.load(cache, allow_pickle=False)
            return list(data["ids"]), data["embeddings"]

        logger.debug(f"[EmbeddingRetriever] Computing embeddings for {len(skills)} skills...")
        model = _get_shared_st_model(self.model, self._device)

        texts = [_skill_text(s, include_section=self._include_section) for s in skills]
        skill_ids = [s.id for s in skills]
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Cache to disk
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=skill_ids, embeddings=embeddings)
        logger.debug(f"[EmbeddingRetriever] Cached skill embeddings to {cache}")

        return skill_ids, embeddings

    def _embed_query(self, instance: dict) -> np.ndarray:
        """Embed the issue text from a SWE-bench instance dict."""
        repo, title, body = extract_issue_info(instance)

        parts = []
        if self._include_section and repo:
            parts.append(repo)
        parts.append(title)
        parts.append(body)
        issue_text = "\n".join(parts)

        qhash = _query_hash(self.model, issue_text)
        cache = self._query_cache_path(qhash)

        if cache.exists():
            return np.load(cache)

        model = _get_shared_st_model(self.model, self._device)
        vec = model.encode([issue_text], normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)[0]

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vec)
        return vec

    # ------------------------------------------------------------------
    # Cache paths
    # ------------------------------------------------------------------

    def _skill_cache_path(self, book_hash: str) -> Path:
        safe_model = self.model.replace("/", "__")
        suffix = "_wid" if self._include_section else "_noid"
        return self._cache_dir / f"skill_emb_{safe_model}_{book_hash}{suffix}.npz"

    def _query_cache_path(self, qhash: str) -> Path:
        safe_model = self.model.replace("/", "__")
        return self._cache_dir / f"query_emb_{safe_model}_{qhash}.npy"
