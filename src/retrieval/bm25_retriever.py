# src/retrieval/bm25_retriever.py
"""BM25 lexical top-k skill retrieval via the ``bm25s`` package.

Tokenizes each skill and the issue text, builds a sparse Okapi BM25 index,
and returns the top-k highest-scoring skills for the issue. No GPU or LLM
calls — tokenization and scoring are local, fast, and fully deterministic.

The skillbook index is built lazily on first ``retrieve()`` and cached in
memory keyed by skillbook content, so the common workload — one global
skillbook queried across many validation instances — only tokenizes and
indexes once. Concurrent calls share one retriever (iterate_repos), so the
cache and retrieve path are guarded by a lock.
"""

import hashlib
import threading
from typing import TYPE_CHECKING

from ace import Skillbook
from ace.core.skillbook import Skill
from loguru import logger

from .base import SkillRetrieverBase, extract_issue_info

if TYPE_CHECKING:
    # Imported lazily at runtime (see _get_bm25s) to keep this module importable
    # when bm25s is absent; the type import is for static analysis only.
    from bm25s import BM25 as _BM25Index


def _get_bm25s():
    """Import and return the ``bm25s`` module, raising a helpful error if absent."""
    try:
        import bm25s
    except ImportError as e:
        raise ImportError(
            "bm25s is required for BM25Retriever. "
            "Install with: pip install bm25s"
        ) from e
    return bm25s


def _skill_text(skill: Skill, *, include_section: bool = False) -> str:
    """Build indexable text from a Skill object."""
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


class BM25Retriever(SkillRetrieverBase):
    """BM25 (Okapi) lexical retriever via ``bm25s``.

    Tokenizes each skill and the issue, builds a sparse BM25 index, and
    returns the top-k highest-scoring skills.

    The index is cached in memory per skillbook (keyed by a content hash),
    so the common workload — one global skillbook queried across many
    instances — only tokenizes and indexes once.

    Args:
        top_k: Number of skills to retrieve.
        skip_threshold: Skip retrieval if skillbook has <= this many skills.
        k1: BM25 term-frequency saturation parameter (typical 1.2-2.0).
        b: BM25 length-normalization parameter (0..1, typical 0.75).
        include_section: Prepend skill section to indexed/query text.
    """

    def __init__(
        self,
        top_k: int = 5,
        skip_threshold: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
        include_section: bool = False,
    ) -> None:
        self.top_k = top_k
        self.skip_threshold = skip_threshold
        self.k1 = k1
        self.b = b
        self.model = "bm25"
        self._include_section = include_section
        self._index_cache: dict[str, _BM25Index] = {}
        self._lock = threading.Lock()

    def retrieve(self, skillbook: Skillbook, instance: dict) -> list[Skill]:
        """Return top-k skills by BM25 score against the issue.

        Returns all skills if count <= skip_threshold (no indexing).
        """
        skills = skillbook.skills()
        if not skills:
            return []

        if len(skills) <= self.skip_threshold:
            logger.debug(
                f"[BM25Retriever] {len(skills)} skills <= "
                f"skip_threshold={self.skip_threshold}, skipping retrieval"
            )
            return list(skills)

        bm25s = _get_bm25s()

        # Tokenize the query (per-call, no shared state) outside the lock.
        repo, title, body = extract_issue_info(instance)
        query_parts = []
        if self._include_section and repo:
            query_parts.append(repo)
        query_parts.append(title)
        query_parts.append(body)
        query_tokens = bm25s.tokenize("\n".join(query_parts), show_progress=False)

        book_hash = _skillbook_hash(skills, self._include_section)
        k = min(self.top_k, len(skills))

        # Build/cache index and retrieve under one lock: bm25s.BM25 is not
        # guaranteed thread-safe, and retrieval is cheap (numpy), so
        # serializing it across workers is negligible vs the agent/LLM cost.
        with self._lock:
            bm25 = self._index_cache.get(book_hash)
            if bm25 is None:
                texts = [
                    _skill_text(s, include_section=self._include_section) for s in skills
                ]
                corpus_tokens = bm25s.tokenize(texts, show_progress=False)
                bm25 = bm25s.BM25(k1=self.k1, b=self.b)
                bm25.index(corpus_tokens, show_progress=False)
                self._index_cache[book_hash] = bm25
                logger.debug(f"[BM25Retriever] Built index for {len(skills)} skills")

            results, scores = bm25.retrieve(query_tokens, k=k, show_progress=False)

        # results/scores have shape (1, k) for a single query.
        top_indices = results[0]
        selected = [skills[i] for i in top_indices if 0 <= i < len(skills)]
        logger.debug(
            f"[BM25Retriever] Selected {len(selected)}/{len(skills)} skills "
            f"(scores: {', '.join(f'{scores[0][j]:.3f}' for j in range(len(selected)))})"
        )
        return selected

    def get_config_summary(self) -> dict:
        """Return retriever config including BM25-specific fields."""
        base = super().get_config_summary()
        base["k1"] = self.k1
        base["b"] = self.b
        return base
