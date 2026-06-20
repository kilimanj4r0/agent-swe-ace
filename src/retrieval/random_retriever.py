# src/retrieval/random_retriever.py
"""Random baseline retriever — picks k random skills, no LLM calls."""

import random

from ace import Skillbook
from ace.core.skillbook import Skill
from loguru import logger

from .base import SkillRetrieverBase


class RandomRetriever(SkillRetrieverBase):
    """Random baseline retriever.

    Selects k skills uniformly at random from the skillbook.
    Useful as a baseline to measure whether structured retrieval adds value.

    Args:
        top_k: Number of skills to retrieve.
        skip_threshold: Skip retrieval if skillbook has ≤ this many skills.
        seed: Random seed for reproducibility (None = non-deterministic).
    """

    def __init__(
        self,
        top_k: int = 5,
        skip_threshold: int = 10,
        seed: int | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self.skip_threshold = skip_threshold
        self.model = "random"
        self._rng = random.Random(seed)

    def retrieve(self, skillbook: Skillbook, instance: dict) -> list[Skill]:
        """Return k random skills from the skillbook.

        Returns all skills if count ≤ skip_threshold (no sampling).
        """
        skills = skillbook.skills()
        if not skills:
            return []

        if len(skills) <= self.skip_threshold:
            logger.debug(
                f"[RandomRetriever] {len(skills)} skills ≤ "
                f"skip_threshold={self.skip_threshold}, skipping retrieval"
            )
            return list(skills)

        k = min(self.top_k, len(skills))
        selected = self._rng.sample(skills, k)
        logger.debug(f"[RandomRetriever] Sampled {len(selected)}/{len(skills)} skills")
        return selected
