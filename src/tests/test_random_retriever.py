# src/tests/test_random_retriever.py
"""Tests for RandomRetriever."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_skill(skill_id="s1", section="Testing", content="Test skill content"):
    """Create a mock Skill object."""
    skill = Mock()
    skill.id = skill_id
    skill.section = section
    skill.content = content
    return skill


def _make_skillbook(n_skills=5):
    """Create a mock Skillbook with n_skills skills."""
    sb = Mock()
    skills = [_make_skill(f"skill_{i}", f"Section_{i}", f"Content for skill {i}")
              for i in range(n_skills)]
    sb.skills.return_value = skills
    return sb, skills


def _make_instance():
    """Create a mock SWE-bench instance dict."""
    return {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "problem_statement": "Bug in migrations\nDetailed description here.",
    }


class TestRandomRetrieverSkipThreshold:
    """Test that retrieval is skipped when skill count ≤ threshold."""

    def test_skip_threshold_returns_all(self):
        from retrieval.random_retriever import RandomRetriever

        retriever = RandomRetriever(top_k=3, skip_threshold=10)
        sb, skills = _make_skillbook(n_skills=5)
        result = retriever.retrieve(sb, _make_instance())

        assert result == skills
        assert len(result) == 5

    def test_empty_skillbook_returns_empty(self):
        from retrieval.random_retriever import RandomRetriever

        retriever = RandomRetriever()
        sb = Mock()
        sb.skills.return_value = []

        result = retriever.retrieve(sb, _make_instance())
        assert result == []


class TestRandomRetrieverSelection:
    """Test that random retriever selects correct number of skills."""

    def test_selects_top_k_skills(self):
        from retrieval.random_retriever import RandomRetriever

        retriever = RandomRetriever(top_k=3, skip_threshold=5, seed=42)
        sb, skills = _make_skillbook(n_skills=20)

        result = retriever.retrieve(sb, _make_instance())

        assert len(result) == 3
        # All selected skills must be from the original set
        original_ids = {s.id for s in skills}
        for s in result:
            assert s.id in original_ids

    def test_no_duplicate_selections(self):
        from retrieval.random_retriever import RandomRetriever

        retriever = RandomRetriever(top_k=5, skip_threshold=2, seed=42)
        sb, _ = _make_skillbook(n_skills=20)

        result = retriever.retrieve(sb, _make_instance())

        ids = [s.id for s in result]
        assert len(ids) == len(set(ids))  # No duplicates

    def test_top_k_capped_at_skill_count(self):
        from retrieval.random_retriever import RandomRetriever

        # top_k=50 but only 8 skills (skip_threshold=2 so retrieval runs)
        retriever = RandomRetriever(top_k=50, skip_threshold=2, seed=42)
        sb, _ = _make_skillbook(n_skills=8)

        result = retriever.retrieve(sb, _make_instance())
        assert len(result) == 8  # All skills, capped by count


class TestRandomRetrieverReproducibility:
    """Test that seed produces deterministic results."""

    def test_same_seed_same_result(self):
        from retrieval.random_retriever import RandomRetriever

        sb, _ = _make_skillbook(n_skills=20)

        retriever1 = RandomRetriever(top_k=5, skip_threshold=2, seed=123)
        result1 = retriever1.retrieve(sb, _make_instance())

        retriever2 = RandomRetriever(top_k=5, skip_threshold=2, seed=123)
        result2 = retriever2.retrieve(sb, _make_instance())

        ids1 = [s.id for s in result1]
        ids2 = [s.id for s in result2]
        assert ids1 == ids2

    def test_different_seed_different_result(self):
        from retrieval.random_retriever import RandomRetriever

        sb, _ = _make_skillbook(n_skills=20)

        retriever1 = RandomRetriever(top_k=5, skip_threshold=2, seed=1)
        result1 = retriever1.retrieve(sb, _make_instance())

        retriever2 = RandomRetriever(top_k=5, skip_threshold=2, seed=2)
        result2 = retriever2.retrieve(sb, _make_instance())

        ids1 = [s.id for s in result1]
        ids2 = [s.id for s in result2]
        assert ids1 != ids2  # Extremely unlikely to match


class TestRandomRetrieverConfigSummary:
    """Test get_config_summary."""

    def test_config_summary_fields(self):
        from retrieval.random_retriever import RandomRetriever

        retriever = RandomRetriever(top_k=7, skip_threshold=15, seed=42)
        summary = retriever.get_config_summary()

        assert summary["type"] == "RandomRetriever"
        assert summary["model"] == "random"
        assert summary["top_k"] == 7
        assert summary["skip_threshold"] == 15


class TestRandomRetrieverValidation:
    """Test constructor validation."""

    def test_zero_top_k_raises(self):
        from retrieval.random_retriever import RandomRetriever
        with pytest.raises(ValueError, match="top_k"):
            RandomRetriever(top_k=0)

    def test_negative_top_k_raises(self):
        from retrieval.random_retriever import RandomRetriever
        with pytest.raises(ValueError, match="top_k"):
            RandomRetriever(top_k=-1)
