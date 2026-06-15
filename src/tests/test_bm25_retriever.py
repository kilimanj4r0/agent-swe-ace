# src/tests/test_bm25_retriever.py
"""Tests for BM25Retriever (real bm25s — no GPU or LLM required)."""

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
    """Create a mock Skillbook with n_skills lexically-distinct skills."""
    sb = Mock()
    skills = [
        _make_skill(f"skill_{i}", f"Section_{i}", f"unique topic number {i} details about widgets")
        for i in range(n_skills)
    ]
    sb.skills.return_value = skills
    return sb, skills


def _make_instance(problem_statement="Bug in migrations\nDetailed description here."):
    """Create a mock SWE-bench instance dict."""
    return {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "problem_statement": problem_statement,
    }


class TestBM25RetrieverSkipThreshold:
    """Test that retrieval is skipped when skill count <= threshold."""

    def test_skip_threshold_returns_all(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(top_k=3, skip_threshold=10)
        sb, skills = _make_skillbook(n_skills=5)

        result = retriever.retrieve(sb, _make_instance())
        assert result == skills
        assert len(result) == 5

    def test_empty_skillbook_returns_empty(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever()
        sb = Mock()
        sb.skills.return_value = []

        result = retriever.retrieve(sb, _make_instance())
        assert result == []


class TestBM25RetrieverSelection:
    """Test that BM25Retriever selects the correct number of skills."""

    def test_selects_top_k_skills(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(top_k=3, skip_threshold=2)
        sb, skills = _make_skillbook(n_skills=20)

        result = retriever.retrieve(sb, _make_instance("widgets topic details\nmore context"))

        assert len(result) == 3
        original_ids = {s.id for s in skills}
        for s in result:
            assert s.id in original_ids

    def test_no_duplicate_selections(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(top_k=5, skip_threshold=2)
        sb, _ = _make_skillbook(n_skills=20)

        result = retriever.retrieve(sb, _make_instance("widgets topic details"))

        ids = [s.id for s in result]
        assert len(ids) == len(set(ids))  # No duplicates

    def test_top_k_capped_at_skill_count(self):
        from retrieval.bm25_retriever import BM25Retriever

        # top_k=50 but only 8 skills (skip_threshold=2 so retrieval runs)
        retriever = BM25Retriever(top_k=50, skip_threshold=2)
        sb, _ = _make_skillbook(n_skills=8)

        result = retriever.retrieve(sb, _make_instance("widgets topic"))
        assert len(result) == 8  # All skills, capped by count


class TestBM25RetrieverRelevance:
    """Test that BM25 ranks lexically-relevant skills first."""

    def test_relevant_skill_ranks_first(self):
        from retrieval.bm25_retriever import BM25Retriever

        skills = [
            _make_skill("target", "Migrations", "fix migration bug in django orm models"),
            _make_skill("other1", "Performance", "optimize postgresql query performance indexing"),
            _make_skill("other2", "Testing", "pytest fixture setup and teardown"),
            _make_skill("other3", "Templates", "render html templates with context variables"),
            _make_skill("other4", "Logging", "configure logging handlers and formatters"),
        ]
        sb = Mock()
        sb.skills.return_value = skills

        retriever = BM25Retriever(top_k=3, skip_threshold=2)
        instance = _make_instance("Migration bug causes django to crash\nThe migration system is broken.")

        result = retriever.retrieve(sb, instance)
        assert result[0].id == "target"  # Strongest lexical overlap

    def test_include_section_changes_ranking(self):
        """Section text is indexed when include_section=True."""
        from retrieval.bm25_retriever import BM25Retriever

        # Same content, different sections; query matches one section's name.
        skills = [
            _make_skill("a", "Migrations", "generic advice about fixing issues"),
            _make_skill("b", "Templates", "generic advice about fixing issues"),
            _make_skill("c", "Caching", "generic advice about fixing issues"),
            _make_skill("d", "Signals", "generic advice about fixing issues"),
        ]
        sb = Mock()
        sb.skills.return_value = skills

        retriever = BM25Retriever(top_k=1, skip_threshold=2, include_section=True)
        instance = _make_instance("Templates rendering broken\nSomething about templates.")

        result = retriever.retrieve(sb, instance)
        assert result[0].id == "b"  # Section "Templates" matched


class TestBM25RetrieverReproducibility:
    """Test deterministic retrieval."""

    def test_same_config_same_result(self):
        from retrieval.bm25_retriever import BM25Retriever

        sb, _ = _make_skillbook(n_skills=20)

        r1 = BM25Retriever(top_k=5, skip_threshold=2)
        result1 = r1.retrieve(sb, _make_instance("widgets topic details number"))

        r2 = BM25Retriever(top_k=5, skip_threshold=2)
        result2 = r2.retrieve(sb, _make_instance("widgets topic details number"))

        assert [s.id for s in result1] == [s.id for s in result2]


class TestBM25RetrieverIndexCache:
    """Test the in-memory skillbook index cache."""

    def test_cache_reuses_index_for_same_skillbook(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(top_k=3, skip_threshold=2)
        sb, _ = _make_skillbook(n_skills=20)

        retriever.retrieve(sb, _make_instance("widgets topic one"))
        assert len(retriever._index_cache) == 1

        # Second retrieve with the same skillbook should NOT add a new index.
        retriever.retrieve(sb, _make_instance("different query text"))
        assert len(retriever._index_cache) == 1

    def test_cache_separates_different_skillbooks(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(top_k=3, skip_threshold=2)

        retriever.retrieve(_make_skillbook(n_skills=20)[0], _make_instance("widgets"))
        retriever.retrieve(
            _make_skillbook(n_skills=15)[0],  # Different size/content → different hash
            _make_instance("widgets"),
        )
        assert len(retriever._index_cache) == 2


class TestBM25RetrieverConfigSummary:
    """Test get_config_summary."""

    def test_config_summary_fields(self):
        from retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(top_k=7, skip_threshold=15, k1=1.4, b=0.8)
        summary = retriever.get_config_summary()

        assert summary["type"] == "BM25Retriever"
        assert summary["model"] == "bm25"
        assert summary["top_k"] == 7
        assert summary["skip_threshold"] == 15
        assert summary["k1"] == 1.4
        assert summary["b"] == 0.8
