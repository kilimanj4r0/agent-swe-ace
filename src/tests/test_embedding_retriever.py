# src/tests/test_embedding_retriever.py
"""Tests for EmbeddingRetriever (mocked — no model or GPU required)."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
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


class TestEmbeddingRetrieverSkipThreshold:
    """Test that retrieval is skipped when skill count ≤ threshold."""

    def test_skip_threshold_returns_all(self):
        from retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever(top_k=3, skip_threshold=10)
        sb, skills = _make_skillbook(n_skills=5)

        result = retriever.retrieve(sb, _make_instance())
        assert result == skills
        assert len(result) == 5

    def test_empty_skillbook_returns_empty(self):
        from retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever()
        sb = Mock()
        sb.skills.return_value = []

        result = retriever.retrieve(sb, _make_instance())
        assert result == []


class TestEmbeddingRetrieverSelection:
    """Test retrieval with mocked sentence-transformers model."""

    @patch("retrieval.embedding_retriever._get_shared_st_model")
    def test_selects_top_k_skills(self, mock_get_model, tmp_path):
        from retrieval.embedding_retriever import EmbeddingRetriever

        # Mock SentenceTransformer model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Skill embeddings: 10 skills with 10-dim one-hot vectors
        # Make skill_7 most similar to the query
        skill_embs = np.eye(10, dtype=np.float32)
        # encode([text], ...) returns shape (1, dim) — so wrap query in batch dim
        mock_model.encode.side_effect = [
            skill_embs,                      # Skill embeddings: shape (10, 10)
            skill_embs[7:8],                 # Query embedding: shape (1, 10) → matches skill_7
        ]

        retriever = EmbeddingRetriever(
            model_name="test-model",
            top_k=3,
            skip_threshold=2,
            cache_dir=str(tmp_path / "cache"),
        )
        sb, _ = _make_skillbook(n_skills=10)

        result = retriever.retrieve(sb, _make_instance())

        assert len(result) == 3
        assert result[0].id == "skill_7"  # Highest similarity

    @patch("retrieval.embedding_retriever._get_shared_st_model")
    def test_top_k_capped_at_skill_count(self, mock_get_model, tmp_path):
        from retrieval.embedding_retriever import EmbeddingRetriever

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # 3 skills, top_k=50 → should return all 3
        skill_embs = np.eye(3, dtype=np.float32)
        mock_model.encode.side_effect = [
            skill_embs,           # Skill embeddings: shape (3, 3)
            skill_embs[0:1],      # Query embedding: shape (1, 3)
        ]

        retriever = EmbeddingRetriever(
            model_name="test-model",
            top_k=50,
            skip_threshold=2,
            cache_dir=str(tmp_path / "cache"),
        )
        sb, _ = _make_skillbook(n_skills=3)

        result = retriever.retrieve(sb, _make_instance())
        assert len(result) == 3


class TestEmbeddingRetrieverConfigSummary:
    """Test get_config_summary."""

    def test_config_summary_fields(self):
        from retrieval.embedding_retriever import EmbeddingRetriever

        retriever = EmbeddingRetriever(
            model_name="Qwen/Qwen3-Embedding-4B",
            top_k=7,
            skip_threshold=15,
        )
        summary = retriever.get_config_summary()

        assert summary["type"] == "EmbeddingRetriever"
        assert summary["model"] == "Qwen/Qwen3-Embedding-4B"
        assert summary["top_k"] == 7
        assert summary["skip_threshold"] == 15


class TestEmbeddingRetrieverCache:
    """Test embedding cache key computation."""

    def test_skillbook_hash_deterministic(self):
        from retrieval.embedding_retriever import _skillbook_hash

        skills = [_make_skill("b", content="second"), _make_skill("a", content="first")]
        h1 = _skillbook_hash(skills, include_section=False)
        h2 = _skillbook_hash(skills, include_section=False)
        assert h1 == h2

    def test_skillbook_hash_changes_with_content(self):
        from retrieval.embedding_retriever import _skillbook_hash

        skills_a = [_make_skill("s1", content="alpha")]
        skills_b = [_make_skill("s1", content="beta")]
        h_a = _skillbook_hash(skills_a, include_section=False)
        h_b = _skillbook_hash(skills_b, include_section=False)
        assert h_a != h_b

    def test_skillbook_hash_changes_with_include_section(self):
        from retrieval.embedding_retriever import _skillbook_hash

        skills = [_make_skill("s1", section="Testing", content="content")]
        h_no_section = _skillbook_hash(skills, include_section=False)
        h_with_section = _skillbook_hash(skills, include_section=True)
        assert h_no_section != h_with_section
