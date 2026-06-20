# src/tests/test_embedding_retriever.py
"""Tests for EmbeddingRetriever (mocked — no model or GPU required)."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _reset_shared_st_model_singleton():
    """Reset the module-level SentenceTransformer singleton between tests.

    Embedding tests mock ``_get_shared_st_model`` so the real globals are
    never populated today; this guards against order-dependent leaks if a
    future test forgets the patch.
    """
    import retrieval.embedding_retriever as _er
    _er._shared_model = None
    _er._shared_model_name = None
    yield
    _er._shared_model = None
    _er._shared_model_name = None


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
            device="cpu",
            top_k=7,
            skip_threshold=15,
            include_section=True,
            batch_size=16,
        )
        summary = retriever.get_config_summary()

        assert summary["type"] == "EmbeddingRetriever"
        assert summary["model"] == "Qwen/Qwen3-Embedding-4B"
        assert summary["top_k"] == 7
        assert summary["skip_threshold"] == 15
        assert summary["device"] == "cpu"
        assert summary["include_section"] is True
        assert summary["batch_size"] == 16


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


class TestEmbeddingRetrieverValidation:
    """Test constructor validation."""

    def test_zero_top_k_raises(self):
        from retrieval.embedding_retriever import EmbeddingRetriever
        with pytest.raises(ValueError, match="top_k"):
            EmbeddingRetriever(top_k=0)

    def test_negative_top_k_raises(self):
        from retrieval.embedding_retriever import EmbeddingRetriever
        with pytest.raises(ValueError, match="top_k"):
            EmbeddingRetriever(top_k=-1)


class TestEmbeddingRetrieverCacheRoundTrip:
    """Exercise the on-disk cache LOAD path (np.load round-trip)."""

    @patch("retrieval.embedding_retriever._get_shared_st_model")
    def test_second_retrieve_hits_cache(self, mock_get_model, tmp_path):
        from retrieval.embedding_retriever import EmbeddingRetriever

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        skill_embs = np.eye(10, dtype=np.float32)
        mock_model.encode.side_effect = [
            skill_embs,        # 1st retrieve: skill embeddings (cache miss)
            skill_embs[3:4],   # 1st retrieve: query embedding (cache miss)
        ]

        retriever = EmbeddingRetriever(
            model_name="test-model",
            top_k=3,
            skip_threshold=2,
            cache_dir=str(tmp_path / "cache"),
        )
        sb, _ = _make_skillbook(n_skills=10)

        result1 = retriever.retrieve(sb, _make_instance())
        result2 = retriever.retrieve(sb, _make_instance())  # loads from disk

        # Only the first retrieve touches the model; the second is a cache hit,
        # exercising the np.load round-trip of the .npz/.npy schema.
        assert mock_model.encode.call_count == 2
        assert [s.id for s in result1] == [s.id for s in result2]


class TestEmbeddingRetrieverRanking:
    """Ranking correctness beyond the symmetric one-hot case."""

    @patch("retrieval.embedding_retriever._get_shared_st_model")
    def test_nontrivial_ranking_order(self, mock_get_model, tmp_path):
        from retrieval.embedding_retriever import EmbeddingRetriever

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Non-orthogonal skill vectors; query is not a one-hot. Dot products
        # are s0=0.6, s1=0.8, s2=1.0 -> expected order 2, 1, 0. This would fail
        # if the matmul were transposed or the descending reversal dropped.
        skill_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
        query_vec = np.array([[0.6, 0.8]], dtype=np.float32)
        mock_model.encode.side_effect = [skill_embs, query_vec]

        retriever = EmbeddingRetriever(
            model_name="m", top_k=3, skip_threshold=2,
            cache_dir=str(tmp_path / "c"),
        )
        sb, _ = _make_skillbook(n_skills=3)

        result = retriever.retrieve(sb, _make_instance())
        assert [s.id for s in result] == ["skill_2", "skill_1", "skill_0"]


class TestEmbeddingRetrieverIncludeSection:
    """include_section must control whether the repo reaches the query text."""

    @staticmethod
    def _query_text(mock_model):
        # encode call #2 is the query; its first positional arg is [issue_text].
        return mock_model.encode.call_args_list[1].args[0][0]

    @patch("retrieval.embedding_retriever._get_shared_st_model")
    def test_repo_prepended_when_enabled(self, mock_get_model, tmp_path):
        from retrieval.embedding_retriever import EmbeddingRetriever
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        skill_embs = np.eye(4, dtype=np.float32)
        mock_model.encode.side_effect = [skill_embs, skill_embs[0:1]]
        retriever = EmbeddingRetriever(
            model_name="m", top_k=2, skip_threshold=2,
            include_section=True, cache_dir=str(tmp_path / "c"),
        )
        sb, _ = _make_skillbook(4)
        retriever.retrieve(sb, _make_instance())
        assert "django__django" in self._query_text(mock_model)

    @patch("retrieval.embedding_retriever._get_shared_st_model")
    def test_repo_absent_when_disabled(self, mock_get_model, tmp_path):
        from retrieval.embedding_retriever import EmbeddingRetriever
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        skill_embs = np.eye(4, dtype=np.float32)
        mock_model.encode.side_effect = [skill_embs, skill_embs[0:1]]
        retriever = EmbeddingRetriever(
            model_name="m", top_k=2, skip_threshold=2,
            include_section=False, cache_dir=str(tmp_path / "c"),
        )
        sb, _ = _make_skillbook(4)
        retriever.retrieve(sb, _make_instance())
        assert "django__django" not in self._query_text(mock_model)
