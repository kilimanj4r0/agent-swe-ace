# src/tests/test_skill_retriever.py
"""Tests for top-k skill retrieval."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_skill(skill_id="s1", section="Testing", content="Test skill content",
                justification=None, evidence=None):
    """Create a mock Skill object."""
    skill = Mock()
    skill.id = skill_id
    skill.section = section
    skill.content = content
    skill.justification = justification
    skill.evidence = evidence
    return skill


def _make_skillbook(n_skills=5):
    """Create a mock Skillbook with n_skills skills."""
    sb = Mock()
    skills = [_make_skill(f"skill_{i}", f"Section_{i}", f"Content for skill {i}")
              for i in range(n_skills)]
    sb.skills.return_value = skills
    sb.add_skill = Mock()
    return sb, skills


def _make_instance(repo="django/django", problem="Bug in migrations\nDetailed description here."):
    """Create a mock SWE-bench instance dict."""
    return {
        "instance_id": "django__django-12345",
        "repo": repo,
        "problem_statement": problem,
    }


class TestSkillRetrieverSkipThreshold:
    """Test that retrieval is skipped when skill count ≤ threshold."""

    def test_skip_threshold_returns_all(self):
        from retrieval.skill_retriever import SkillRetriever

        retriever = SkillRetriever(
            model="test-model",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            top_k=5,
            skip_threshold=10,
        )
        sb, skills = _make_skillbook(n_skills=5)
        instance = _make_instance()

        result = retriever.retrieve(sb, instance)

        assert result == skills
        assert len(result) == 5

    def test_empty_skillbook_returns_empty(self):
        from retrieval.skill_retriever import SkillRetriever

        retriever = SkillRetriever(
            model="test-model",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
        )
        sb = Mock()
        sb.skills.return_value = []

        result = retriever.retrieve(sb, _make_instance())

        assert result == []


class TestSkillRetrieverFilterFallback:
    """Test graceful fallback when filter returns nothing."""

    @patch("retrieval.skill_retriever.OpenAI")
    def test_filter_returns_zero_falls_back_to_all(self, mock_openai_cls):
        from retrieval.skill_retriever import SkillRetriever

        # Mock LLM to return empty indices every time
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({"relevant_indices": []})
        mock_client.chat.completions.create.return_value = mock_resp

        retriever = SkillRetriever(
            model="test-model",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            skip_threshold=2,  # 5 skills > 2, so retrieval runs
            max_retries=1,
        )
        sb, skills = _make_skillbook(n_skills=5)

        result = retriever.retrieve(sb, _make_instance())

        # Falls back to all skills when filter fails
        assert len(result) == 5


class TestSkillRetrieverRankFewerThanK:
    """Test when rank returns fewer skills than requested k."""

    @patch("retrieval.skill_retriever.OpenAI")
    def test_rank_returns_3_of_5(self, mock_openai_cls):
        from retrieval.skill_retriever import SkillRetriever

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Filter returns indices 0, 1, 2, 3, 4 (all pass)
        filter_resp = MagicMock()
        filter_resp.choices[0].message.content = json.dumps(
            {"relevant_indices": [0, 1, 2, 3, 4]}
        )
        # Rank returns only 3 of 5 (indices are relative to filtered list)
        rank_resp = MagicMock()
        rank_resp.choices[0].message.content = json.dumps({
            "selections": [
                {"idx": 0, "reason": "relevant"},
                {"idx": 2, "reason": "useful"},
                {"idx": 4, "reason": "important"},
            ]
        })

        mock_client.chat.completions.create.side_effect = [filter_resp, rank_resp]

        retriever = SkillRetriever(
            model="test-model",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            top_k=3,  # Request 3 so rank stage runs (5 filtered > 3)
            skip_threshold=2,
            max_retries=1,
        )
        sb, skills = _make_skillbook(n_skills=5)

        result = retriever.retrieve(sb, _make_instance())

        assert len(result) == 3
        assert result[0].id == "skill_0"
        assert result[1].id == "skill_2"
        assert result[2].id == "skill_4"


class TestSkillRetrieverSuccess:
    """Test successful two-stage retrieval."""

    @patch("retrieval.skill_retriever.OpenAI")
    def test_retrieve_selects_top_k(self, mock_openai_cls):
        from retrieval.skill_retriever import SkillRetriever

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Filter keeps indices 0, 2, 4, 6, 8 → filtered list has 5 items
        filter_resp = MagicMock()
        filter_resp.choices[0].message.content = json.dumps(
            {"relevant_indices": [0, 2, 4, 6, 8]}
        )
        # Rank picks top 3 from filtered list (indices 0-4 in filtered list)
        # filtered[0]=skill_0, filtered[1]=skill_2, filtered[2]=skill_4,
        # filtered[3]=skill_6, filtered[4]=skill_8
        rank_resp = MagicMock()
        rank_resp.choices[0].message.content = json.dumps({
            "selections": [
                {"idx": 0, "reason": "directly relevant"},   # skill_0
                {"idx": 2, "reason": "testing pattern"},     # skill_4
                {"idx": 4, "reason": "cautionary note"},     # skill_8
            ]
        })

        mock_client.chat.completions.create.side_effect = [filter_resp, rank_resp]

        retriever = SkillRetriever(
            model="test-model",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            top_k=3,
            skip_threshold=5,
            max_retries=1,
        )
        sb, skills = _make_skillbook(n_skills=10)

        result = retriever.retrieve(sb, _make_instance())

        assert len(result) == 3
        assert result[0].id == "skill_0"
        assert result[1].id == "skill_4"
        assert result[2].id == "skill_8"


class TestCustomPromptsFromFile:
    """Test that custom prompt files override defaults."""

    def test_load_prompt_from_file(self, tmp_path):
        from retrieval.skill_retriever import _load_prompt

        custom = "Custom prompt with {repo} and {title}"
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text(custom)

        result = _load_prompt(str(prompt_file), "default prompt")
        assert result == custom

    def test_load_prompt_returns_default_when_none(self):
        from retrieval.skill_retriever import _load_prompt

        result = _load_prompt(None, "default prompt")
        assert result == "default prompt"


class TestPredictPhaseWithRetriever:
    """Test PredictPhase integration with skill retriever."""

    def test_retrieval_on_val_phase(self, tmp_path):
        """Retrieval triggers on phase='val' (val skillbook pass)."""
        from agents.miniswe_agent import AgentResult
        from phases.predict import PredictPhase

        mock_skillbook = Mock()
        skills = [_make_skill(f"s{i}", f"Sec{i}", f"Content {i}") for i in range(15)]
        mock_skillbook.skills.return_value = skills

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = skills[:3]
        mock_retriever.skip_threshold = 10
        mock_retriever.model = "test-model"

        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="Submitted", patch="diff --git",
            trajectory=[{"role": "user", "content": "test"}], error=None,
        )

        phase = PredictPhase(
            agent=mock_agent, output_dir=tmp_path,
            skill_retriever=mock_retriever,
        )

        phase.run(instance=_make_instance(), skillbook=mock_skillbook, iteration=0, phase="val")

        mock_retriever.retrieve.assert_called_once()

    def test_retrieval_on_single_phase(self, tmp_path):
        """Retrieval triggers when phase=None (single-phase / per_instance)."""
        from agents.miniswe_agent import AgentResult
        from phases.predict import PredictPhase

        mock_skillbook = Mock()
        skills = [_make_skill(f"s{i}") for i in range(15)]
        mock_skillbook.skills.return_value = skills

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = skills[:3]
        mock_retriever.skip_threshold = 10
        mock_retriever.model = "test-model"

        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="Submitted", patch="diff --git", trajectory=[], error=None,
        )

        phase = PredictPhase(
            agent=mock_agent, output_dir=tmp_path,
            skill_retriever=mock_retriever,
        )

        # phase=None (default) → single-phase mode
        phase.run(instance=_make_instance(), skillbook=mock_skillbook, iteration=0)
        assert mock_retriever.retrieve.call_count == 1

    def test_retrieval_skipped_on_train_phase(self, tmp_path):
        """Retrieval skipped on phase='train'."""
        from agents.miniswe_agent import AgentResult
        from phases.predict import PredictPhase

        mock_skillbook = Mock()
        skills = [_make_skill(f"s{i}") for i in range(15)]
        mock_skillbook.skills.return_value = skills

        mock_retriever = Mock()

        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="Submitted", patch="diff --git", trajectory=[], error=None,
        )

        phase = PredictPhase(
            agent=mock_agent, output_dir=tmp_path,
            skill_retriever=mock_retriever,
        )

        phase.run(instance=_make_instance(), skillbook=mock_skillbook, iteration=0, phase="train")
        mock_retriever.retrieve.assert_not_called()

    def test_retrieval_skipped_on_val_baseline_phase(self, tmp_path):
        """Retrieval skipped on phase='val_baseline'."""
        from agents.miniswe_agent import AgentResult
        from phases.predict import PredictPhase

        mock_skillbook = Mock()
        skills = [_make_skill(f"s{i}") for i in range(15)]
        mock_skillbook.skills.return_value = skills

        mock_retriever = Mock()

        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="Submitted", patch="diff --git", trajectory=[], error=None,
        )

        phase = PredictPhase(
            agent=mock_agent, output_dir=tmp_path,
            skill_retriever=mock_retriever,
        )

        phase.run(instance=_make_instance(), skillbook=mock_skillbook, iteration=0, phase="val_baseline")
        mock_retriever.retrieve.assert_not_called()

    def test_no_retriever_passes_through(self, tmp_path):
        from agents.miniswe_agent import AgentResult
        from phases.predict import PredictPhase

        mock_skillbook = Mock()
        skills = [_make_skill(f"s{i}") for i in range(5)]
        mock_skillbook.skills.return_value = skills

        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="Submitted", patch="diff --git", trajectory=[], error=None,
        )

        phase = PredictPhase(agent=mock_agent, output_dir=tmp_path)
        phase.run(instance=_make_instance(), skillbook=mock_skillbook, iteration=0)

        agent_call_args = mock_agent.run.call_args
        passed_sb = agent_call_args.kwargs.get("skillbook") or agent_call_args[1].get("skillbook")
        assert passed_sb is mock_skillbook


def _retrieval_llm_section(**overrides):
    effective = {
        "provider": "hosted_vllm",
        "model": "Qwen/test",
        "api_base": "http://localhost:8800/v1",
        "api_key_env": "HOSTED_VLLM_API_KEY",
        "temperature": 0.0,
        "max_tokens": 2048,
        "extra_kwargs": {},
    }
    effective.update(overrides)
    return {
        "preset": "test-preset",
        "overrides": {},
        "effective": effective,
    }


class TestBuildSkillRetriever:
    """Test _build_skill_retriever config wiring."""

    def test_disabled_returns_none(self):
        from cli.commands import _build_skill_retriever

        result = _build_skill_retriever({"skillbook": {"retrieval": {"enabled": False}}})
        assert result is None

    def test_no_config_returns_none(self):
        from cli.commands import _build_skill_retriever

        result = _build_skill_retriever({})
        assert result is None

    @patch.dict("os.environ", {"HOSTED_VLLM_API_KEY": "test-key-123"})
    @patch("cli.commands.SkillRetriever")
    def test_enabled_uses_nested_effective_llm(self, retriever_cls):
        from cli.commands import _build_skill_retriever

        result = _build_skill_retriever(
            {
                "skillbook": {
                    "retrieval": {
                        "enabled": True,
                        "llm": _retrieval_llm_section(
                            temperature=0.4,
                            max_tokens=3072,
                        ),
                        "top_k": 3,
                        "skip_threshold": 7,
                        "filter_prompt": "filter.txt",
                        "rank_prompt": "rank.txt",
                        "chunk_size": 50,
                        "filter_target": 25,
                    }
                }
            }
        )

        assert result is retriever_cls.return_value
        retriever_cls.assert_called_once_with(
            model="Qwen/test",
            api_base="http://localhost:8800/v1",
            api_key="test-key-123",
            top_k=3,
            skip_threshold=7,
            filter_prompt="filter.txt",
            rank_prompt="rank.txt",
            chunk_size=50,
            filter_target=25,
            temperature=0.4,
            max_tokens=3072,
        )


class TestBuildSkillRetrieverDispatch:
    """Test _build_skill_retriever dispatches by type field."""

    def test_type_random_returns_random_retriever(self):
        from cli.commands import _build_skill_retriever
        from retrieval.random_retriever import RandomRetriever

        result = _build_skill_retriever({
            "skillbook": {
                "retrieval": {
                    "enabled": True,
                    "type": "random",
                    "top_k": 3,
                    "skip_threshold": 7,
                    "seed": 42,
                }
            }
        })

        assert isinstance(result, RandomRetriever)
        assert result.top_k == 3
        assert result.skip_threshold == 7

    def test_type_embedding_returns_embedding_retriever(self):
        from cli.commands import _build_skill_retriever
        from retrieval.embedding_retriever import EmbeddingRetriever

        result = _build_skill_retriever({
            "skillbook": {
                "retrieval": {
                    "enabled": True,
                    "type": "embedding",
                    "model": "test-emb-model",
                    "top_k": 5,
                    "skip_threshold": 3,
                    "device": "cpu",
                }
            }
        })

        assert isinstance(result, EmbeddingRetriever)
        assert result.top_k == 5
        assert result.skip_threshold == 3
        assert result.model == "test-emb-model"

    def test_type_bm25_returns_bm25_retriever(self):
        from cli.commands import _build_skill_retriever
        from retrieval.bm25_retriever import BM25Retriever

        result = _build_skill_retriever({
            "skillbook": {
                "retrieval": {
                    "enabled": True,
                    "type": "bm25",
                    "top_k": 4,
                    "skip_threshold": 8,
                    "k1": 1.2,
                    "b": 0.5,
                    "include_section": True,
                }
            }
        })

        assert isinstance(result, BM25Retriever)
        assert result.top_k == 4
        assert result.skip_threshold == 8
        assert result.k1 == 1.2
        assert result.b == 0.5

    def test_type_llm_default_no_type_field(self):
        """When type is not specified, defaults to 'llm'."""
        from cli.commands import _build_skill_retriever
        from retrieval.skill_retriever import SkillRetriever

        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key-123"}):
            result = _build_skill_retriever({
                "skillbook": {
                    "retrieval": {
                        "enabled": True,
                        "llm": _retrieval_llm_section(
                            api_key_env="ZAI_API_KEY",
                        ),
                        "top_k": 3,
                    }
                }
            })

        assert isinstance(result, SkillRetriever)
        assert result.top_k == 3

    def test_unknown_type_raises_error(self):
        from cli.commands import _build_skill_retriever

        with pytest.raises(ValueError, match="Unknown retriever type"):
            _build_skill_retriever({
                "skillbook": {
                    "retrieval": {
                        "enabled": True,
                        "type": "nonexistent",
                    }
                }
            })

    def test_type_llm_explicit(self):
        """Explicit type: 'llm' behaves the same as default."""
        from cli.commands import _build_skill_retriever
        from retrieval.skill_retriever import SkillRetriever

        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key-123"}):
            result = _build_skill_retriever({
                "skillbook": {
                    "retrieval": {
                        "enabled": True,
                        "type": "llm",
                        "llm": _retrieval_llm_section(
                            api_key_env="ZAI_API_KEY",
                        ),
                    }
                }
            })

        assert isinstance(result, SkillRetriever)


class TestSkillRetrieverConfigSummary:
    """Test SkillRetriever.get_config_summary keys (statistics contract)."""

    def test_config_summary_fields(self):
        from retrieval.skill_retriever import SkillRetriever

        retriever = SkillRetriever(
            model="glm-4.5-flash",
            api_base="https://api.example.com/v1",
            api_key="k",
            top_k=6,
            skip_threshold=12,
            chunk_size=150,
            filter_target=80,
        )
        summary = retriever.get_config_summary()

        assert summary["type"] == "SkillRetriever"
        assert summary["model"] == "glm-4.5-flash"
        assert summary["top_k"] == 6
        assert summary["skip_threshold"] == 12
        assert summary["filter_target"] == 80
        assert summary["chunk_size"] == 150
