# src/tests/test_retrieval_val_pass.py
"""Retrieval in the val (frozen) pass must happen once per instance, so all k
attempts share the same retrieved skillbook — not once per attempt.

Background: val_pass_k runs k attempts per instance to measure pass@k of *the agent
with a fixed retrieved skillbook*. Re-running retrieval per attempt (the old behavior)
re-randomized the random retriever each attempt and redundantly recomputed the
deterministic ones, conflating retrieval variance with agent sampling variance.
"""
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from ace import Skillbook


def _big_skillbook(n=20):
    """Skillbook with n skills (> skip_threshold so retrieval actually fires)."""
    sb = Skillbook()
    for i in range(n):
        sb.add_skill(section="fixing", content=f"skill content {i}")
    return sb


class TestRetrievalOncePerInstance:
    def test_frozen_val_pass_retrieves_once_across_k_attempts(self, tmp_path):
        from runners.main_loop import ExperimentLoop
        from phases.predict import PredictPhase

        sb = _big_skillbook(20)

        # Spy retriever: returns a fixed 2-skill subset on each call.
        retriever = Mock()
        retriever.skip_threshold = 0
        retriever.top_k = 2
        retriever.retrieve.return_value = sb.skills()[:2]
        retriever.get_config_summary.return_value = {
            "type": "MockRetriever", "model": "mock", "top_k": 2, "skip_threshold": 0,
        }

        agent = Mock()
        agent.run.return_value = Mock(
            exit_status="submitted", patch="p", trajectory=[], error=None
        )

        predict = PredictPhase(
            agent=agent, output_dir=tmp_path, run_name="t",
            benchmark="princeton-nlp__SWE-bench_Verified", skill_retriever=retriever,
        )

        evaluate = Mock()
        evaluate.run.return_value = Mock(
            instance_id="repo__i-1", resolved=False, feedback="", metrics={},
        )

        loop = ExperimentLoop(
            predict_phase=predict, evaluate_phase=evaluate, learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1,
        )

        instance = {"instance_id": "repo__i-1", "problem_statement": "Fix it"}
        loop.run_instance(
            instance, initial_skillbook=sb, frozen_skillbook=True,
            max_attempts_override=5, phase="val", skip_baseline_reuse=True,
        )

        # Retrieval must run exactly ONCE for the instance (frozen skillbook shared
        # across all 5 attempts), not once per attempt.
        assert retriever.retrieve.call_count == 1, (
            f"expected retrieval once per instance, got {retriever.retrieve.call_count}"
        )
        # All 5 attempts still execute.
        assert agent.run.call_count == 5
        # ...and they all received the IDENTICAL narrowed skillbook (2 skills),
        # i.e. true "retrieve once -> 5 identical attempts".
        used = [c.kwargs["skillbook"] for c in agent.run.call_args_list]
        assert len(used) == 5
        assert all(sb is used[0] for sb in used), "attempts used different skillbooks"
        assert len(used[0].skills()) == 2, "skillbook not narrowed to retrieved subset"


class TestWorkerPredictForwardsRetriever:
    """Regression for the concurrency-gated retrieval no-op (commit b77f430).

    _make_worker_predict() rebuilds a fresh PredictPhase per concurrent worker.
    It used to omit skill_retriever, so every run with concurrency>1 executed the
    val/run pass with skill_retriever=None. prepare_skillbook then bailed on its
    first guard and the FULL skillbook was fed to the agent (instances_retrieved=0).
    The retriever is designed to be shared across worker threads, so it must be
    forwarded to each worker.
    """

    def test_make_worker_predict_forwards_skill_retriever(self, tmp_path):
        from runners.main_loop import ExperimentLoop
        from phases.predict import PredictPhase

        retriever = Mock()
        retriever.skip_threshold = 0
        agent = Mock()

        predict = PredictPhase(
            agent=agent, output_dir=tmp_path, run_name="t",
            benchmark="princeton-nlp__SWE-bench_Verified", skill_retriever=retriever,
        )
        loop = ExperimentLoop(
            predict_phase=predict, evaluate_phase=Mock(), learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1,
            agent_factory=lambda: Mock(),
        )

        worker = loop._make_worker_predict()
        assert worker.skill_retriever is retriever, (
            "_make_worker_predict dropped skill_retriever -> concurrent (concurrency>1) "
            "runs silently skip retrieval and feed the full skillbook"
        )
