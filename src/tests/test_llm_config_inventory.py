"""Golden-inventory checks for the LLM preset migration."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "src/tests/fixtures/llm_config_legacy_snapshot.json"


def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def test_legacy_snapshot_covers_every_config():
    data = _snapshot()
    assert data["base_config"] == "config.yaml"
    assert len(data["override_configs"]) == 68
    assert data["group_counts"] == {
        "configs/test.yaml": 1,
        "configs/test": 14,
        "configs/princeton-nlp__SWE-bench_Lite": 10,
        "configs/princeton-nlp__SWE-bench_Verified": 43,
    }
    assert len(data["effective_by_config"]) == 69


def test_legacy_snapshot_preserves_retrieval_distribution():
    data = _snapshot()
    assert data["retrieval_counts"] == {
        "disabled_llm": 37,
        "enabled_llm": 9,
        "enabled_bm25": 8,
        "enabled_embedding": 7,
        "enabled_random": 7,
    }


def test_legacy_snapshot_records_known_edge_cases():
    data = _snapshot()["effective_by_config"]
    hot = data[
        "configs/princeton-nlp__SWE-bench_Verified/"
        "agent-qwen3next-ace-qwen3next-full-global-split-default-025-ret-bm25.yaml"
    ]
    assert hot["llm.agent"]["temperature"] == 1.0
    assert hot["llm.ace"]["temperature"] == 1.0

    warm = data[
        "configs/princeton-nlp__SWE-bench_Verified/"
        "agent-qwen3-ace-qwen3-full-global-split-swe-025.yaml"
    ]
    assert warm["llm.agent"]["temperature"] == 0.7
    assert warm["llm.ace"]["temperature"] == 0.7

    ret = data["configs/test/11_retrieval_llm.yaml"]
    assert ret["retrieval"]["top_k"] == 1
    assert ret["retrieval"]["skip_threshold"] == 0
    assert ret["retrieval.llm"]["max_tokens"] == 2048


def test_legacy_snapshot_records_mixed_agent_ace_configs():
    data = _snapshot()
    assert data["mixed_agent_ace_configs"] == [
        "configs/princeton-nlp__SWE-bench_Lite/"
        "agent-qwen3-ace-qwen3next-full-4a-default.yaml",
        "configs/princeton-nlp__SWE-bench_Verified/"
        "agent-qwen3-ace-qwen3next-4a-retrieval-verified.yaml",
        "configs/princeton-nlp__SWE-bench_Verified/"
        "agent-qwen3-ace-qwen3next-full-4a-default.yaml",
    ]
