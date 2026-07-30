"""Golden-inventory checks for the LLM preset migration."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "src/tests/fixtures/llm_config_legacy_snapshot.json"
sys.path.insert(0, str(ROOT / "scripts"))

VERIFIED_QWEN3_CONFIGS = (
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-4a-default.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-4a-swe.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-default-025-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-default-025-ret-embedding.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-default-025-ret-random.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-default-025-ret.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-default-025.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-default.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-swe-025-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-swe-025-ret-embedding.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-swe-025-ret-random.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-swe-025-ret.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-full-global-split-swe-025.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-default-025-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-default-025-ret-embedding.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-default-025-ret-random.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-default-025-ret.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-default-025.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-default.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-swe-025-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-swe-025-ret-embedding.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-swe-025-ret-random.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-swe-025-ret.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-swe-025.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-repos-split-swe.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-full-4a-no-skillbook.yaml",
)
REMAINING_VERIFIED_CONFIGS = (
    "configs/princeton-nlp__SWE-bench_Verified/agent-ace-qwen3-16opus45-distil-repos-split-default.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-ace-qwen3-16opus45-distil-repos-split-swe.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3next-4a-retrieval-verified.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3next-full-4a-default.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025-eval-on-train-noret.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025-eval-on-train-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025-ret-embedding.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025-ret-llm.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025-ret-random.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-global-split-default-025.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-repos-split-default-025-ret-bm25.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-repos-split-default-025-ret-embedding.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-repos-split-default-025-ret-llm.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-repos-split-default-025-ret-random.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/agent-qwen3next-ace-qwen3next-full-repos-split-default-025.yaml",
    "configs/princeton-nlp__SWE-bench_Verified/val-retrieval-distil-repos-split.yaml",
)

from llm_config_inventory import (  # noqa: E402
    build_current_snapshot,
    compare_current_to_golden,
    validate_commented_aliases,
)


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


def test_migrated_test_configs_match_legacy_snapshot():
    compare_current_to_golden(
        ROOT,
        SNAPSHOT,
        include_prefixes=("configs/test.yaml", "configs/test/"),
    )


def test_migrated_test_config_comments_use_catalog_aliases():
    validate_commented_aliases(
        ROOT,
        include_prefixes=("configs/test.yaml", "configs/test/"),
    )


def test_migrated_lite_configs_match_legacy_snapshot():
    compare_current_to_golden(
        ROOT,
        SNAPSHOT,
        include_prefixes=("configs/princeton-nlp__SWE-bench_Lite/",),
    )


def test_migrated_verified_qwen3_configs_match_legacy_snapshot():
    assert len(VERIFIED_QWEN3_CONFIGS) == 26
    compare_current_to_golden(
        ROOT,
        SNAPSHOT,
        include_prefixes=VERIFIED_QWEN3_CONFIGS,
    )


def test_migrated_verified_qwen3_comments_use_catalog_aliases():
    validate_commented_aliases(
        ROOT,
        include_prefixes=VERIFIED_QWEN3_CONFIGS,
    )


def test_migrated_remaining_verified_configs_match_legacy_snapshot():
    assert len(REMAINING_VERIFIED_CONFIGS) == 17
    compare_current_to_golden(
        ROOT,
        SNAPSHOT,
        include_prefixes=REMAINING_VERIFIED_CONFIGS,
    )


def test_verified_groups_cover_every_golden_verified_config():
    golden_verified = {
        path
        for path in _snapshot()["override_configs"]
        if path.startswith("configs/princeton-nlp__SWE-bench_Verified/")
    }
    assert set(VERIFIED_QWEN3_CONFIGS) | set(REMAINING_VERIFIED_CONFIGS) == (
        golden_verified
    )


def test_all_migrated_configs_match_legacy_snapshot():
    compare_current_to_golden(ROOT, SNAPSHOT)


def test_all_current_configs_preserve_inventory_shape():
    current = build_current_snapshot(ROOT)
    golden = _snapshot()
    assert len(current["effective_by_config"]) == 69
    assert len(current["override_configs"]) == 68
    assert current["retrieval_counts"] == golden["retrieval_counts"]
    assert current["mixed_agent_ace_configs"] == golden["mixed_agent_ace_configs"]


def test_all_migrated_comments_use_catalog_aliases():
    validate_commented_aliases(ROOT)
