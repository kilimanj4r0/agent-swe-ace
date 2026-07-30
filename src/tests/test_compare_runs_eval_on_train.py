"""Golden-output tests for compare_runs.py eval_on_train support."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))


def _llm_roles(wrapped: bool) -> dict:
    effective = {
        "provider": "hosted_vllm",
        "model": "Qwen/Qwen3-Coder-Next-FP8",
        "api_base": "http://10.100.30.241:8801/v1",
        "api_key_env": "HOSTED_VLLM_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra_kwargs": {},
    }
    role = (
        {
            "preset": "qwen-next",
            "overrides": {},
            "effective": effective,
        }
        if wrapped
        else effective
    )
    return {"agent": role, "ace": role}


def _make_run_dir(tmp_path, *, retrieval_enabled=False, wrapped=False):
    d = tmp_path / f"run_eval_on_train_{'wrapped' if wrapped else 'legacy'}"
    d.mkdir()
    stats = {
        "run_name": "eval-on-train",
        "timestamp": "2026-06-28T00:00:00",
        "total_instances": 2,
        "processed_instances": 2,
        "resolved_count": 1,
        "unresolved_count": 1,
        "resolution_rate": 0.5,
        "resolved_ids": ["t1"], "unresolved_ids": ["t2"],
        "status": "completed",
        "train_eval_baseline_phase": {
            "total_instances": 2, "resolved_count": 0, "unresolved_count": 2,
            "resolution_rate": 0.0, "resolved_ids": [], "unresolved_ids": ["t1", "t2"],
            "pass_at_k": {}, "per_attempt_rate": {},
        },
        "train_eval_phase": {
            "total_instances": 2, "resolved_count": 1, "unresolved_count": 1,
            "resolution_rate": 0.5, "resolved_ids": ["t1"], "unresolved_ids": ["t2"],
            "pass_at_k": {}, "per_attempt_rate": {},
        },
        "summary": {
            "train_eval_baseline_resolution_rate": 0.0,
            "train_eval_resolution_rate": 0.5,
            "train_skillbook_improvement": "+0.500",
        },
        "retrieval": {"enabled": retrieval_enabled, "type": "bm25" if retrieval_enabled else None},
    }
    (d / "statistics.json").write_text(json.dumps(stats))
    cfg = {
        "experiment": {
            "name": "eval-on-train", "max_attempts": 1, "concurrency": 6,
            "skillbook": {"mode": "global", "retrieval": {"enabled": retrieval_enabled}},
            "val_pass_k": 1,
        },
        "llm": _llm_roles(wrapped),
        "benchmark": {"dataset": "princeton-nlp/SWE-bench_Verified"},
    }
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_load_run_detects_eval_on_train_phases(tmp_path):
    from compare_runs import load_run
    r = load_run(_make_run_dir(tmp_path))
    assert r is not None
    assert "train_eval" in r["split"] or r["split"].get("train_eval")
    assert "train_eval_baseline" in r["split"]


def test_load_run_train_eval_rates(tmp_path):
    from compare_runs import load_run
    r = load_run(_make_run_dir(tmp_path))
    assert r["split"]["train_eval"]["resolved"] == 1
    assert r["split"]["train_eval_baseline"]["resolved"] == 0


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_run_reads_legacy_and_wrapper_models_identically(tmp_path, wrapped):
    from compare_runs import load_run

    run = load_run(_make_run_dir(tmp_path, wrapped=wrapped))

    assert run["agent_llm"] == "Qwen/Qwen3-Coder-Next-FP8"
    assert run["ace_llm"] == "Qwen/Qwen3-Coder-Next-FP8"
