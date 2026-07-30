"""Tests for watch_experiments.py eval_on_train phase handling."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))


def _llm_roles(wrapped: bool) -> dict:
    effective = {
        "provider": "hosted_vllm",
        "model": "Qwen/test",
        "api_base": "http://localhost:8800/v1",
        "api_key_env": "HOSTED_VLLM_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra_kwargs": {},
    }
    role = (
        {"preset": "qwen", "overrides": {}, "effective": effective}
        if wrapped
        else effective
    )
    return {"agent": role, "ace": role}


def _make_run(tmp_path, phases, *, wrapped=False):
    """phases: {phase_name: n_instance_dirs}. Each instance gets iter_0.json."""
    bench = "princeton-nlp__SWE-bench_Verified"
    d = tmp_path / "run_watch"
    d.mkdir()
    cfg = {
        "experiment": {"name": "eval", "max_attempts": 1, "concurrency": 6,
                       "skillbook": {"mode": "global"}, "eval_on_train": True,
                       "eval_on_train_pass_k": 1, "split": {"manifest": "x"},
                       "skillbook_source_dir": "x"},
        "llm": _llm_roles(wrapped),
        "benchmark": {"dataset": bench},
    }
    (d / "config.json").write_text(json.dumps(cfg))
    for phase, n in phases.items():
        for i in range(n):
            idir = d / bench / "results" / phase / f"inst-{i}"
            idir.mkdir(parents=True)
            (idir / "iter_0.json").write_text('{"resolved": true}')
    return d


def test_scan_progress_no_triple_count(tmp_path):
    """train_eval + train_eval_baseline are re-passes of train; the headline
    total must not sum all phase dir counts."""
    from watch_experiments import scan_progress
    d = _make_run(tmp_path, {"train_eval": 5, "train_eval_baseline": 5})
    prog = scan_progress(d, dataset="princeton-nlp/SWE-bench_Verified",
                         expected_counts={"train": 5, "val": 5})
    # Headline reflects the TrainSB (train_eval) pass — resolved/processed come
    # from train_eval, NOT 0 (train+val derivation yields 0 with no train/val
    # dirs) and NOT 10 (naive sum of both re-passes).
    assert prog["resolved"] == 5
    assert prog["processed"] == 5
    assert prog["total"] <= 5
    # Both phases are visible in the phase breakdown.
    assert "train_eval" in prog["phases"]
    assert "train_eval_baseline" in prog["phases"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_watch_reads_legacy_and_wrapper_llms_identically(wrapped):
    from watch_experiments import _effective_llm_roles, collect_endpoints

    roles = _effective_llm_roles({"llm": _llm_roles(wrapped)})
    assert roles["agent"]["model"] == "Qwen/test"
    assert roles["agent"]["api_base"] == "http://localhost:8800/v1"

    entries = [
        {
            "status": "RUNNING",
            "llm_config": _llm_roles(wrapped),
        }
    ]
    with patch("watch_experiments.get_endpoint_health", return_value="UP"):
        endpoints = collect_endpoints(entries)
    assert endpoints[0]["api_base"] == "http://localhost:8800/v1"
    assert endpoints[0]["models"] == "Qwen/test"
