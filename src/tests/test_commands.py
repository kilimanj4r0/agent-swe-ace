"""Tests for pure functions from src/cli/commands.py."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands import (
    _strict_exit_code,
    deep_merge,
    _load_split_manifest,
    _split_from_manifest,
    split_instances,
    _persist_per_repo_skillbook,
)


class TestStrictExit:
    @pytest.mark.parametrize(
        ("status", "strict", "expected"),
        [
            ("completed", False, 0),
            ("degraded", False, 0),
            ("interrupted", False, 0),
            ("completed", True, 0),
            ("degraded", True, 1),
            ("interrupted", True, 1),
        ],
    )
    def test_strict_exit_code(self, status, strict, expected):
        assert _strict_exit_code({"status": status}, strict) == expected

    def test_main_exits_after_degraded_run_returns(self):
        from cli.commands import main

        events = []

        def fake_run(config, args):
            events.append("run-returned")
            return {"status": "degraded"}

        with (
            patch.object(sys, "argv", ["agent-swe-ace", "--strict"]),
            patch("cli.commands._setup_console_logging"),
            patch("cli.commands.load_config", return_value={}),
            patch("cli.commands.apply_litellm_config"),
            patch("cli.commands.run_full_experiment", side_effect=fake_run) as run,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        assert events == ["run-returned"]
        run.assert_called_once()

    def test_main_non_strict_keeps_degraded_run_successful(self):
        from cli.commands import main

        with (
            patch.object(sys, "argv", ["agent-swe-ace"]),
            patch("cli.commands._setup_console_logging"),
            patch("cli.commands.load_config", return_value={}),
            patch("cli.commands.apply_litellm_config"),
            patch(
                "cli.commands.run_full_experiment",
                return_value={"status": "degraded"},
            ) as run,
        ):
            main()

        run.assert_called_once()


class TestAggregateIterateStatus:
    @pytest.mark.parametrize(
        ("repo_statuses", "expected"),
        [
            (["completed", "completed"], "completed"),
            (["completed", "degraded"], "degraded"),
            (["completed", "interrupted"], "interrupted"),
        ],
    )
    def test_aggregate_preserves_run_severity(
        self, tmp_path, repo_statuses, expected
    ):
        from cli.commands import _aggregate_iterate_stats

        repo_stats = {
            f"org/repo-{index}": {
                "status": status,
                "start_time": f"2026-07-30T10:0{index}:00",
                "end_time": f"2026-07-30T10:0{index}:30",
                "total_instances": 1,
                "processed_instances": 1,
                "resolved_count": 0,
                "infrastructure_error_ids": (
                    [f"instance-{index}"] if status == "degraded" else []
                ),
            }
            for index, status in enumerate(repo_statuses)
        }
        config = {
            "benchmark": {"dataset": "org/benchmark"},
            "experiment": {"name": "test"},
        }

        combined = _aggregate_iterate_stats(repo_stats, config, tmp_path)

        assert combined["status"] == expected
        if expected == "degraded":
            assert combined["infrastructure_error_ids"] == ["instance-1"]


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 10}
        override = {"a": {"y": 99, "z": 3}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}

    def test_override_replaces_non_dict(self):
        base = {"a": "str"}
        override = {"a": {"nested": True}}
        result = deep_merge(base, override)
        assert result == {"a": {"nested": True}}

    def test_base_not_modified(self):
        base = {"a": {"x": 1}, "b": 2}
        override = {"a": {"y": 3}, "c": 4}
        result = deep_merge(base, override)
        # original base must remain unchanged
        assert base == {"a": {"x": 1}, "b": 2}
        assert result != base
        assert result == {"a": {"x": 1, "y": 3}, "b": 2, "c": 4}

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_deeply_nested_merge(self):
        base = {"l1": {"l2": {"l3": {"x": 1, "y": 2}}}}
        override = {"l1": {"l2": {"l3": {"y": 99, "z": 3}}}}
        result = deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3": {"x": 1, "y": 99, "z": 3}}}}


# ---------------------------------------------------------------------------
# _split_from_manifest
# ---------------------------------------------------------------------------


def _make_instance(instance_id: str) -> dict:
    return {"instance_id": instance_id, "repo": "test/repo"}


class TestSplitFromManifest:
    def test_global_split(self):
        instances = [
            _make_instance("id1"),
            _make_instance("id2"),
            _make_instance("id3"),
            _make_instance("id4"),
        ]
        manifest = {
            "train_instances": ["id1", "id3"],
            "val_instances": ["id2", "id4"],
        }
        train, val = _split_from_manifest(instances, manifest, repo=None)
        train_ids = {i["instance_id"] for i in train}
        val_ids = {i["instance_id"] for i in val}
        assert train_ids == {"id1", "id3"}
        assert val_ids == {"id2", "id4"}

    def test_per_repo_split(self):
        instances = [
            _make_instance("id1"),
            _make_instance("id2"),
            _make_instance("id3"),
        ]
        manifest = {
            "train_instances": ["id1"],
            "val_instances": ["id2"],
            "per_repo": {
                "test/repo": {
                    "train": ["id1", "id2"],
                    "val": ["id3"],
                }
            },
        }
        train, val = _split_from_manifest(instances, manifest, repo="test/repo")
        train_ids = {i["instance_id"] for i in train}
        val_ids = {i["instance_id"] for i in val}
        assert train_ids == {"id1", "id2"}
        assert val_ids == {"id3"}

    def test_missing_repo_returns_empty(self):
        instances = [
            _make_instance("id1"),
            _make_instance("id2"),
        ]
        manifest = {
            "train_instances": ["id1"],
            "val_instances": ["id2"],
            "per_repo": {
                "other/repo": {
                    "train": ["id1"],
                    "val": ["id2"],
                }
            },
        }
        # Requesting a repo that is NOT in per_repo -> falls through to global
        train, val = _split_from_manifest(instances, manifest, repo="absent/repo")
        train_ids = {i["instance_id"] for i in train}
        val_ids = {i["instance_id"] for i in val}
        # Falls back to global lists
        assert train_ids == {"id1"}
        assert val_ids == {"id2"}

    def test_per_repo_missing_completely_returns_global(self):
        """When repo is provided but per_repo key is absent, global split used."""
        instances = [_make_instance("id1"), _make_instance("id2")]
        manifest = {
            "train_instances": ["id1"],
            "val_instances": ["id2"],
        }
        train, val = _split_from_manifest(instances, manifest, repo="any/repo")
        train_ids = {i["instance_id"] for i in train}
        val_ids = {i["instance_id"] for i in val}
        assert train_ids == {"id1"}
        assert val_ids == {"id2"}

    def test_extra_instances_not_in_manifest_ignored(self):
        """Instances not mentioned in manifest are excluded from results."""
        instances = [
            _make_instance("id1"),
            _make_instance("id2"),
            _make_instance("id3"),
        ]
        manifest = {
            "train_instances": ["id1"],
            "val_instances": ["id2"],
        }
        train, val = _split_from_manifest(instances, manifest, repo=None)
        train_ids = {i["instance_id"] for i in train}
        val_ids = {i["instance_id"] for i in val}
        assert train_ids == {"id1"}
        assert val_ids == {"id2"}
        assert len(train) + len(val) == 2


# ---------------------------------------------------------------------------
# _load_split_manifest
# ---------------------------------------------------------------------------


class TestLoadSplitManifest:
    def test_returns_none_when_no_manifest(self):
        config = {"experiment": {"split": {"val_ratio": 0.2}}}
        assert _load_split_manifest(config) is None

    def test_returns_none_when_manifest_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        config = {"experiment": {"split": {"manifest": str(missing)}}}
        assert _load_split_manifest(config) is None

    def test_loads_manifest_file(self, tmp_path):
        manifest_file = tmp_path / "split.json"
        manifest_data = {"train_instances": ["a"], "val_instances": ["b"]}
        manifest_file.write_text(json.dumps(manifest_data))
        config = {"experiment": {"split": {"manifest": str(manifest_file)}}}
        result = _load_split_manifest(config)
        assert result == manifest_data

    def test_returns_none_when_no_experiment_key(self):
        assert _load_split_manifest({}) is None


# ---------------------------------------------------------------------------
# split_instances
# ---------------------------------------------------------------------------


class TestSplitInstances:
    def test_no_split_returns_all_train(self):
        instances = [_make_instance("id1"), _make_instance("id2")]
        config = {}  # no experiment.split
        train, val = split_instances(instances, config)
        assert train == instances
        assert val == []

    def test_no_split_config_returns_all_train(self):
        instances = [_make_instance("id1")]
        config = {"experiment": {}}  # no 'split' key
        train, val = split_instances(instances, config)
        assert train == instances
        assert val == []

    def test_seed_split_deterministic(self):
        instances = [_make_instance(f"id{i}") for i in range(20)]
        config = {
            "experiment": {
                "split": {"val_ratio": 0.2},
                "random_seed": 42,
            }
        }
        train1, val1 = split_instances(instances, config)
        train2, val2 = split_instances(instances, config)
        train1_ids = [i["instance_id"] for i in train1]
        train2_ids = [i["instance_id"] for i in train2]
        val1_ids = [i["instance_id"] for i in val1]
        val2_ids = [i["instance_id"] for i in val2]
        assert train1_ids == train2_ids
        assert val1_ids == val2_ids

    def test_seed_split_val_ratio(self):
        instances = [_make_instance(f"id{i}") for i in range(10)]
        config = {
            "experiment": {
                "split": {"val_ratio": 0.2},
                "random_seed": 42,
            }
        }
        train, val = split_instances(instances, config)
        # val_count = max(1, int(10 * 0.2)) = 2
        assert len(val) == 2
        assert len(train) == 8

    def test_seed_split_default_seed(self):
        instances = [_make_instance(f"id{i}") for i in range(10)]
        config = {
            "experiment": {
                "split": {"val_ratio": 0.3},
                # no random_seed -> defaults to 42
            }
        }
        train, val = split_instances(instances, config)
        # val_count = max(1, int(10 * 0.3)) = 3
        assert len(val) == 3
        assert len(train) == 7

    def test_manifest_split(self, tmp_path):
        instances = [
            _make_instance("id1"),
            _make_instance("id2"),
            _make_instance("id3"),
            _make_instance("id4"),
        ]
        manifest_data = {
            "train_instances": ["id1", "id3"],
            "val_instances": ["id2", "id4"],
            "val_ratio": 0.5,
        }
        manifest_file = tmp_path / "split.json"
        manifest_file.write_text(json.dumps(manifest_data))
        config = {
            "experiment": {
                "split": {
                    "manifest": str(manifest_file),
                    "val_ratio": 0.5,
                },
            },
        }
        train, val = split_instances(instances, config)
        train_ids = {i["instance_id"] for i in train}
        val_ids = {i["instance_id"] for i in val}
        assert train_ids == {"id1", "id3"}
        assert val_ids == {"id2", "id4"}

    def test_manifest_split_takes_priority_over_seed(self, tmp_path):
        """When both manifest and val_ratio are present, manifest wins."""
        instances = [_make_instance(f"id{i}") for i in range(6)]
        manifest_data = {
            "train_instances": ["id0", "id1", "id2"],
            "val_instances": ["id3", "id4", "id5"],
            "val_ratio": 0.5,
        }
        manifest_file = tmp_path / "split.json"
        manifest_file.write_text(json.dumps(manifest_data))
        config = {
            "experiment": {
                "split": {
                    "manifest": str(manifest_file),
                    "val_ratio": 0.5,
                },
                "random_seed": 42,
            },
        }
        train, val = split_instances(instances, config)
        assert len(train) == 3
        assert len(val) == 3

    def test_single_instance_gives_one_val(self):
        """With 1 instance and val_ratio 0.2, val_count = max(1, 0) = 1."""
        instances = [_make_instance("only")]
        config = {
            "experiment": {
                "split": {"val_ratio": 0.2},
                "random_seed": 42,
            }
        }
        train, val = split_instances(instances, config)
        assert len(val) == 1
        assert len(train) == 0


class TestResolveIterateReposConcurrency:
    """Between-repo concurrency resolves from experiment.iterate_repos_concurrency,
    falling back to the legacy top-level experiment.concurrency (>1) for migration."""

    def test_default_is_one(self):
        from cli.commands import _resolve_iterate_repos_concurrency
        assert _resolve_iterate_repos_concurrency({}) == 1

    def test_explicit_key_wins(self):
        from cli.commands import _resolve_iterate_repos_concurrency
        assert _resolve_iterate_repos_concurrency({"iterate_repos_concurrency": 4}) == 4
        assert _resolve_iterate_repos_concurrency(
            {"iterate_repos_concurrency": 4, "concurrency": 2}
        ) == 4

    def test_falls_back_to_legacy_concurrency(self):
        from cli.commands import _resolve_iterate_repos_concurrency
        assert _resolve_iterate_repos_concurrency({"concurrency": 3}) == 3

    def test_legacy_concurrency_of_one_does_not_warn(self, caplog):
        import logging
        from cli.commands import _resolve_iterate_repos_concurrency
        with caplog.at_level(logging.WARNING):
            assert _resolve_iterate_repos_concurrency({"concurrency": 1}) == 1
        assert not any("iterate_repos_concurrency" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _persist_per_repo_skillbook
# ---------------------------------------------------------------------------


class TestPersistPerRepoSkillbook:
    """Per-repo persistence must write `sources` on every skill.

    Regression: the per-repo final_skillbook.json previously used an inline
    serialization copy that dropped `sources`. The shared helper now backs both
    the normal iterate_repos path and the learn-replay path.
    """

    def test_writes_sources_and_correct_path(self, tmp_path):
        from ace import Skillbook, Skill

        sb = Skillbook()
        sb._skills["debugging-00001"] = Skill(
            id="debugging-00001",
            section="debugging",
            content="Reproduce before fixing",
            sources=[{"instance_id": "django__django-14376", "repo": "django/django"}],
        )

        out_dir = tmp_path / "run_x"
        out_path = _persist_per_repo_skillbook(
            sb, out_dir, benchmark="princeton-nlp__SWE-bench_Verified", repo="django/django"
        )

        expected = (
            out_dir / "princeton-nlp__SWE-bench_Verified" / "skillbooks" / "per_repo"
            / "django__django" / "final_skillbook.json"
        )
        assert out_path == expected
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["skill_count"] == 1
        skill = data["skills"]["debugging-00001"]
        assert skill["sources"] == [
            {"instance_id": "django__django-14376", "repo": "django/django"}
        ]
        assert set(skill.keys()) == {
            "id", "section", "content", "justification", "evidence", "sources"
        }

    def test_every_skill_carries_sources_key(self, tmp_path):
        """Skills with empty sources still get the key — no field drift."""
        from ace import Skillbook, Skill

        sb = Skillbook()
        for i in range(3):
            sb._skills[f"s-{i}"] = Skill(id=f"s-{i}", section="general", content=f"skill {i}")

        out_dir = tmp_path / "run_y"
        out_path = _persist_per_repo_skillbook(sb, out_dir, benchmark="bench", repo="o/r")

        data = json.loads(out_path.read_text())
        assert data["skill_count"] == 3
        assert all("sources" in s for s in data["skills"].values())


# ---------------------------------------------------------------------------
# _resolve_learn_replay_settings (CLI > config priority)
# ---------------------------------------------------------------------------


class TestResolveLearnReplaySettings:
    """Learn-replay settings resolve with CLI > config priority."""

    def _args(self, **kw):
        defaults = {
            "replay_run_dir": None,
            "replay_repos": None,
            "replay_concurrency": None,
            "in_place": False,
        }
        defaults.update(kw)
        return types.SimpleNamespace(**defaults)

    def test_no_request_returns_none_run_dir(self):
        from cli.commands import _resolve_learn_replay_settings
        lr = _resolve_learn_replay_settings({}, self._args())
        assert lr["run_dir"] is None
        assert lr["concurrency"] == 1
        assert lr["in_place"] is False

    def test_cli_overrides_config(self):
        from cli.commands import _resolve_learn_replay_settings
        config = {"experiment": {"learn_replay": {
            "run_dir": "data/cfg_run",
            "concurrency": 4,
            "in_place": True,
            "repos": ["a/b"],
        }}}
        lr = _resolve_learn_replay_settings(
            config, self._args(replay_run_dir="data/cli_run", replay_concurrency=2)
        )
        assert lr["run_dir"] == "data/cli_run"   # CLI wins
        assert lr["concurrency"] == 2             # CLI wins
        assert lr["in_place"] is True             # config (CLI flag not set)
        assert lr["repos"] == ["a/b"]             # config (CLI not set)

    def test_config_only(self):
        from cli.commands import _resolve_learn_replay_settings
        config = {"experiment": {"learn_replay": {"run_dir": "data/cfg_run", "concurrency": 8}}}
        lr = _resolve_learn_replay_settings(config, self._args())
        assert lr["run_dir"] == "data/cfg_run"
        assert lr["concurrency"] == 8
        assert lr["in_place"] is False
        assert lr["repos"] is None

    def test_concurrency_defaults_to_one(self):
        from cli.commands import _resolve_learn_replay_settings
        lr = _resolve_learn_replay_settings({}, self._args(replay_run_dir="x"))
        assert lr["concurrency"] == 1


# ---------------------------------------------------------------------------
# _resolve_eval_on_train
# ---------------------------------------------------------------------------


class TestResolveEvalOnTrain:
    def test_defaults(self):
        from cli.commands import _resolve_eval_on_train
        assert _resolve_eval_on_train({}) == (False, 1)

    def test_reads_keys(self):
        from cli.commands import _resolve_eval_on_train
        assert _resolve_eval_on_train({"eval_on_train": True, "eval_on_train_pass_k": 5}) == (True, 5)

    def test_enabled_only(self):
        from cli.commands import _resolve_eval_on_train
        assert _resolve_eval_on_train({"eval_on_train": True}) == (True, 1)
