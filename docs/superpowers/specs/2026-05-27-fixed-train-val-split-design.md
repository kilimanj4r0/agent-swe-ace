# Fixed Train/Val Split Manifests

## Problem

Train/val splits are recomputed each run via seed-based shuffle. The global split (all instances shuffled together) and per-repo splits (each repo shuffled independently) produce different val sets. We need a fixed split where the global val set is the union of per-repo val sets, ensuring consistency across experiment modes.

## Design

### Manifest Files

Location: `configs/splits/<benchmark>/val_ratio_<ratio>.json`

Stored in git. Generated once by script, reused by all experiments.

```json
{
  "benchmark": "princeton-nlp__SWE-bench_Lite",
  "val_ratio": 0.25,
  "seed": 42,
  "total_instances": 300,
  "train_instances": ["django__django-11099", "..."],
  "val_instances": ["django__django-11039", "..."],
  "per_repo": {
    "django/django": {"train": ["..."], "val": ["..."]},
    "scikit-learn/scikit-learn": {"train": ["..."], "val": ["..."]}
  }
}
```

`train_instances` and `val_instances` are the union of all per-repo splits. `per_repo` maps repo names to their individual train/val instance ID lists.

### Generation Script: `scripts/generate_splits.py`

```
uv run python scripts/generate_splits.py \
  --benchmark princeton-nlp__SWE-bench_Lite \
  --val-ratio 0.25 --seed 42
```

Algorithm:
1. Load all instance IDs from SWE-bench dataset for the given benchmark
2. Group instances by repo
3. For each repo: create `random.Random(seed + repo_name_as_string)`, shuffle, take `val_count = max(1, int(n * val_ratio))` from front
4. Global train = union of all per-repo train, global val = union of all per-repo val
5. Save manifest to `configs/splits/<benchmark>/val_ratio_0.25.json`
6. Print summary stats

### Config Integration

New config key: `experiment.split.manifest` — path to manifest file (relative to project root or absolute).

Modified `split_instances()` in `commands.py`:
- If `experiment.split.manifest` is set: load manifest, return train/val based on mode
- If not set: fallback to current seed-based algorithm (backward compatible)
- `--val-ratio` CLI flag still works as override (validates against manifest ratio if manifest is provided)

### Usage by Experiment Mode

| Mode | Split source |
|---|---|
| **Global** (`run_full_experiment`) | `manifest["train_instances"]` / `manifest["val_instances"]`, optionally filtered by `--filter-repos` |
| **iterate_repos** | `manifest["per_repo"][repo]` for each repo in the iterate_repos list |
| **Single repo** (`--filter-repos django/django`) | `manifest["per_repo"]["django/django"]` |

When `--filter-repos` is used in global mode, the train/val sets are intersected with the filtered repos' instances from the manifest's `per_repo` section.

### Validation

On loading a manifest:
- Warn if any manifest instance IDs are missing from the current benchmark dataset
- Warn if the current dataset has instances not in the manifest (new benchmark version)
- Fail if `val_ratio` in manifest doesn't match the requested `val_ratio`

### Files to Create/Modify

- **Create**: `scripts/generate_splits.py` — generation script
- **Create**: `configs/splits/` — directory for manifest files
- **Modify**: `src/cli/commands.py` — `split_instances()` to support manifest loading
- **Modify**: `src/cli/commands.py` — `_run_single_repo_experiment()` and `run_full_experiment()` to pass manifest data
