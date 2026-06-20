# ACE-SWE: Skillbook Learning for SWE-bench

Python research project integrating ACE skillbook learning with mini-swe-agent for SWE-bench Lite issue resolution.

## Commands

```bash
# Setup
uv sync                           # Install dependencies

# Run experiment
uv run python -m src.cli.commands --max-instances 10
uv run python -m src.cli.commands --instance django__django-12345

# List repos and preview train/val split
uv run python -m src.cli.commands --list-repos
uv run python -m src.cli.commands --list-repos --filter-repos django/django --val-ratio 0.2

# Two-phase skillbook experiment (single repo)
uv run python -m src.cli.commands --filter-repos django/django --val-ratio 0.2 --config configs/agent-glm-ace-glm.yaml
uv run python -m src.cli.commands --filter-repos django/django --val-ratio 0.2 --baseline-run-dir data/run_20260415_xxx

# Two-phase skillbook experiment (multiple repos via iterate_repos)
uv run python -m src.cli.commands --config configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-qwen3-verified-iterate-repos-swe.yaml

# Run with override config (deep-merged on top of config.yaml)
uv run python -m src.cli.commands --config configs/agent-glm-ace-glm.yaml
uv run python -m src.cli.commands --config configs/agent-qwen3-ace-qwen3-full-4a-swe.yaml --custom-swe-learn --observe

# Resume from previous runs (copies completed instances, continues partial)
uv run python -m src.cli.commands --resume-dir data/run_20260415_020540 data/run_20260415_020217

# Validation-only mode: reuse skillbooks from a previous run, skip training
# Requires skillbook_source_dir + val_pass_k in config
uv run python -m src.cli.commands --config configs/val_only_k4.yaml

# Teacher distillation: learn skillbooks from Opus 4.5 teacher trajectories
uv run python -m src.cli.commands --config configs/agent-qwen3-ace-distill.yaml
uv run python -m src.cli.commands --config configs/princeton-nlp__SWE-bench_Verified/agent-qwen3-ace-distill-iterate-repos.yaml

# Raw multi-attempt benchmark (no skillbooks, 4 attempts per instance)
uv run python -m src.cli.commands --config configs/no_skillbook.yaml

# Run with Opik observability
uv run python -m src.cli.commands --observe

# Individual phases (--instance required; --trajectory or --patch for evaluate; --trajectory for learn)
uv run python -m src.cli.commands --phase predict --instance <id> --skillbook <path>
uv run python -m src.cli.commands --phase evaluate --instance <id> --trajectory <path>
uv run python -m src.cli.commands --phase evaluate --instance <id> --patch "diff --git..."
uv run python -m src.cli.commands --phase learn --instance <id> --trajectory <path>

# Docker image preparation (required before first run)
uv run python scripts/prepare_images.py                    # All instances
uv run python scripts/prepare_images.py --instances django__django-11099  # Specific
uv run python scripts/prepare_images.py --workers 8 --force

# Utilities
uv run python scripts/analyze_token_usage.py --data-dir data/ --output-csv tokens.csv

# Testing (unit + integration, no Docker/LLM needed)
uv run pytest src/tests/ -v                              # All tests (integration tests may hang without LLM)
uv run pytest src/tests/ -v -m "not integration"         # Unit tests only (~176 tests, runs in ~4s)
uv run pytest src/tests/ -v -k "test_phases"             # Specific test file pattern
uv run pytest -m integration                             # Only real API call tests

# Test files cover: phases, main_loop, context_manager, miniswe_agent, io, resume_scanner,
# custom_swe_learn, llm_config, llm_health, commands (pure functions)

# End-to-end smoke tests for all experiment modes (requires Docker + LLM)
uv run python scripts/test_modes.py                  # run all 10 modes
uv run python scripts/test_modes.py --only 01 04      # specific modes
uv run python scripts/test_modes.py --keep --verbose  # keep output, show logs

# Transform baseline data to run format
uv run python scripts/transform_baseline_to_run_format.py --in-place

# Watch running experiments (CLI dashboard)
python scripts/watch_experiments.py              # auto-refresh every 10s
python scripts/watch_experiments.py -n            # one-shot
python scripts/watch_experiments.py --running     # only active runs
python scripts/watch_experiments.py --all         # include old completed runs
```

## Architecture

```
src/
├── phases/           # predict.py, evaluate.py, learn.py
├── runners/          # main_loop.py (ExperimentLoop with resume support)
├── agents/           # miniswe_agent.py (MiniSWEAgent), context_manager.py (5-level truncation)
├── config/           # llm.py (LLMConfig, model creation, PydanticAI schema patch)
├── data_io/          # readers.py, writers.py, resume_scanner.py
├── environments/     # Docker/swebench environment
├── evaluation/       # swebench.py (serialized via threading.Lock)
├── cli/              # commands.py (entry point)
├── prompts/          # SWE-optimized learning (SWEReflector, SWESkillManager, output models)
├── retrieval/        # skill retrieval: skill_retriever.py (LLM filter+rank), bm25/embedding/random_retriever.py, base.py
├── tests/            # Unit and integration tests
└── utils/            # logging.py (thread-local instance context), platform.py, llm_observer.py

scripts/
├── prepare_images.py                     # Build Docker images for evaluation
├── analyze_token_usage.py                # Token usage analysis
├── transform_baseline_to_run_format.py   # Convert baseline trajectories to run format
├── compare_runs.py                       # Compare completed experiment runs (summary table)
├── watch_experiments.py                  # Live CLI dashboard for running experiments
├── test_modes.py                         # End-to-end smoke tests for all experiment modes
├── run_vllm_watchdog.sh                  # vLLM watchdog auto-restart (parameterized port/GPU/model)
│                                          # Usage: CUDA_VISIBLE_DEVICES=0,1 PORT=8800 bash run_vllm_watchdog.sh

configs/                                   # Override configs (deep-merged on top of config.yaml)
├── agent-glm-ace-glm.yaml
├── agent-glm-ace-glm-default.yaml
├── agent-qwen3-ace-qwen3-full-1a-baseline.yaml
├── agent-qwen3-ace-qwen3-full-4a-default.yaml
├── agent-qwen3-ace-qwen3-full-4a-swe.yaml
├── agent-qwen3-ace-qwen3-full-global-split-default.yaml
├── agent-qwen3-ace-distill.yaml
├── agent-qwen3-ace-glm.yaml
├── test.yaml
└── test/                                   # End-to-end smoke test configs (run via scripts/test_modes.py)
    ├── 01_basic.yaml                       # Multi-attempt with learning
    ├── 02_skip_learn.yaml                  # Skip-learn raw benchmark
    ├── 03_concurrent.yaml                  # Parallel execution
    ├── 04_two_phase.yaml                   # Two-phase per_repo + custom SWE
    ├── 05_two_phase_global.yaml            # Two-phase global skillbook
    ├── 06_iterate_repos.yaml               # iterate_repos with concurrency
    ├── 07_resume.yaml                      # Resume from previous run
    ├── 08_baseline_reuse.yaml              # Baseline reuse in two-phase
    ├── 09_distillation.yaml                # Teacher distillation
    └── 10_validation_only.yaml             # Validation-only mode

data/
├── <run_timestamp>/  # Output per run
│   ├── config.json
│   ├── statistics.json  # Includes observability_project_url when --observe
│   ├── experiment.log
│   # New layout (benchmark-scoped subdirectory):
│   ├── princeton-nlp__SWE-bench_Lite/
│   │   ├── trajectories/<instance>/iter_N.json
│   │   ├── results/<instance>/iter_N.json
│   │   └── skillbooks/<instance>/iter_N.json
│   # Old layout (flat, still supported):
│   ├── trajectories/<instance>/iter_N.json
│   ├── results/<instance>/iter_N.json
│   └── skillbooks/<instance>/iter_N.json
│   # Two-phase mode adds subdirectories:
│   ├── trajectories/{train,val_baseline,val}/<instance>/iter_N.json
│   ├── results/{train,val_baseline,val}/<instance>/iter_N.json
│   ├── skillbooks/train/iter_N.json
│   ├── skillbooks/final_skillbook.json
│   └── skillbooks/per_repo/<repo>/final_skillbook.json  # Per-repo skillbooks (iterate_repos mode)
```

## Experiment Flow

0. **Retrieve** (optional): when `experiment.skillbook.retrieval.enabled`, narrow the skillbook to the top-k skills most relevant to the instance (type: `llm` two-stage filter+rank, `bm25`, `embedding`, or `random` baseline)
1. **Predict**: MiniSWEAgent generates patch using skillbook
2. **Evaluate**: Test patch in Docker container (SWE-bench harness)
3. **Learn**: If unresolved, ACE Reflector analyzes failure, updates skillbook
4. Loop until resolved or max_attempts reached

## Configuration

- Config layers (later overrides earlier): config.yaml < --config <file> < CLI args
- Override configs live in `configs/`, only need keys they change
- `config.yaml` - Main config (LLM providers, limits, benchmark)
- `experiment.resume_dirs` in config.yaml also works (CLI takes priority)
- `.env` - API keys (ZAI_API_KEY, HOSTED_VLLM_API_KEY)
  - Observability: OPIK_API_KEY="local", OPIK_URL_OVERRIDE="http://localhost:5173/api"
  - ACE learn retries: ACE_LEARN_MAX_RETRIES (default "50")

## Key Patterns

- Dual LLM config: `agent` (runs mini-swe-agent) and `ace` (runs Reflector/SkillManager)
- `--custom-swe-learn` flag: uses SWE-optimized Reflector/SkillManager (anti-patterns, type prefixes) vs default ACE
- Skillbook modes: `per_instance`, `per_repo`, or `global`
- Two-phase experiment: `--filter-repos <repo> --val-ratio 0.2` splits into train/val
  - Train: 1 attempt per instance, force learn (even on success), skillbook accumulates
  - Val baseline: K attempts per instance (default 1), empty skillbook, no learning
  - Val skillbook: K attempts per instance (default 1), learned skillbook from train, no learning
  - statistics.json includes `train_phase`, `val_baseline_phase`, `val_skillbook_phase`, `summary`
- `benchmark.iterate_repos`: run independent per-repo two-phase experiments for each repo listed
  - Each repo gets its own train/val split and skillbook
  - Per-repo skillbooks persisted to `skillbooks/per_repo/<repo>/final_skillbook.json`
  - If `concurrency > 1`, repos run in parallel (ThreadPoolExecutor)
  - Per-repo stats in `statistics_per_repo/<repo>.json`, combined in `statistics.json`
  - Orchestration in `_run_iterate_repos()` / `_run_single_repo_experiment()` in commands.py
- `experiment.val_pass_k`: number of attempts per val instance (default: 1). Works in both normal and validation-only mode.
- `experiment.skillbook_source_dir`: load per-repo skillbooks from a completed run, skip training entirely (validation-only mode)
  - Loads from `skillbooks/per_repo/<repo>/final_skillbook.json`, falls back to global `final_skillbook.json`
- `experiment.train_trajs_dir`: when set with `val_ratio`, train phase loads teacher trajectories instead of running predict→eval
  - Only learn phase runs on train instances (distillation from expert trajectories)
  - Missing teacher trajectories are skipped (not retried) and reported as `teacher_trajs_skipped` in statistics
  - `train_phase` statistics include `teacher_trajs_found`, `teacher_trajs_skipped`, `teacher_trajs_resolved`, `train_trajs_dir`
  - Works with `iterate_repos` for per-repo distillation
  - `skillbook_source_dir` can reuse the skillbook output from a distillation run
- `experiment.skip_learn`: when true, completely skip Learn phase. Use with `max_attempts > 1` for raw multi-attempt benchmark runs without skillbooks. Resolved instances are not retried.
- `--list-repos` prints all repos with counts and optional split preview
- `--baseline-run-dir` loads existing baseline results to avoid re-running val baseline
- Docker required for evaluation (SWE-bench images)
- Output uses compact timestamps: `run_20260319_143052`
- Trajectory files include `message_count` and `assistant_message_count` for quick analysis
- Skillbook files include `skill_count` at top level
- `experiment.log` is saved to each run folder via `setup_logging(run_dir=...)`
- Skill deduplication via experiment.skillbook.deduplication with configurable similarity threshold
- Skill retrieval (optional, runs before predict): `experiment.skillbook.retrieval.{enabled,type,top_k,skip_threshold}` — type is `llm` (default, two-stage filter+rank), `bm25`, `embedding`, or `random` baseline; built by `_build_skill_retriever()` in commands.py. A single retriever instance is shared across worker threads under `concurrency > 1` (BM25/embedding guard shared state with locks).
- Config structure: `experiment.skillbook.{mode,custom_swe_learn,deduplication}`, `agent.context.{enabled,context_window,...}`
- All config reads are in `src/cli/commands.py` — other files receive values via constructor args
- PydanticAI schema monkey-patch on import in `config/llm.py`: inlines `$ref`/`$defs` in tool schemas (Z.AI/GLM models can't handle JSON references)
- Default ACE routing: when `api_base` is set, code sets `OPENAI_BASE_URL`/`OPENAI_API_KEY` env vars at runtime and prefixes model with `"openai:"` so PydanticAI uses OpenAIProvider
- Model `n_calls`/`cost` counters reset to 0 before each agent run (accumulation would trigger immediate LimitsExceeded)
- Concurrency (`experiment.concurrency > 1`): each worker gets its own agent via `agent_factory()`; evaluation still serialized via global lock
- Context window: `max_input_tokens = context_window - max_tokens - 2000` (hardcoded safety buffer)

## Testing Strategy

Three layers of testing, from fastest to most comprehensive:

1. **Unit tests** (`uv run pytest src/tests/ -m "not integration"`) — ~176 tests, ~4s, no external deps
   - Phases (predict/evaluate/learn), main_loop, context_manager, miniswe_agent
   - Data I/O (readers, writers), resume scanner, LLM config validation
   - Commands pure functions (deep_merge, split_instances, manifest loading)
   - Custom SWE learn (output types, prefix mapping)

2. **End-to-end smoke tests** (`uv run python scripts/test_modes.py`) — 10 modes, ~6min, requires Docker + LLM
   - Configs in `configs/test/01_basic.yaml` through `10_validation_only.yaml`
   - Exercises all experiment modes: basic, skip_learn, concurrent, two_phase, iterate_repos, resume, baseline_reuse, distillation, validation_only
   - Runner handles data dependencies (parallel standalone, sequential dependent)

3. **Single-instance real runs** — for debugging specific configs or phases:

```bash
# Baseline quick test (1 instance, 1 attempt, 3 agent steps, cheap model)
uv run python -m src.cli.commands --config configs/test.yaml
```

Adjust params per test case by overriding config keys or CLI flags:

| What you're testing | Override |
|---|---|
| Learn/retry flow | `--max-attempts 2` (so the loop actually retries) |
| Skip-learn mode | `--config configs/no_skillbook.yaml` |
| Resume from previous run | `--resume-dir data/run_<from_test> --config configs/test.yaml` |
| Specific instance | `--instance django__django-12345` |
| Multiple instances | `--max-instances 3` |
| Two-phase experiment | add `--filter-repos django/django --val-ratio 0.2` + set skillbook mode to `per_repo` |

The test config uses `glm-4.5-flash` (fast/cheap), `step_limit: 3`, `max_instances: 1`, `max_attempts: 1`. Tweak only what the specific test case requires — everything else stays minimal.

## Gotchas

- Requires Docker running for evaluation phase
- API keys must be in `.env` (not config.yaml)
- mini-swe-agent is installed from git (v1 branch)
- Step/cost limits in config control agent behavior
- `--resume-dir` accepts multiple paths, resumes from last successful iteration per instance
- Good exit statuses for resume: `"Submitted"`, `"LimitsExceeded"` — anything else breaks the iteration chain
- Each run creates unique Opik project with run_id as name
- statistics.json includes `observability_project_url` when observability enabled
- Resumed runs include `resume_dirs` and `resumed_complete_count` in statistics.json
- Evaluation is always serialized (threading.Lock) even with concurrency > 1 — concurrency only overlaps prediction
- `per_instance` skillbook mode is incompatible with two-phase experiments — use `per_repo` or `global`
- `uv run pytest` must be run from project root (`/root/makharev/agent-swe-ace`)

## Git Commits

- Do not add Co-Authored-By line to commits
