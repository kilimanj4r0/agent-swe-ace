# ACE-SWE: Skillbook Learning for SWE-bench

Python research project integrating ACE skillbook learning with mini-swe-agent for SWE-bench Lite issue resolution.

## Commands

```bash
# Setup
uv sync                           # Install dependencies

# Run experiment
uv run python -m src.cli.commands --max-instances 10
uv run python -m src.cli.commands --instance django__django-12345

# Resume from baseline (skip iter_0 predict/evaluate)
uv run python -m src.cli.commands --baseline-dir data/run_baseline_qwen3coder --max-attempts 3

# Run with Opik observability
uv run python -m src.cli.commands --observe

# Individual phases
uv run python -m src.cli.commands --phase predict --instance <id> --skillbook <path>
uv run python -m src.cli.commands --phase evaluate --instance <id> --trajectory <path>
uv run python -m src.cli.commands --phase learn --instance <id> --trajectory <path>

# Testing
uv run pytest src/tests/ -v
uv run pytest src/tests/ -v -k "not docker"  # Skip Docker tests

# Transform baseline data to run format
uv run python scripts/transform_baseline_to_run_format.py --in-place
```

## Architecture

```
src/
├── phases/           # predict.py, evaluate.py, learn.py
├── runners/          # main_loop.py (ExperimentLoop with baseline support)
├── agents/           # MiniSWEAgent wrapper
├── config/           # LLM configuration (llm.py)
├── data_io/          # readers.py, writers.py
├── environments/     # Docker/swebench environment
├── evaluation/       # SWE-bench evaluation
├── cli/              # commands.py (entry point)
└── utils/            # Platform, LLM observer (Opik)

scripts/
├── transform_baseline_to_run_format.py  # Convert baseline trajectories to run format

data/
├── <run_timestamp>/  # Output per run
│   ├── config.json
│   ├── statistics.json  # Includes observability_project_url when --observe
│   ├── trajectories/<instance>/iter_N.json
│   ├── results/<instance>/iter_N.json
│   └── skillbooks/<instance>/iter_N.json
```

## Experiment Flow

1. **Predict**: MiniSWEAgent generates patch using skillbook
2. **Evaluate**: Test patch in Docker container (SWE-bench harness)
3. **Learn**: If unresolved, ACE Reflector analyzes failure, updates skillbook
4. Loop until resolved or max_attempts reached

## Configuration

- `config.yaml` - Main config (LLM providers, limits, benchmark)
- `.env` - API keys (ZAI_API_KEY, HOSTED_VLLM_API_KEY)
  - Observability: OPIK_API_KEY="local", OPIK_URL_OVERRIDE="http://localhost:5173/api"

## Key Patterns

- Dual LLM config: `agent` (runs mini-swe-agent) and `ace` (runs Reflector/SkillManager)
- Skillbook modes: `per_instance` or `per_run`
- Docker required for evaluation (SWE-bench images)
- Output uses compact timestamps: `run_20260319_143052`
- Trajectory files include `message_count` and `assistant_message_count` for quick analysis
- Skillbook files include `skill_count` at top level
- `experiment.log` is saved to each run folder via `setup_run_logging()`
- Skill deduplication via ace_next.deduplication with configurable similarity threshold

## Gotchas

- Requires Docker running for evaluation phase
- API keys must be in `.env` (not config.yaml)
- mini-swe-agent is installed from git (v1 branch)
- Step/cost limits in config control agent behavior
- `--baseline-dir` loads existing iter_0 results and continues from iter_1
- Each run creates unique Opik project with run_id as name
- statistics.json includes `observability_project_url` when observability enabled
- Baseline runs have `config.baseline: true` in statistics.json

## Git Commits

- Do not add Co-Authored-By line to commits
