# ACE-SWE: Skillbook Learning for SWE-bench

Python research project integrating ACE skillbook learning with mini-swe-agent for SWE-bench Lite issue resolution.

## Commands

```bash
# Setup
uv sync                           # Install dependencies

# Run experiment
uv run python -m src.cli.commands --max-instances 10
uv run python -m src.cli.commands --instance django__django-12345

# Individual phases
uv run python -m src.cli.commands --phase predict --instance <id> --skillbook <path>
uv run python -m src.cli.commands --phase evaluate --instance <id> --trajectory <path>
uv run python -m src.cli.commands --phase learn --instance <id> --trajectory <path>

# Testing
uv run pytest src/tests/ -v
uv run pytest src/tests/ -v -k "not docker"  # Skip Docker tests
```

## Architecture

```
src/
├── phases/           # predict.py, evaluate.py, learn.py
├── agents/           # MiniSWEAgent wrapper
├── config/           # LLM configuration (llm.py)
├── data_io/          # readers.py, writers.py
├── environments/     # Docker/swebench environment
├── evaluation/       # SWE-bench evaluation
├── cli/              # commands.py (entry point)
└── utils/            # Platform, LLM observer

data/
├── <run_timestamp>/  # Output per run
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
