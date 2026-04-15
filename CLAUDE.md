# ACE-SWE: Skillbook Learning for SWE-bench

Python research project integrating ACE skillbook learning with mini-swe-agent for SWE-bench Lite issue resolution.

## Commands

```bash
# Setup
uv sync                           # Install dependencies

# Run experiment
uv run python -m src.cli.commands --max-instances 10
uv run python -m src.cli.commands --instance django__django-12345

# Run with override config (deep-merged on top of config.yaml)
uv run python -m src.cli.commands --config configs/agent-glm-ace-glm.yaml
uv run python -m src.cli.commands --config configs/agent-qwen3-ace-qwen3.yaml --custom-swe-learn --observe

# Resume from previous runs (copies completed instances, continues partial)
uv run python -m src.cli.commands --resume-dir data/run_20260415_020540 data/run_20260415_020217

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

# Testing
uv run pytest src/tests/ -v
uv run pytest src/tests/ -v -k "not docker"  # Skip Docker tests
uv run pytest -m integration                 # Only real API call tests

# Transform baseline data to run format
uv run python scripts/transform_baseline_to_run_format.py --in-place
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
└── utils/            # logging.py (thread-local instance context), platform.py, llm_observer.py

├── prompts/          # SWE-optimized learning (SWEReflector, SWESkillManager, output models)

scripts/
├── prepare_images.py                     # Build Docker images for evaluation
├── analyze_token_usage.py                # Token usage analysis
└── transform_baseline_to_run_format.py   # Convert baseline trajectories to run format

configs/                                   # Override configs (deep-merged on top of config.yaml)
├── agent-glm-ace-glm.yaml
├── agent-qwen3-ace-qwen3.yaml
└── test.yaml

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
- Skillbook modes: `per_instance` or `per_run`
- Docker required for evaluation (SWE-bench images)
- Output uses compact timestamps: `run_20260319_143052`
- Trajectory files include `message_count` and `assistant_message_count` for quick analysis
- Skillbook files include `skill_count` at top level
- `experiment.log` is saved to each run folder via `setup_logging(run_dir=...)`
- Skill deduplication via ace.deduplication with configurable similarity threshold
- PydanticAI schema monkey-patch on import in `config/llm.py`: inlines `$ref`/`$defs` in tool schemas (Z.AI/GLM models can't handle JSON references)
- Default ACE routing: when `api_base` is set, code sets `OPENAI_BASE_URL`/`OPENAI_API_KEY` env vars at runtime and prefixes model with `"openai:"` so PydanticAI uses OpenAIProvider
- Model `n_calls`/`cost` counters reset to 0 before each agent run (accumulation would trigger immediate LimitsExceeded)
- Concurrency (`experiment.concurrency > 1`): each worker gets its own agent via `agent_factory()`; evaluation still serialized via global lock
- Context window: `max_input_tokens = context_window - max_tokens - 2000` (hardcoded safety buffer)

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

## Git Commits

- Do not add Co-Authored-By line to commits
