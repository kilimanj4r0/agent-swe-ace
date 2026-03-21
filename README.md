# ACE-SWE: Skillbook Learning for SWE-bench

Integrates ACE skillbook learning with mini-swe-agent to improve SWE-bench Lite issue resolution rates through iterative learning.

## Overview

The system learns from failed attempts by reflecting on trajectories and updating a skillbook of strategies. Each unresolved issue triggers learning that can help future attempts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Loop                           │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Predict  │───▶│ Evaluate │───▶│  Learn   │──┐           │
│  │ (Agent)  │    │(SWE-bench)│   │  (ACE)   │  │           │
│  └──────────┘    └──────────┘    └──────────┘  │           │
│       ▲              │ Resolved?                │           │
│       │              └──────────────────────────┘           │
│       │                      No                             │
│       └──────────────────────────────────┐                 │
│                                          │                  │
│                          With updated    │                  │
│                          skillbook       │                  │
│                                          ▼                  │
│                                   Max attempts?             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run experiment
uv run python -m src.cli.commands --max-instances 10

# Run specific instance
uv run python -m src.cli.commands --instance django__django-12345

# Run with observability
uv run python -m src.cli.commands --observe
```

## Usage

### Full Experiment

```bash
# Run on all instances with 2 attempts each
uv run python -m src.cli.commands

# Limit instances and attempts
uv run python -m src.cli.commands --max-instances 50 --max-attempts 3
```

### Individual Phases

```bash
# Phase 1: Predict (run agent)
uv run python -m src.cli.commands \
    --phase predict \
    --instance django__django-12345 \
    --skillbook data/skillbooks/run_001/iter_0.json

# Phase 2: Evaluate (test patch)
uv run python -m src.cli.commands \
    --phase evaluate \
    --instance django__django-12345 \
    --trajectory data/trajectories/run_001/django__django-12345/iter_0.json

# Phase 3: Learn (update skillbook)
uv run python -m src.cli.commands \
    --phase learn \
    --instance django__django-12345 \
    --trajectory data/trajectories/run_001/django__django-12345/iter_0.json
```

## Configuration

See `config.yaml` for all options:

```yaml
experiment:
  name: "mini-swe-v1-skillbook-learning"
  max_attempts: 2
  skillbook_mode: "per_instance"  # per_instance, per_run

llm:
  agent:
    provider: "hosted_vllm"
    model: "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  ace:
    provider: "zai"
    model: "glm-4.5-airx"

benchmark:
  dataset: "princeton-nlp/SWE-bench_Lite"
  max_instances: null  # all instances

# Skill deduplication settings
deduplication:
  enabled: true
  similarity_threshold: 0.85
  embedding_provider: "sentence_transformers"
  local_model_name: "all-MiniLM-L6-v2"
```

Copy `.env.example` to `.env` and add your API keys.

## Project Structure

```
src/
├── phases/          # Predict, Evaluate, Learn
├── runners/         # Main experiment loop
├── agents/          # mini-swe-agent wrapper
├── data_io/         # Data loading/saving
├── config/          # LLM configuration
├── cli/             # CLI commands (entry point)
├── evaluation/      # SWE-bench evaluation
├── environments/    # Docker environment
└── tests/           # Test suite

data/
├── <run_timestamp>/ # Output per run
│   ├── trajectories/
│   ├── results/
│   └── skillbooks/
```

## Output Structure

```
data/
└── run_20260319_143052/              # run_<compact_timestamp>
    ├── config.json                    # Config used for this run
    ├── statistics.json                # Counts, resolved/unresolved lists, skills
    ├── experiment.log                 # Run-specific log file
    └── princeton-nlp__SWE-bench_Lite/ # Benchmark from config
        ├── trajectories/
        │   └── django__django-12345/
        │       ├── iter_0.json        # Includes message_count, assistant_message_count
        │       └── iter_1.json
        ├── results/
        │   └── django__django-12345/
        │       ├── iter_0.json
        │       └── iter_1.json
        └── skillbooks/
            └── django__django-12345/
                ├── iter_0.json        # Empty (initial), includes skill_count
                └── iter_1.json        # After learning
```

## Dependencies

- **mini-swe-agent** (v1) - SWE resolution agent
- **ace-framework** - Skillbook learning
- **swebench** - Docker harness for evaluation
- **litellm** - Unified LLM API (Z.AI/vLLM)
- **opik** - LLM observability (optional)

## License

MIT
