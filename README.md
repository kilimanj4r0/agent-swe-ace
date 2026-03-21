# ACE-SWE: Skillbook Learning for SWE-bench

Integrates ACE skillbook learning with mini-swe-agent to improve SWE-bench Lite issue resolution rates through iterative learning.

## Overview

The system learns from failed attempts by reflecting on trajectories and updating a skillbook of strategies. Each unresolved issue triggers learning that can help future attempts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Loop                          │
│                                                             │
│   ┌─────────────────────────┐                               │
│   │ Start (or load baseline)│                               │
│   └───────────┬─────────────┘                               │
│               ▼                                             │
│   ┌───────────────────┐                                     │
│   │     Predict       │  Agent + Skillbook → Patch          │
│   └─────────┬─────────┘                                     │
│             ▼                                               │
│   ┌───────────────────┐                                     │
│   │     Evaluate      │  SWE-bench Docker → Resolved?       │
│   └─────────┬─────────┘                                     │
│             │                                               │
│             ▼                                               │
│       ┌─────────────┐                                       │
│       │  Resolved?  │───Yes──▶ Done (resolved)              │
│       └──────┬──────┘                                       │
│              │No                                            │
│              ▼                                              │
│       ┌─────────────┐                                       │
│       │Max attempts?│───Yes──▶ Done (unresolved)            │
│       └──────┬──────┘                                       │
│              │No                                            │
│              ▼                                              │
│       ┌─────────────┐                                       │
│       │    Learn    │  ACE Reflector → Update Skillbook     │
│       └──────┬──────┘                                       │
│              │                                              │
│              └──────────────────────▶ Predict (next iter)   │
│                                                             │
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

# Run with observability (creates unique Opik project per run)
uv run python -m src.cli.commands --observe
```

### Resume from Baseline

You can skip predict/evaluate for iter_0 by providing a baseline run directory with existing results:

```bash
# Run experiment starting from baseline iter_0 results
uv run python -m src.cli.commands \
    --config config.yaml \
    --baseline-dir data/run_baseline_qwen3coder \
    --max-attempts 3 \
    --observe
```

This will:
1. Load existing iter_0 trajectories/results from the baseline directory
2. Skip predict/evaluate for iter_0 (saves time and API costs)
3. For resolved instances: mark as resolved and skip further iterations
4. For unresolved instances: run learn phase, then continue with iter_1, iter_2, etc.

The baseline directory must have the following structure:
```
data/run_baseline_qwen3coder/
├── config.json
├── statistics.json
└── princeton-nlp__SWE-bench_Lite/
    ├── trajectories/
    │   └── {instance_id}/
    │       └── iter_0.json
    └── results/
        └── {instance_id}/
            └── iter_0.json
```

Use the provided transformation script to convert baseline data:
```bash
uv run python scripts/transform_baseline_to_run_format.py --in-place
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
    ├── statistics.json                # Counts, resolved/unresolved lists, observability URL
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

### statistics.json Format

```json
{
  "run_name": "mini-swe-v1-skillbook-learning",
  "timestamp": "2026-03-22T05:45:00.000000",
  "total_instances": 300,
  "resolved_count": 52,
  "unresolved_count": 248,
  "resolution_rate": 0.173,
  "resolved_ids": ["django__django-11039", ...],
  "unresolved_ids": ["astropy__astropy-12907", ...],
  "config": {
    "max_attempts": 3,
    "skillbook_mode": "per_instance"
  },
  "observability_project_url": "http://localhost:5173/projects/run_20260322_054500",
  "baseline_dir": "data/run_baseline_qwen3coder"  // Only if --baseline-dir was used
}
```

## Dependencies

- **mini-swe-agent** (v1) - SWE resolution agent
- **ace-framework** - Skillbook learning
- **swebench** - Docker harness for evaluation
- **litellm** - Unified LLM API (Z.AI/vLLM)
- **opik** - LLM observability (optional)

## License

MIT
