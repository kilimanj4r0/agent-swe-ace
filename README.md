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

# Prepare Docker images (required for evaluation)
uv run python scripts/prepare_images.py                    # All instances
uv run python scripts/prepare_images.py --instances django__django-11099  # Specific

# Run experiment (uses config.yaml as base)
uv run python -m src.cli.commands --max-instances 10

# Run with override config (deep-merged on top of config.yaml)
uv run python -m src.cli.commands --config configs/agent-glm-ace-glm.yaml

# Run with override config + CLI args (CLI args take precedence)
uv run python -m src.cli.commands --config configs/agent-glm-ace-glm.yaml --max-instances 10

# Run with observability
uv run python -m src.cli.commands --observe
```

### Docker Image Preparation

Evaluation requires SWE-bench Docker images. Prepare them with:

```bash
# Prepare all images (reads namespace from config.yaml)
uv run python scripts/prepare_images.py

# Prepare specific instances
uv run python scripts/prepare_images.py --instances django__django-11099 django__django-12345

# Options
uv run python scripts/prepare_images.py --workers 8     # Parallel builds
uv run python scripts/prepare_images.py --force         # Rebuild
```

The default registry is `ghcr.io/epoch-research/`. Set `environment.namespace` in `config.yaml` to use a different registry.

**7 instances are excluded** from experiments due to unfixable upstream build failures (see `benchmark.exclude_instances` in `config.yaml`). Details in [swe-bench-lite_exclude.md](../swe-bench-lite_exclude.md). A plan to patch these exists at [docs/plan_swebench_build_patches.md](docs/plan_swebench_build_patches.md).

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
    --config configs/agent-qwen3-ace-qwen3.yaml \
    --baseline-dir data/run_baseline_qwen3coder \
    --max-attempts 2 \
    --max-instances 50 \
    --custom-swe-learn \
    --observe
    # --instance astropy__astropy-14182 \
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

Configuration is loaded in layers (later overrides earlier):

1. **`config.yaml`** — base defaults (always loaded)
2. **`--config <file>`** — override config, deep-merged on top of base
3. **CLI arguments** — override specific values

Override configs (in `configs/`) only need to specify keys they change:

```yaml
# configs/agent-glm-ace-glm.yaml — overrides LLM and step limit only
experiment:
  name: "agent-glm-ace-glm"
  max_attempts: 5

agent:
  step_limit: 250

llm:
  agent:
    provider: "zai"
    model: "glm-4.7-flash"
  ace:
    provider: "zai"
    model: "glm-4.7-flash"
```

Full options in `config.yaml`:

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
  max_instances: null              # null = all instances
  exclude_instances:               # Instances with broken Docker images
    - pylint-dev__pylint-7114
    - sympy__sympy-20590
    # ... see config.yaml for full list

environment:
  namespace: "ghcr.io/epoch-research/"  # Docker registry prefix

evaluation:
  use_docker: true
  timeout: 1800                     # Seconds per patch evaluation
  rm_image: false                   # Keep images (avoids rebuilds)

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
