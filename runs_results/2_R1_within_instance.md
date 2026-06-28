# R1 — Within-instance / test-time scaling (single-phase)

**Goal:** Quantify whether retry-driven test-time scaling — more attempts per instance with a per-instance skillbook accumulating across retries — improves single-instance resolution across attempt counts, learn modes, and backbones.

*Single-phase runs (no train/val split); per-instance skillbook accumulates across retries. `iter0` = first-attempt (pass@1) raw ability; `pass@N` = any-of-N resolved. Numeric cells left empty (to fill from `compare_runs.py`).*

---

### Table R1.1 — Lite, within-instance scaling (pass@1)
**Goal:** Measure how much allowing two or more retries per instance, with the skillbook accumulating across attempts, lifts resolution above the single-attempt floor, across backbones and attempt counts.

| Run | Backbone | Learn | Attempts | iter0 (pass@1) % | Resolved n/N | pass@N % | Δ vs iter0 (pp) | sb-assisted |
|---|---|---|---|---|---|---|---|---|
| `run_20260426_210831_completed_baseline` | Q30 | no-SB | 1 | | | | | |
| `run_20260404_150133_completed_qwen3_1a_swe` | Q30 | S | 1 | | | | | |
| `run_20260404_150204_completed_qwen3_1a_default` | Q30 | default | 1 | | | | | |
| `run_20260414_015144_completed_glm_4a_swe` | GLM | S | 4 | | | | | |
| `run_20260414_015225_completed_glm_4a_default` | GLM | default | 4 | | | | | |
| `run_20260415_020217_completed_qwen3next_4a_default` | QNext | default | 4 | | | | | |
| `run_20260415_020540_completed_qwen3next_4a_swe` | QNext | S | 4 | | | | | |
| `run_20260526_133345_completed_qwen3_4a_no_skillbook` | Q30 | no-SB | 4 | | | | | |
| `run_20260426_211426_completed_qwen3_4a_swe` | Q30 | S | 4 | | | | | |
| `run_20260426_211500_completed_qwen3_4a_default` | Q30 | default | 4 | | | | | |
| `run_20260525_133304_completed_qwen3_qwen3next_4a_default` | Q30/QNext | default | 4 | | | | | |
| `run_20260402_235422_completed_qwen3_6a_default` | Q30 | default | 6 | | | | | |
| `run_20260402_235456_completed_qwen3_6a_swe` | Q30 | S | 6 | | | | | |

---

### Table R1.2 — Verified, 4-attempt ablation (pass@4)
**Goal:** Repeat the 4-attempt default/SWE/no-SB comparison on the harder Verified split to test whether within-instance learning and the skillbook ablation generalize off Lite.

| Run | Backbone | Learn | iter0 (pass@1) % | Resolved n/N | pass@4 % | Δ vs iter0 (pp) | sb-assisted |
|---|---|---|---|---|---|---|---|
| `run_20260521_154504_completed_qwen3_4a_no_skillbook_verified` | Q30 | no-SB | | | | | |
| `run_20260520_123809_completed_qwen3_4a_swe_verified` | Q30 | S | | | | | |
| `run_20260520_144216_completed_qwen3_4a_default_verified` | Q30 | default | | | | | |
| `run_20260524_160825_completed_qwen3_qwen3next_4a_default_verified` | Q30/QNext | default | | | | | |

---

### Table R1.3 — Aggregated no-skillbook test-time scaling (split025 val, n=113)
**Goal:** Quantify pure test-time scaling with no skillbook — empty-skillbook resolution scaled to 60 attempts per instance (12 runs × 5) — which doubles as the stable, low-variance Δ reference baseline for R2–R5.
*QNext global skillbook aggregate (`val_skillbook_aggregated_split025_vpk5_qwen3next_global_default`, n=4×5) is the skillbook counterpart, kept as a commented reference.*

| Run | pass@1 % | pass@5 % | avg % | resolved-any (pass@60) % |
|---|---|---|---|---|
| `val_baseline_aggregated_split025_vpk5_qwen3` | | | | |
