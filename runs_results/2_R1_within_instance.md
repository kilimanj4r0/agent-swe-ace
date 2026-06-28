# R1 — Within-instance / test-time scaling (single-phase)

**Goal:** Quantify whether retry-driven test-time scaling — more attempts per instance with a per-instance skillbook accumulating across retries — improves single-instance resolution across attempt counts, learn modes, and backbones.

*Single-phase runs (no train/val split); per-instance skillbook accumulates across retries. `iter0` = first-attempt (pass@1) raw ability; `pass@N` = any-of-N resolved. Numeric cells filled from each run's `statistics.json` via `compare_runs.py`.*

---

### Table R1.1 — Lite, within-instance scaling (pass@1)
**Goal:** Measure how much allowing two or more retries per instance, with the skillbook accumulating across attempts, lifts resolution above the single-attempt floor, across backbones and attempt counts.

| Run | Backbone | Learn | Attempts | iter0 (pass@1) % | Resolved n/N | pass@N % | Δ vs iter0 (pp) | sb-assisted |
|---|---|---|---|---|---|---|---|---|
| `run_20260426_210831_completed_baseline` | Q30 | no-SB | 1 | 13.7 | 40/292 | 13.7 | +0.0 | 0 |
| `run_20260404_150133_completed_qwen3_1a_swe` | Q30 | S | 1 | 9.6 | 28/292 | 9.6 | +0.0 | 0 |
| `run_20260404_150204_completed_qwen3_1a_default` | Q30 | default | 1 | 7.2 | 21/292 | 7.2 | +0.0 | 0 |
| `run_20260414_015144_completed_glm_4a_swe` | GLM | S | 4 | 34.2 | 102/292 | 34.9 | +0.7 | 2 |
| `run_20260414_015225_completed_glm_4a_default` | GLM | default | 4 | 37.0 | 110/292 | 37.7 | +0.7 | 2 |
| `run_20260415_020217_completed_qwen3next_4a_default` | QNext | default | 4 | 38.0 | 117/292 | 40.1 | +2.1 | 6 |
| `run_20260415_020540_completed_qwen3next_4a_swe` | QNext | S | 4 | 37.7 | 116/292 | 39.7 | +2.1 | 6 |
| `run_20260526_133345_completed_qwen3_4a_no_skillbook` | Q30 | no-SB | 4 | 14.4 | 76/292 | 26.0 | +11.6 | 34 |
| `run_20260426_211426_completed_qwen3_4a_swe` | Q30 | S | 4 | 14.0 | 59/292 | 20.2 | +6.2 | 18 |
| `run_20260426_211500_completed_qwen3_4a_default` | Q30 | default | 4 | 13.4 | 54/292 | 18.5 | +5.1 | 15 |
| `run_20260525_133304_completed_qwen3_qwen3next_4a_default` | Q30/QNext | default | 4 | 13.4 | 82/292 | 28.1 | +14.7 | 43 |
| `run_20260402_235422_completed_qwen3_6a_default` | Q30 | default | 6 | 16.4 | 79/292 | 27.1 | +10.6 | 31 |
| `run_20260402_235456_completed_qwen3_6a_swe` | Q30 | S | 6 | 16.4 | 78/292 | 26.7 | +10.3 | 30 |

---

### Table R1.2 — Verified, 4-attempt ablation (pass@4)
**Goal:** Repeat the 4-attempt default/SWE/no-SB comparison on the harder Verified split to test whether within-instance learning and the skillbook ablation generalize off Lite.

| Run | Backbone | Learn | iter0 (pass@1) % | Resolved n/N | pass@4 % | Δ vs iter0 (pp) | sb-assisted |
|---|---|---|---|---|---|---|---|
| `run_20260521_154504_completed_qwen3_4a_no_skillbook_verified` | Q30 | no-SB | 19.1 | 126/487 | 25.9 | +6.8 | 33 |
| `run_20260520_123809_completed_qwen3_4a_swe_verified` | Q30 | S | 19.1 | 138/487 | 28.3 | +9.2 | 45 |
| `run_20260520_144216_completed_qwen3_4a_default_verified` | Q30 | default | 19.1 | 160/487 | 32.9 | +13.8 | 67 |
| `run_20260524_160825_completed_qwen3_qwen3next_4a_default_verified` | Q30/QNext | default | 19.1 | 160/487 | 32.9 | +13.8 | 67 |

---

### Table R1.3 — Aggregated no-skillbook test-time scaling (split025 val, n=113)
**Goal:** Quantify pure test-time scaling with no skillbook — empty-skillbook resolution scaled to 60 attempts per instance (12 runs × 5) — which doubles as the stable, low-variance Δ reference baseline for R2–R5.
*QNext global skillbook aggregate (`val_skillbook_aggregated_split025_vpk5_qwen3next_global_default`, n=4×5) is the skillbook counterpart, kept as a commented reference.*

| Run | pass@1 % | pass@5 % | avg % | resolved-any (pass@60) % |
|---|---|---|---|---|
| `val_baseline_aggregated_split025_vpk5_qwen3` | 20.4 | 38.7 | 20.4 | 50.4 |

<!-- QNext global skillbook aggregate (commented reference, n=4×5=20 att/inst): `val_skillbook_aggregated_split025_vpk5_qwen3next_global_default` → pass@1 61.5 / pass@5 73.4 / avg 61.5 / resolved-any 76.1 -->

---

#### R1.3b — Significance vs this baseline (paired t-test)

[TODO]