# R3 — Global skillbook transfer (no retrieval)

**Goal:** Test whether one global skillbook learned across all repositories transfers to held-out val (full skillbook, no retrieval), in direct contrast to R2's per-repo setup.

*Two-phase: ONE shared skillbook learned across all train instances, applied whole to val. Direct counterpart to R2 (per-repo) and the no-retrieval control for R4. `-` = column not applicable to that row (pass@1 vs pass@5). Numeric cells filled from each run's `statistics.json` via `compare_runs.py`.*

---

### Table R3.1 — Old split 020, global transfer (pass@1)
**Goal:** First global-transfer check on the old split (Lite + Verified, pass@1): does a single shared cross-repo skillbook help or hurt val resolution.
*Old split: valB is each run's own. Pass@1 (single val attempt) → no avg/p@k columns.*

| Run | Benchmark | Train n/N % | sk | valB % | valS % | Δ (pp) |
|---|---|---|---|---|---|---|
| `run_20260429_111748_completed_global_split_default` | Lite | 36/234 15.4% | 661 | 8.6 | 12.1 | +3.4 |
| `run_20260523_182739_completed_global_split_default_verified` | Verified | 75/390 19.2% | 2140 | 18.6 | 16.5 | -2.1 |

---

### Table R3.2 — split025 global transfer (primary; pass@1 & pass@5)
**Goal:** Re-run global transfer on split025 at pass@1 and pass@5 for the paper's primary global numbers, including QNext, where full-skillbook conditioning turns negative at pass@5.
*valB = each run's own 5-attempt val_baseline (empty skillbook); valS = skillbook val. Δ = valS − valB (5v5, k-symmetric; valB p1 == avg valB, so Δavg = valS p1 − valB p1). Pass@1-only rows (`_vpk5` absent) show valB p1 / valS p1 only. Negative Δp5 on the Q30 full-skillbook rows is the headline that motivates the R4 retrieval sweep.*

| Run | Backbone | Learn | valB p1 (avg) % | valB p5 % | valS p1 (avg) % | valS p5 % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|
| `run_20260529_164021_completed_qwen3_global_split025_default_verified` | Q30 | default | 16.0 | - | 18.5 | - | - | - |
| `run_20260602_004243_completed_qwen3_global_split025_swe_verified` | Q30 | S | 17.6 | - | 18.5 | - | - | - |
| `run_20260605_111718_completed_qwen3_global_split025_swe_verified_vpk5` | Q30 | S | 20.2 | 26.5 | 14.5 | 23.0 | -3.5 | -5.7 |
| `run_20260605_111733_completed_qwen3_global_split025_default_verified_vpk5` | Q30 | default | 21.6 | 27.4 | 16.1 | 22.1 | -5.3 | -5.5 |
| `run_20260615_125626_completed_qwen3next_global_split025_default_verified_vpk5` | QNext | default | 60.5 | 63.7 | 61.2 | 63.7 | +0.0 | +0.7 |
