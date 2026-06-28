# R3 — Global skillbook transfer (no retrieval)

**Goal:** Test whether one global skillbook learned across all repositories transfers to held-out val (full skillbook, no retrieval), in direct contrast to R2's per-repo setup.

*Two-phase: ONE shared skillbook learned across all train instances, applied whole to val. Direct counterpart to R2 (per-repo) and the no-retrieval control for R4. `-` = column not applicable to that row (pass@1 vs pass@5). Numeric cells left empty (to fill from `compare_runs.py`).*

---

### Table R3.1 — Old split 020, global transfer (pass@1)
**Goal:** First global-transfer check on the old split (Lite + Verified, pass@1): does a single shared cross-repo skillbook help or hurt val resolution.
*Old split: valB is each run's own. Pass@1 (single val attempt) → no avg/p@k columns.*

| Run | Benchmark | Train n/N % | sk | valB % | valS % | Δ (pp) |
|---|---|---|---|---|---|---|
| `run_20260429_111748_completed_global_split_default` | Lite | | | | | |
| `run_20260523_182739_completed_global_split_default_verified` | Verified | | | | | |

---

### Table R3.2 — split025 global transfer (primary; pass@1 & pass@5)
**Goal:** Re-run global transfer on split025 at pass@1 and pass@5 for the paper's primary global numbers, including QNext, where full-skillbook conditioning turns negative at pass@5.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4); QNext = 61.9 / 63.7. Pass@1 rows (`_vpk5` absent) have `-` in the p5/avg/Δ cells. Negative Δp5 here is the headline finding that motivates R4 retrieval.*

| Run | Backbone | Learn | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|
| `run_20260529_164021_completed_qwen3_global_split025_default_verified` | Q30 | default | | - | - | - | - |
| `run_20260602_004243_completed_qwen3_global_split025_swe_verified` | Q30 | S | | - | - | - | - |
| `run_20260605_111718_completed_qwen3_global_split025_swe_verified_vpk5` | Q30 | S | | | | | |
| `run_20260605_111733_completed_qwen3_global_split025_default_verified_vpk5` | Q30 | default | | | | | |
| `run_20260615_125626_completed_qwen3next_global_split025_default_verified_vpk5` | QNext | default | | | | | |
