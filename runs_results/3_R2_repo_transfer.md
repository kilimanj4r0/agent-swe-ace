# R2 — Repository-level transfer (no retrieval)

**Goal:** Test whether a skillbook learned on each repository's train instances transfers to that repository's held-out val (full skillbook, no retrieval), across learn modes, benchmarks, splits, and self-learned vs distilled skills.

*Two-phase: per-repo skillbook learned on train, applied whole to val. `valB` = empty-skillbook val, `valS` = skillbook val. `-` = column not applicable to that row (pass@1 vs pass@5). Numeric cells left empty (to fill from `compare_runs.py`).*

---

### Table R2.1 — Old split 020, per-repo transfer (pass@3)
**Goal (Lite):** First per-repo transfer check on Lite (old split, pass@3): does a per-repo skillbook beat the empty-skillbook val baseline for default vs SWE learn.
**Goal (Verified):** Repeat the old-split per-repo transfer check on Verified at pass@3 to see whether the no-transfer / slight-harm pattern holds on the harder benchmark.
*Old split: valB is each run's own (no aggregated reference). p@3 run → show pass@1, pass@3, and per-attempt avg for both phases.*

| Run | Benchmark | Learn | valB p1 % | valB p3 % | avg valB % | valS p1 % | valS p3 % | avg valS % | Δp3 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| `run_20260521_034008_completed_qwen3_repos_split_default` | Lite | default | | | | | | | | |
| `run_20260521_034013_completed_qwen3_repos_split_swe` | Lite | S | | | | | | | | |
| `run_20260520_172940_completed_qwen3_repos_split_default_verified` | Verified | default | | | | | | | | |
| `run_20260520_172953_completed_qwen3_repos_split_swe_verified` | Verified | S | | | | | | | | |

---

### Table R2.2 — Teacher-trajectory distillation, per-repo transfer (pass@1)
**Goal:** Test whether skills distilled from teacher (proprietary-LLM) trajectories transfer better than self-learned ones, leveraging the teacher's far higher train solve rate.
*Caption note (to add with values): teacher train solve rate ≈ 78.2% vs ≈ 21% self-learned. Pass@1 (single val attempt) → no avg/p@k columns.*

| Run | Learn | Train n/N % | sk | valB % | valS % | Δ (pp) |
|---|---|---|---|---|---|---|
| `run_20260524_021832_completed_repos_split_default_distil_verified` | default | | | | | |
| `run_20260524_073812_completed_repos_split_swe_distil_verified` | S | | | | | |

---

### Table R2.3 — split025 per-repo transfer (primary; pass@1 & pass@5)
**Goal:** Re-run per-repo transfer on the deterministic split025 (8 repos) at pass@1 and pass@5 to obtain the paper's primary per-repo transfer numbers, including the QNext backbone.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4); QNext = 61.9 / 63.7. Pass@1 rows (`_vpk5` absent) have `-` in the p5/avg/Δ cells.*

| Run | Backbone | Learn | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|
| `run_20260602_005227_completed_qwen3_repos_split025_default_verified` | Q30 | default | | - | - | - | - |
| `run_20260602_103702_completed_qwen3_repos_split025_swe_verified` | Q30 | S | | - | - | - | - |
| `run_20260605_111658_completed_qwen3_repos_split025_swe_verified_vpk5` | Q30 | S | | | | | |
| `run_20260605_111708_completed_qwen3_repos_split025_default_verified_vpk5` | Q30 | default | | | | | |
| `run_20260622_105529_completed_qwen3next_repos_split025_default_verified_vpk5` | QNext | default | | | | | |

---

### Table R2.4 — Per-repo transfer breakdown (split025, pass@5; Q30 default vs SWE)
**Goal:** Show how skillbook transfer is distributed across the 8 repos — which repositories benefit from same-repo skills and which do not (the aggregate Δ hides this spread).
*Per-repo rates come from each run's `summary.per_repo` (only populated for pass@5/vpk5 repos runs). valB and train solve are shared across default/SWE (same Q30 backbone, empty-skillbook val); valS differs by learn mode. Source: `run_20260605_111708` (default), `run_20260605_111658` (SWE).*

| Repo | Train n/N % | valB % | valS default % | Δ default (pp) | valS SWE % | Δ SWE (pp) |
|---|---|---|---|---|---|---|
| sympy/sympy | | | | | | |
| sphinx-doc/sphinx | | | | | | |
| matplotlib/matplotlib | | | | | | |
| scikit-learn/scikit-learn | | | | | | |
| astropy/astropy | | | | | | |
| pydata/xarray | | | | | | |
| pytest-dev/pytest | | | | | | |
| django/django | | | | | | |

---

### Table R2.5 — Insight: per-repo Δ by backbone / learn mode (split025, pass@5)
**Goal:** Compare, repo by repo, how transfer Δ changes with learn mode (default vs SWE) and backbone (Q30 vs QNext) — to see whether SWE skills or a stronger backbone widen the set of repos that benefit.
*Δ = valS − valB per repo (pass@5). Source: `run_20260605_111708` (Q30 default), `run_20260605_111658` (Q30 SWE), `run_20260622_105529` (QNext default).*

| Repo | Δ Q30 default (pp) | Δ Q30 SWE (pp) | Δ QNext default (pp) |
|---|---|---|---|
| sympy/sympy | | | |
| sphinx-doc/sphinx | | | |
| matplotlib/matplotlib | | | |
| scikit-learn/scikit-learn | | | |
| astropy/astropy | | | |
| pydata/xarray | | | |
| pytest-dev/pytest | | | |
| django/django | | | |
