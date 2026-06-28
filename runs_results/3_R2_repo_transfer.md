# R2 — Repository-level transfer (no retrieval)

**Goal:** Test whether a skillbook learned on each repository's train instances transfers to that repository's held-out val (full skillbook, no retrieval), across learn modes, benchmarks, splits, and self-learned vs distilled skills.

*Two-phase: per-repo skillbook learned on train, applied whole to val. `valB` = empty-skillbook val, `valS` = skillbook val. `-` = column not applicable to that row (pass@1 vs pass@5). Numeric cells filled from each run's `statistics.json` / per-repo files via `compare_runs.py`.*

---

### Table R2.1 — Old split 020, per-repo transfer (pass@3)
**Goal (Lite):** First per-repo transfer check on Lite (old split, pass@3): does a per-repo skillbook beat the empty-skillbook val baseline for default vs SWE learn.
**Goal (Verified):** Repeat the old-split per-repo transfer check on Verified at pass@3 to see whether the no-transfer / slight-harm pattern holds on the harder benchmark.
*Split note: these repos runs use a `val_ratio=0.2` seeded split (no manifest) — 8 repos / val=90 (Verified), 6 repos / val=50 (Lite) — distinct from the 12-repo old split in legend S.1/S.2 (which is the R3.1 global partition) and from split025; their instance sets are not listed in S.2. valB is each run's own (no aggregated reference). p@3 run → show pass@1, pass@3, and per-attempt avg for both phases.*

| Run | Benchmark | Learn | valB p1 % | valB p3 % | avg valB % | valS p1 % | valS p3 % | avg valS % | Δp3 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| `run_20260521_034008_completed_qwen3_repos_split_default` | Lite | default | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 9.3 | +0.0 | -0.7 |
| `run_20260521_034013_completed_qwen3_repos_split_swe` | Lite | S | 12.0 | 14.0 | 10.0 | 12.0 | 12.0 | 12.0 | -2.0 | +2.0 |
| `run_20260520_172940_completed_qwen3_repos_split_default_verified` | Verified | default | 18.9 | 22.2 | 18.9 | 15.6 | 22.2 | 17.0 | +0.0 | -1.9 |
| `run_20260520_172953_completed_qwen3_repos_split_swe_verified` | Verified | S | 20.0 | 26.7 | 20.7 | 17.8 | 22.2 | 17.8 | -4.4 | -3.0 |

---

### Table R2.2 — Teacher-trajectory distillation, per-repo transfer (pass@1)
**Goal:** Test whether skills distilled from teacher (proprietary-LLM) trajectories transfer better than self-learned ones, leveraging the teacher's far higher train solve rate.
*Caption note (to add with values): teacher train solve rate ≈ 78.2% vs ≈ 21% self-learned. Pass@1 (single val attempt) → no avg/p@k columns.*

| Run | Learn | Train n/N % | sk | valB % | valS % | Δ (pp) |
|---|---|---|---|---|---|---|
| `run_20260524_021832_completed_repos_split_default_distil_verified` | default | 295/377 78.2% | 617 | 18.9 | 26.7 | +7.8 |
| `run_20260524_073812_completed_repos_split_swe_distil_verified` | S | 295/377 78.2% | 484 | 21.1 | 21.1 | +0.0 |

---

### Table R2.3 — split025 per-repo transfer (primary; pass@1 & pass@5)
**Goal:** Re-run per-repo transfer on the deterministic split025 (8 repos) at pass@1 and pass@5 to obtain the paper's primary per-repo transfer numbers, including the QNext backbone.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4); QNext = 61.9 / 63.7. Pass@1 rows (`_vpk5` absent) have `-` in the p5/avg/Δ cells.*

| Run | Backbone | Learn | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|
| `run_20260602_005227_completed_qwen3_repos_split025_default_verified` | Q30 | default | 22.1 | - | - | - | - |
| `run_20260602_103702_completed_qwen3_repos_split025_swe_verified` | Q30 | S | 31.0 | - | - | - | - |
| `run_20260605_111658_completed_qwen3_repos_split025_swe_verified_vpk5` | Q30 | S | 22.1 | 26.5 | 23.2 | -12.1 | +2.8 |
| `run_20260605_111708_completed_qwen3_repos_split025_default_verified_vpk5` | Q30 | default | 21.2 | 23.9 | 20.2 | -14.8 | -0.2 |
| `run_20260622_105529_completed_qwen3next_repos_split025_default_verified_vpk5` | QNext | default | 58.4 | 68.1 | 63.4 | +4.4 | +2.8 |

---

### Table R2.4 — Per-repo transfer breakdown (split025, pass@5; Q30 default vs SWE)
**Goal:** Show how skillbook transfer is distributed across the 8 repos — which repositories benefit from same-repo skills and which do not (the aggregate Δ hides this spread).
*Per-repo rates come from each run's `summary.per_repo` (only populated for pass@5/vpk5 repos runs). valB and train solve are shared across default/SWE (same Q30 backbone, empty-skillbook val); valS differs by learn mode. Source: `run_20260605_111708` (default), `run_20260605_111658` (SWE).*

| Repo | Train n/N % | valB % | valS default % | Δ default (pp) | valS SWE % | Δ SWE (pp) |
|---|---|---|---|---|---|---|
| sympy/sympy | 8/56 14.3% | 33.3 | 33.3 | +0.0 | 38.9 | +0.0 |
| sphinx-doc/sphinx | 1/33 3.0% | 18.2 | 18.2 | +0.0 | 18.2 | +0.0 |
| matplotlib/matplotlib | 3/24 12.5% | 12.5 | 12.5 | +0.0 | 25.0 | +0.0 |
| scikit-learn/scikit-learn | 7/19 36.8% | 33.3 | 33.3 | +0.0 | 50.0 | +0.0 |
| astropy/astropy | 2/16 12.5% | 25.0 | 25.0 | +0.0 | 25.0 | +0.0 |
| pydata/xarray | 7/17 41.2% | 60.0 | 60.0 | +0.0 | 20.0 | +0.0 |
| pytest-dev/pytest | 4/15 26.7% | 25.0 | 25.0 | +0.0 | 0.0 | +0.0 |
| django/django | 51/174 29.3% | 17.5 | 19.3 | +1.8 | 24.6 | +1.8 |

*Note: the `valB %` column shows the default run's per-repo empty-skillbook val (reported once as the shared Q30 baseline); each Δ is computed against that run's own per-repo valB, so `Δ SWE` here equals the `Δ Q30 SWE` column of R2.5 (the two runs' valB differ by ±1 instance on the small-repo val sets, e.g. sympy 6/18 vs 7/18).*

---

### Table R2.5 — Insight: per-repo Δ by backbone / learn mode (split025, pass@5)
**Goal:** Compare, repo by repo, how transfer Δ changes with learn mode (default vs SWE) and backbone (Q30 vs QNext) — to see whether SWE skills or a stronger backbone widen the set of repos that benefit.
*Δ = valS − valB per repo (pass@5). Source: `run_20260605_111708` (Q30 default), `run_20260605_111658` (Q30 SWE), `run_20260622_105529` (QNext default).*

| Repo | Δ Q30 default (pp) | Δ Q30 SWE (pp) | Δ QNext default (pp) |
|---|---|---|---|
| sympy/sympy | +0.0 | +0.0 | +5.6 |
| sphinx-doc/sphinx | +0.0 | +0.0 | +0.0 |
| matplotlib/matplotlib | +0.0 | +0.0 | +0.0 |
| scikit-learn/scikit-learn | +0.0 | +0.0 | +0.0 |
| astropy/astropy | +0.0 | +0.0 | +0.0 |
| pydata/xarray | +0.0 | +0.0 | +0.0 |
| pytest-dev/pytest | +0.0 | +0.0 | +25.0 |
| django/django | +1.8 | +1.8 | +5.3 |
