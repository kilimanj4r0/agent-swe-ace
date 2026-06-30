# R2 — Repository-level transfer (no retrieval)

**Goal:** Test whether a skillbook learned on each repository's train instances transfers to that repository's held-out val (full skillbook, no retrieval), across learn modes, benchmarks, splits, and self-learned vs distilled skills.

*Two-phase: per-repo skillbook learned on train, applied whole to val. `valB` = empty-skillbook val, `valS` = skillbook val. `-` = column not applicable to that row (pass@1 vs pass@5). Numeric cells filled from each run's `statistics.json` / per-repo files via `compare_runs.py`.*

---

### Table R2.1 — Old split 020, per-repo transfer (pass@3)
**Goal (Lite):** First per-repo transfer check on Lite (old split, pass@3): does a per-repo skillbook beat the empty-skillbook val baseline for default vs SWE learn.
**Goal (Verified):** Repeat the old-split per-repo transfer check on Verified at pass@3 to see whether the no-transfer / slight-harm pattern holds on the harder benchmark.
*Split note: these repos runs use a `val_ratio=0.2` seeded split (no manifest) — 8 repos / val=90 (Verified), 6 repos / val=50 (Lite) — distinct from the 12-repo old split in legend S.1/S.2 (which is the R3.1 global partition) and from split025; their instance sets are not listed in S.2. valB is each run's own (no aggregated reference). p@3 run → show pass@1 (avg), pass@3 for both phases (pass@1 == per-attempt avg, so shown once as `p1 (avg)`).*

| Run | Benchmark | Learn | valB p1 (avg) % | valB p3 % | valS p1 (avg) % | valS p3 % | Δp3 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|
| `run_20260521_034008_completed_qwen3_repos_split_default` | Lite | default | 10.0 | 10.0 | 9.3 | 10.0 | +0.0 | -0.7 |
| `run_20260521_034013_completed_qwen3_repos_split_swe` | Lite | S | 10.0 | 14.0 | 12.0 | 12.0 | -2.0 | +2.0 |
| `run_20260520_172940_completed_qwen3_repos_split_default_verified` | Verified | default | 18.9 | 22.2 | 17.0 | 22.2 | +0.0 | -1.9 |
| `run_20260520_172953_completed_qwen3_repos_split_swe_verified` | Verified | S | 20.7 | 26.7 | 17.8 | 22.2 | -4.4 | -3.0 |

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
*valB = each run's own 5-attempt val_baseline (empty skillbook); valS = skillbook val. Δ = valS − valB (5v5, k-symmetric; Δavg = valS p1 − valB p1). Pass@1-only rows (`_vpk5` absent) show valB p1 / valS p1 only. (The shared 60-attempt aggregated reference Q30 = 20.4 / 38.7 is in R6's 60v5 block; against it the Q30 full-book Δp5 are strongly negative — see R6 §5.)*

| Run | Backbone | Learn | valB p1 (avg) % | valB p5 % | valS p1 (avg) % | valS p5 % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|
| `run_20260602_005227_completed_qwen3_repos_split025_default_verified` | Q30 | default | 16.8 | - | 22.1 | - | - | - |
| `run_20260602_103702_completed_qwen3_repos_split025_swe_verified` | Q30 | S | 17.7 | - | 31.0 | - | - | - |
| `run_20260605_111658_completed_qwen3_repos_split025_swe_verified_vpk5` | Q30 | S | 21.1 | 25.7 | 23.2 | 26.5 | +0.8 | +2.1 |
| `run_20260605_111708_completed_qwen3_repos_split025_default_verified_vpk5` | Q30 | default | 19.8 | 23.0 | 20.2 | 23.9 | +0.9 | +0.4 |
| `run_20260622_105529_completed_qwen3next_repos_split025_default_verified_vpk5` | QNext | default | 60.5 | 63.7 | 63.4 | 68.1 | +4.4 | +2.9 |

---

### Table R2.4 — Per-repo transfer breakdown (split025, pass@1/avg; Q30 default/SWE + QNext default)
**Goal:** Show how per-attempt reliability transfer (pass@1/avg) distributes across the 8 repos for three conditions — Q30 default, Q30 SWE, and QNext default — to see which repos the skillbook helps vs hurts, per backbone/learn mode.

*All cells are pass@1 (avg) %; Δp1 in pp. valB = each run's own 5-attempt empty-skillbook val; valS = skillbook val; Δp1 = valS − valB (5v5, k-symmetric). Columns: `def` = Q30 default (`run_20260605_111708`), `SWE` = Q30 SWE (`run_20260605_111658`), `QNext` = QNext default (`run_20260622_105529`). Train/val/skillbook sizes are in legend tables S.3/S.4; pass@5 breadth Δp5 is ≈0 per repo (only django moves) and is omitted here — see R6 §5. Values via `compare_runs.py`.*

| Repo | def valB | def valS | def Δp1 | SWE valB | SWE valS | SWE Δp1 | QNext valB | QNext valS | QNext Δp1 |
|---|---|---|---|---|---|---|---|---|---|
| django/django | 15.1 | 14.4 | -0.7 | 16.8 | 20.4 | +3.6 | 58.6 | 62.8 | +4.2 |
| sympy/sympy | 30.0 | 31.1 | +1.1 | 33.3 | 34.4 | +1.1 | 78.9 | 81.1 | +2.2 |
| sphinx-doc/sphinx | 16.4 | 18.2 | +1.8 | 18.2 | 14.5 | -3.7 | 36.4 | 34.5 | -1.9 |
| scikit-learn/scikit-learn | 26.7 | 23.3 | -3.4 | 43.3 | 46.7 | +3.4 | 96.7 | 96.7 | +0.0 |
| matplotlib/matplotlib | 5.0 | 12.5 | +7.5 | 20.0 | 25.0 | +5.0 | 32.5 | 35.0 | +2.5 |
| pydata/xarray | 60.0 | 52.0 | -8.0 | 20.0 | 20.0 | +0.0 | 68.0 | 68.0 | +0.0 |
| astropy/astropy | 25.0 | 25.0 | +0.0 | 25.0 | 25.0 | +0.0 | 75.0 | 70.0 | -5.0 |
| pytest-dev/pytest | 15.0 | 25.0 | +10.0 | 0.0 | 0.0 | +0.0 | 50.0 | 65.0 | +15.0 |

*Takeaway: Δp1 is noisy on the small-repo val sets (±1 instance ≈ 10–25 pp on the 4–6-instance repos) with no consistent sign across conditions; django (57 val, half the set) is the only repo showing a stable positive transfer — for SWE (+3.6) and QNext (+4.2). The aggregate R2.3 gains come almost entirely from django.*
