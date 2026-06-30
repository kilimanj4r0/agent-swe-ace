# R5 — Repository-level Top-k retrieval (validation-only)

**Goal:** Test top-k retrieval over per-repo skillbooks (instead of global) to determine whether scoping memory to the repository makes retrieval more or less effective than in R4.

*Validation-only: per-repo skillbook reused from a trained source dir; retriever selects top-k skills per instance. R2 (no retrieval) is the control; read against R4 to compare repo-scoped vs global retrieval. `top_k` = retriever's skill-selection k. Pass@1 rows (`_vpk5` absent) have `-` in the p5/avg/Δ cells. Numeric cells filled from each run's `statistics.json` / per-repo files via `compare_runs.py`.*

---

### Table R5.1 — Per-repo retrieval, Default skillbook (Q30 + QNext)
**Goal:** Default-learn per-repo retrieval grid: which retriever and k best select relevant skills from each repository's default skillbook at pass@1 and pass@5, for Q30 and QNext.
*valB = each run's own 5-attempt val_baseline (empty skillbook); valS = skillbook val. Δ = valS − valB (5v5, k-symmetric; Δavg = valS p1 − valB p1). Pass@1-only rows (`_vpk5` absent) show valB p1 / valS p1 only.*

| Run | Backbone | Retriever | top_k | valB p1 (avg) % | valB p5 % | valS p1 (avg) % | valS p5 % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|---|
| `run_20260603_010904_completed_qwen3_repos_split025_retk5_default_verified` | Q30 | LLM | 5 | 15.9 | - | 23.9 | - | - | - |
| `run_20260603_164243_completed_qwen3_repos_split025_retk10_default_verified` | Q30 | LLM | 10 | 15.9 | - | 23.0 | - | - | - |
| `run_20260609_155815_completed_qwen3_repos_split025_retk5_default_verified_vpk5` | Q30 | LLM | 5 | 19.3 | 23.0 | 20.7 | 25.7 | +2.7 | +1.4 |
| `run_20260609_155956_completed_qwen3_repos_split025_retk20_default_verified_vpk5` | Q30 | LLM | 20 | 19.6 | 26.5 | 24.2 | 27.4 | +0.9 | +4.6 |
| `run_20260617_215849_completed_qwen3_repos_split025_retk5bm25_default_verified_vpk5` | Q30 | BM25 | 5 | 19.8 | 23.0 | 28.3 | 32.7 | +9.7 | +8.5 |
| `run_20260617_215858_completed_qwen3_repos_split025_retk5random_default_verified_vpk5` | Q30 | Random | 5 | 19.8 | 23.0 | 26.9 | 31.0 | +8.0 | +7.1 |
| `run_20260620_150312_completed_qwen3_repos_split025_retk5emb_default_verified_vpk5` | Q30 | Embedding | 5 | 19.8 | 23.0 | 25.7 | 31.0 | +8.0 | +5.8 |
| `run_20260624_213259_completed_qwen3next_repos_split025_retk5_default_verified_vpk5` | QNext | LLM | 5 | 60.5 | 63.7 | 65.3 | 68.1 | +4.4 | +4.8 |
| `run_20260624_213314_completed_qwen3next_repos_split025_retk5random_default_verified_vpk5` | QNext | Random | 5 | 60.5 | 63.7 | 63.9 | 68.1 | +4.4 | +3.4 |
| `run_20260624_213319_completed_qwen3next_repos_split025_retk5bm25_default_verified_vpk5` | QNext | BM25 | 5 | 60.5 | 63.7 | 64.1 | 67.3 | +3.5 | +3.5 |
| `run_20260627_080359_completed_qwen3next_repos_split025_retk5emb_default_verified_vpk5` | QNext | Embedding | 5 | 60.5 | 63.7 | 68.1 | 71.7 | +8.0 | +7.6 |

---

### Table R5.2 — Per-repo retrieval, SWE skillbook (Q30)
**Goal:** Repeat the per-repo retrieval sweep for the SWE-learned skillbook (Q30) to compare learn-mode sensitivity against the default-learn grid above.
*valB = each run's own 5-attempt val_baseline (empty skillbook); valS = skillbook val. Δ = valS − valB (5v5, k-symmetric; Δavg = valS p1 − valB p1). Pass@1-only rows (`_vpk5` absent) show valB p1 / valS p1 only.*

| Run | Retriever | top_k | valB p1 (avg) % | valB p5 % | valS p1 (avg) % | valS p5 % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|
| `run_20260603_010908_completed_qwen3_repos_split025_retk5_swe_verified` | LLM | 5 | 15.9 | - | 26.5 | - | - | - |
| `run_20260603_164253_completed_qwen3_repos_split025_retk10_swe_verified` | LLM | 10 | 15.9 | - | 22.1 | - | - | - |
| `run_20260609_155832_completed_qwen3_repos_split025_retk5_swe_verified_vpk5` | LLM | 5 | 19.6 | 24.8 | 23.0 | 25.7 | +0.9 | +3.4 |
| `run_20260609_160011_completed_qwen3_repos_split025_retk20_swe_verified_vpk5` | LLM | 20 | 19.8 | 26.5 | 23.2 | 26.5 | +0.0 | +3.4 |
| `run_20260620_150322_completed_qwen3_repos_split025_retk5bm25_swe_verified_vpk5` | BM25 | 5 | 21.1 | 25.7 | 22.7 | 27.4 | +1.8 | +1.6 |
| `run_20260620_163227_completed_qwen3_repos_split025_retk5random_swe_verified_vpk5` | Random | 5 | 21.1 | 25.7 | 21.9 | 26.5 | +0.9 | +0.9 |
| `run_20260625_170035_completed_qwen3_repos_split025_retk5emb_swe_verified_vpk5` | Embedding | 5 | 21.1 | 25.7 | 26.9 | 30.1 | +4.4 | +5.8 |

---

### Table R5.3 — Insight: per-repo retrieval Δ by retriever (split025, pass@1 & pass@5, Q30 default)
**Goal:** Reveal the repo × retriever interaction hidden by the aggregate — which retriever rescues which repositories, and where retrieval (vs full skillbook) helps or hurts — at both per-attempt reliability (pass@1) and any-of-5 breadth (pass@5).
*Δ = valS − valB per repo; each cell shows **Δp1 / Δp5** (pp) — Δp1 = valS pass@1 − valB pass@1 (per-attempt reliability), Δp5 = valS pass@5 − valB pass@5 (any-of-5 breadth). 5v5 (each run's own 5-attempt empty-skillbook valB); combinatorial pass@k (pass@1 == per-attempt avg; pass@5 == resolved-any rate). "no-retrieval" = full skillbook (R2 control). Source: `run_20260605_111708` (no-ret), `run_20260609_155815` (LLM), `run_20260617_215849` (BM25), `run_20260620_150312` (Embedding), `run_20260617_215858` (Random).*

| Repo | no-retrieval Δp1/Δp5 | LLM k5 Δp1/Δp5 | BM25 k5 Δp1/Δp5 | Embedding k5 Δp1/Δp5 | Random k5 Δp1/Δp5 |
|---|---|---|---|---|---|
| sympy/sympy | +1.1 / +0.0 | +7.8 / +5.6 | -5.6 / +0.0 | +1.1 / +0.0 | +3.3 / +5.6 |
| sphinx-doc/sphinx | +1.8 / +0.0 | +9.1 / +9.1 | +0.0 / +0.0 | -3.6 / +0.0 | -10.9 / -9.1 |
| matplotlib/matplotlib | +7.5 / +0.0 | -2.5 / +0.0 | +17.5 / +12.5 | +7.5 / +0.0 | +17.5 / +12.5 |
| scikit-learn/scikit-learn | -3.3 / +0.0 | +10.0 / +16.7 | +20.0 / +16.7 | +33.3 / +50.0 | +20.0 / +33.3 |
| astropy/astropy | +0.0 / +0.0 | -15.0 / +0.0 | -10.0 / +0.0 | -10.0 / +0.0 | -5.0 / +0.0 |
| pydata/xarray | -8.0 / +0.0 | -4.0 / +0.0 | -32.0 / -20.0 | -32.0 / -20.0 | -40.0 / -20.0 |
| pytest-dev/pytest | +10.0 / +0.0 | +5.0 / +0.0 | +35.0 / +25.0 | +50.0 / +50.0 | +35.0 / +25.0 |
| django/django | -0.7 / +1.8 | -1.1 / +0.0 | +15.1 / +15.8 | +7.4 / +8.8 | +11.9 / +10.5 |

*Takeaway: Δp1 (reliability) and Δp5 (breadth) rarely move together — the skillbook mostly sharpens per-attempt reliability without expanding the solved set (no-retrieval: matplotlib +7.5/0.0, pytest +10.0/0.0; only django converts reliability into breadth, +1.8 on Δp5). Genuinely new breadth (Δp5 > 0) is concentrated in BM25/Embedding/Random on the small repos — scikit-learn, pytest, matplotlib — where ±1 instance is ±17–50 pp; LLM k5's breadth gains are confined to sympy/sphinx/scikit-learn. xarray is uniformly negative on Δp1 (−8 to −40 pp) at near-flat breadth.*
