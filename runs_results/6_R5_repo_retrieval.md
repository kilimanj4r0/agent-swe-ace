# R5 — Repository-level Top-k retrieval (validation-only)

**Goal:** Test top-k retrieval over per-repo skillbooks (instead of global) to determine whether scoping memory to the repository makes retrieval more or less effective than in R4.

*Validation-only: per-repo skillbook reused from a trained source dir; retriever selects top-k skills per instance. R2 (no retrieval) is the control; read against R4 to compare repo-scoped vs global retrieval. `top_k` = retriever's skill-selection k. Pass@1 rows (`_vpk5` absent) have `-` in the p5/avg/Δ cells. Numeric cells left empty (to fill from `compare_runs.py`).*

---

### Table R5.1 — Per-repo retrieval, Default skillbook (Q30 + QNext)
**Goal:** Default-learn per-repo retrieval grid: which retriever and k best select relevant skills from each repository's default skillbook at pass@1 and pass@5, for Q30 and QNext.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4); QNext = 61.9 / 63.7.*

| Run | Backbone | Retriever | top_k | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|
| `run_20260603_010904_completed_qwen3_repos_split025_retk5_default_verified` | Q30 | LLM | 5 | | - | - | - | - |
| `run_20260603_164243_completed_qwen3_repos_split025_retk10_default_verified` | Q30 | LLM | 10 | | - | - | - | - |
| `run_20260609_155815_completed_qwen3_repos_split025_retk5_default_verified_vpk5` | Q30 | LLM | 5 | | | | | |
| `run_20260609_155956_completed_qwen3_repos_split025_retk20_default_verified_vpk5` | Q30 | LLM | 20 | | | | | |
| `run_20260617_215849_completed_qwen3_repos_split025_retk5bm25_default_verified_vpk5` | Q30 | BM25 | 5 | | | | | |
| `run_20260617_215858_completed_qwen3_repos_split025_retk5random_default_verified_vpk5` | Q30 | Random | 5 | | | | | |
| `run_20260620_150312_completed_qwen3_repos_split025_retk5emb_default_verified_vpk5` | Q30 | Embedding | 5 | | | | | |
| `run_20260624_213259_completed_qwen3next_repos_split025_retk5_default_verified_vpk5` | QNext | LLM | 5 | | | | | |
| `run_20260624_213314_completed_qwen3next_repos_split025_retk5random_default_verified_vpk5` | QNext | Random | 5 | | | | | |
| `run_20260624_213319_completed_qwen3next_repos_split025_retk5bm25_default_verified_vpk5` | QNext | BM25 | 5 | | | | | |
| `run_20260627_080359_completed_qwen3next_repos_split025_retk5emb_default_verified_vpk5` | QNext | Embedding | 5 | | | | | |

---

### Table R5.2 — Per-repo retrieval, SWE skillbook (Q30)
**Goal:** Repeat the per-repo retrieval sweep for the SWE-learned skillbook (Q30) to compare learn-mode sensitivity against the default-learn grid above.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4).*

| Run | Retriever | top_k | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|
| `run_20260603_010908_completed_qwen3_repos_split025_retk5_swe_verified` | LLM | 5 | | - | - | - | - |
| `run_20260603_164253_completed_qwen3_repos_split025_retk10_swe_verified` | LLM | 10 | | - | - | - | - |
| `run_20260609_155832_completed_qwen3_repos_split025_retk5_swe_verified_vpk5` | LLM | 5 | | | | | |
| `run_20260609_160011_completed_qwen3_repos_split025_retk20_swe_verified_vpk5` | LLM | 20 | | | | | |
| `run_20260620_150322_completed_qwen3_repos_split025_retk5bm25_swe_verified_vpk5` | BM25 | 5 | | | | | |
| `run_20260620_163227_completed_qwen3_repos_split025_retk5random_swe_verified_vpk5` | Random | 5 | | | | | |
| `run_20260625_170035_completed_qwen3_repos_split025_retk5emb_swe_verified_vpk5` | Embedding | 5 | | | | | |

---

### Table R5.3 — Insight: per-repo retrieval Δ by retriever (split025, pass@5, Q30 default)
**Goal:** Reveal the repo × retriever interaction hidden by the aggregate — which retriever rescues which repositories, and where retrieval (vs full skillbook) helps or hurts.
*Δ = valS − valB per repo (pass@5), from each run's `summary.per_repo`. "no-retrieval" = full skillbook (R2 control). Source: `run_20260605_111708` (no-ret), `run_20260609_155815` (LLM), `run_20260617_215849` (BM25), `run_20260620_150312` (Embedding), `run_20260617_215858` (Random).*

| Repo | Δ no-retrieval (pp) | Δ LLM k5 (pp) | Δ BM25 k5 (pp) | Δ Embedding k5 (pp) | Δ Random k5 (pp) |
|---|---|---|---|---|---|
| sympy/sympy | | | | | |
| sphinx-doc/sphinx | | | | | |
| matplotlib/matplotlib | | | | | |
| scikit-learn/scikit-learn | | | | | |
| astropy/astropy | | | | | |
| pydata/xarray | | | | | |
| pytest-dev/pytest | | | | | |
| django/django | | | | | |
