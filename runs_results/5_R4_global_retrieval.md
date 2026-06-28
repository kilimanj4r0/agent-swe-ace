# R4 — Global Top-k retrieval (validation-only)

**Goal:** Test whether retrieving only the top-k most relevant skills per instance, instead of conditioning on the whole global book, recovers the R3 losses by sweeping retriever type, k, and learn mode.

*Validation-only: global skillbook reused from a trained source dir (no re-learning); retriever selects top-k skills per instance. R3 (no retrieval) is the control. `top_k` = retriever's skill-selection k. Pass@1 rows (`_vpk5` absent) have `-` in the p5/avg/Δ cells. Numeric cells filled from each run's `statistics.json` via `compare_runs.py`.*

---

### Table R4.1 — Global retrieval, Default skillbook (Q30 + QNext)
**Goal:** Default-learn global retrieval grid: which retriever and k best select relevant skills from the default global skillbook at pass@1 and pass@5, for Q30 and QNext.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4); QNext = 61.9 / 63.7.*

| Run | Backbone | Retriever | top_k | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|---|
| `run_20260603_010854_completed_qwen3_global_split025_retk5_default_verified` | Q30 | LLM | 5 | 19.3 | - | - | - | - |
| `run_20260603_164304_completed_qwen3_global_split025_retk10_default_verified` | Q30 | LLM | 10 | 25.2 | - | - | - | - |
| `run_20260605_190212_completed_qwen3_global_split025_retk5_defalut_verified_vpk5` | Q30 | LLM | 5 | 20.4 | 27.4 | 20.9 | -11.3 | +0.5 |
| `run_20260608_085351_completed_qwen3_global_split025_retk20_defalut_verified_vpk5` | Q30 | LLM | 20 | 23.0 | 28.3 | 23.9 | -10.4 | +3.5 |
| `run_20260617_215818_completed_qwen3_global_split025_retk5bm25_default_verified_vpk5` | Q30 | BM25 | 5 | 27.4 | 29.2 | 26.2 | -9.5 | +5.8 |
| `run_20260617_215828_completed_qwen3_global_split025_retk5emb_default_verified_vpk5` | Q30 | Embedding | 5 | 29.2 | 31.9 | 26.0 | -6.8 | +5.6 |
| `run_20260617_215839_completed_qwen3_global_split025_retk5random_default_verified_vpk5` | Q30 | Random | 5 | 23.0 | 31.9 | 25.7 | -6.8 | +5.3 |
| `run_20260624_123705_completed_qwen3next_global_split025_retk5random_default_verified_vpk5` | QNext | Random | 5 | 61.9 | 64.6 | 61.4 | +0.9 | +0.9 |
| `run_20260624_123715_completed_qwen3next_global_split025_retk5_default_verified_vpk5` | QNext | LLM | 5 | 58.4 | 65.5 | 60.0 | +1.8 | -0.5 |
| `run_20260624_123726_completed_qwen3next_global_split025_retk5bm25_default_verified_vpk5` | QNext | BM25 | 5 | 62.8 | 68.1 | 63.5 | +4.4 | +3.0 |
| `run_20260626_071054_completed_qwen3next_global_split025_retk5emb_default_verified_vpk5` | QNext | Embedding | 5 | 62.8 | 66.4 | 63.9 | +2.7 | +3.4 |

---

### Table R4.1b — Global retrieval, Default skillbook, temperature sensitivity (QNext, BM25 k=5, pass@5)
**Goal:** Sensitivity check on the strongest global-retrieval cell (QNext, BM25 k=5): whether raising the agent sampling temperature to 1.0 changes the retrieval benefit at pass@5.
*Row 1 (temp 0.0) is the same run as the QNext BM25 row in R4.1, repeated here for direct comparison.*

| Run | Temperature | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|
| `run_20260624_123726_completed_qwen3next_global_split025_retk5bm25_default_verified_vpk5` | 0.0 | 62.8 | 68.1 | 63.5 | +4.4 | +3.0 |
| `run_20260626_205552_completed_qwen3next_global_split025_retk5bm25_default_verified_vpk5_t1` | 1.0 | 61.9 | 64.6 | 61.9 | +0.9 | +1.4 |

---

### Table R4.2 — Global retrieval, SWE skillbook (Q30)
**Goal:** Repeat the global retrieval sweep for the SWE-learned skillbook (Q30) to see whether the best retriever/k and the size of the gain depend on learn mode.
*valB reference (fixed, in caption): Q30 = pass@1 20.4 / pass@5 38.7 (avg 20.4).*

| Run | Retriever | top_k | valS p1 % | valS p5 % | avg valS % | Δp5 (pp) | Δavg (pp) |
|---|---|---|---|---|---|---|---|
| `run_20260603_010859_completed_qwen3_global_split025_retk5_swe_verified` | LLM | 5 | 26.1 | - | - | - | - |
| `run_20260603_164314_completed_qwen3_global_split025_retk10_swe_verified` | LLM | 10 | 25.2 | - | - | - | - |
| `run_20260605_190229_completed_qwen3_global_split025_retk5_swe_verified_vpk5` | LLM | 5 | 24.8 | 26.5 | 22.8 | -12.1 | +2.4 |
| `run_20260608_085407_completed_qwen3_global_split025_retk20_swe_verified_vpk5` | LLM | 20 | 22.1 | 25.7 | 22.8 | -13.0 | +2.4 |
| `run_20260617_215909_completed_qwen3_global_split025_retk5random_swe_verified_vpk5` | Random | 5 | 26.5 | 31.0 | 27.1 | -7.7 | +6.7 |
| `run_20260620_150302_completed_qwen3_global_split025_retk5bm25_swe_verified_vpk5` | BM25 | 5 | 28.3 | 30.1 | 27.1 | -8.6 | +6.7 |
| `run_20260624_150103_completed_qwen3_global_split025_retk5emb_swe_verified_vpk5` | Embedding | 5 | 24.8 | 30.1 | 24.4 | -8.6 | +4.0 |
