# R1 — Within-instance / test-time scaling (single-phase)

**Goal:** Quantify whether retry-driven test-time scaling — more attempts per instance with a per-instance skillbook accumulating across retries — improves single-instance resolution across attempt counts, learn modes, and backbones.

*Single-phase runs (no train/val split); per-instance skillbook accumulates across retries. `iter0` = first-attempt (pass@1) raw ability; `pass@N` = any-of-N resolved. Numeric cells filled from each run's `statistics.json` via `compare_runs.py`.*

---

### Table R1.1 — Lite, within-instance scaling (pass@1)
**Goal:** Measure how much allowing two or more retries per instance, with the skillbook accumulating across attempts, lifts resolution above the single-attempt floor, across backbones and attempt counts.

| Run | Backbone | Learn | Attempts | temp | step | iter0 (pass@1) % | Resolved n/N | pass@N % | Δ vs iter0 (pp) | sb-assisted |
|---|---|---|---|---|---|---|---|---|---|---|
| `run_20260426_210831_completed_baseline` | Q30 | no-SB | 1 | 0.0 | 250 | 13.7 | 40/292 | 13.7 | +0.0 | 0 |
| `run_20260404_150133_completed_qwen3_1a_swe` | Q30 | S | 1 | 0.7 | 200 | 9.6 | 28/292 | 9.6 | +0.0 | 0 |
| `run_20260404_150204_completed_qwen3_1a_default` | Q30 | default | 1 | 0.7 | 200 | 7.2 | 21/292 | 7.2 | +0.0 | 0 |
| `run_20260414_015144_completed_glm_4a_swe` | GLM | S | 4 | 0.7 | 200 | 34.2 | 102/292 | 34.9 | +0.7 | 2 |
| `run_20260414_015225_completed_glm_4a_default` | GLM | default | 4 | 0.7 | 200 | 37.0 | 110/292 | 37.7 | +0.7 | 2 |
| `run_20260415_020217_completed_qwen3next_4a_default` | QNext | default | 4 | 0.7 | 200 | 38.0 | 117/292 | 40.1 | +2.1 | 6 |
| `run_20260415_020540_completed_qwen3next_4a_swe` | QNext | S | 4 | 0.7 | 200 | 37.7 | 116/292 | 39.7 | +2.1 | 6 |
| `run_20260526_133345_completed_qwen3_4a_no_skillbook` | Q30 | no-SB | 4 | 0.0 | 200 | 14.4 | 76/292 | 26.0 | +11.6 | 34 |
| `run_20260426_211426_completed_qwen3_4a_swe` | Q30 | S | 4 | 0.0 | 200 | 14.0 | 59/292 | 20.2 | +6.2 | 18 |
| `run_20260426_211500_completed_qwen3_4a_default` | Q30 | default | 4 | 0.0 | 200 | 13.4 | 54/292 | 18.5 | +5.1 | 15 |
| `run_20260525_133304_completed_qwen3_qwen3next_4a_default` | Q30/QNext | default | 4 | 0.0 | 250 | 13.4 | 82/292 | 28.1 | +14.7 | 43 |
| `run_20260402_235422_completed_qwen3_6a_default` | Q30 | default | 6 | 0.7 | 200 | 16.4 | 79/292 | 27.1 | +10.6 | 31 |
| `run_20260402_235456_completed_qwen3_6a_swe` | Q30 | S | 6 | 0.7 | 200 | 16.4 | 78/292 | 26.7 | +10.3 | 30 |

*`temp` = `llm.agent.temperature`; `step` = `agent.step_limit` (max agent turns/attempt). These differ across rows (e.g. Q30-4a is `0.0/200`, while GLM-4a and QNext-4a are `0.7/200`), so cross-row backbone/attempts comparisons are not controlled for temperature or step limit.*

---

### Table R1.2 — Verified, 4-attempt ablation (pass@4)
**Goal:** Repeat the 4-attempt default/SWE/no-SB comparison on the harder Verified split to test whether within-instance learning and the skillbook ablation generalize off Lite.

| Run | Backbone | Learn | iter0 (pass@1) % | Resolved n/N | pass@4 % | Δ vs iter0 (pp) | sb-assisted |
|---|---|---|---|---|---|---|---|
| `run_20260521_154504_completed_qwen3_4a_no_skillbook_verified` | Q30 | no-SB | 19.1 | 126/487 | 25.9 | +6.8 | 33 |
| `run_20260520_123809_completed_qwen3_4a_swe_verified` | Q30 | S | 19.1 | 138/487 | 28.3 | +9.2 | 45 |
| `run_20260520_144216_completed_qwen3_4a_default_verified` | Q30 | default | 19.1 | 160/487 | 32.9 | +13.8 | 67 |
| `run_20260524_160825_completed_qwen3_qwen3next_4a_default_verified` | Q30/QNext | default | 19.1 | 160/487 | 32.9 | +13.8 | 67 |

*(All four Verified runs share `temp=0.0`, `step=250`, so those columns are omitted here; they are shown in R1.1, where they vary.)*

---

### Significance tests — R1.1 + R1.2

Per-instance pairing over each run's `results/<inst>/iter_*.json`: `iter0` = first-attempt resolved (pass@1); `resolved_any` = resolved in any attempt (pass@N). Paired **t-test** + **10 000-resample instance bootstrap** on the per-instance difference, reusing the helpers in `scripts/q1_stat_tests.py`; Benjamini–Hochberg FDR per family. McNemar is omitted from (A) — it is degenerate there (see below) — and tracked (not shown) in (B), where it is k-symmetric (4-vs-4) and agrees with the t-test.

> **Inference caveat.** The sample unit is the **instance** (n = 292 Lite / 487 Verified) and each cell is a **single run** (n_runs = 1). These tests answer *"is the per-instance lift nonzero over this benchmark population"* — they capture instance-sampling uncertainty only, **not** run-to-run / attempt-sampling variance. Establishing *"config A beats config B in general"* needs the multi-run 5v5 design of R2–R5. Read the across-run (B) p-values as evidence about these specific runs, and lean on the within-run (A) tests for the headline.

**(A) Within-run test-time scaling — iter0 (pass@1) vs pass@N, paired by instance.**
`diff = resolved_any − iter0 ∈ {0,1}`, monotone by construction (pass@N ⊇ pass@1) ⇒ paired t-test / bootstrap are the right tools; **McNemar is degenerate** (the iter0-only cell is always 0). 1-attempt rows have no scaling to test and are excluded.

| Backbone / run | N | iter0 % | pass@N % | Δ pp | t p_raw | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| GLM 4a (swe) | 292 | 34.2 | 34.9 | +0.7 | 0.158 | [+0.0, +1.7] |
| GLM 4a (default) | 292 | 37.0 | 37.7 | +0.7 | 0.158 | [+0.0, +1.7] |
| QNext 4a (default) | 292 | 38.0 | 40.1 | +2.1 | 0.014 | [+0.7, +3.8] |
| QNext 4a (swe) | 292 | 37.7 | 39.7 | +2.1 | 0.014 | [+0.7, +3.8] |
| Q30 4a (no-SB) | 292 | 14.4 | 26.0 | +11.6 | <0.0001 | [+8.2, +15.4] |
| Q30 4a (swe) | 292 | 14.0 | 20.2 | +6.2 | <0.0001 | [+3.4, +9.2] |
| Q30 4a (default) | 292 | 13.4 | 18.5 | +5.1 | <0.0001 | [+2.7, +7.9] |
| Q30/QNext 4a (default) | 292 | 13.4 | 28.1 | +14.7 | <0.0001 | [+11.0, +18.8] |
| Q30 6a (default) | 292 | 16.4 | 27.1 | +10.6 | <0.0001 | [+7.2, +14.4] |
| Q30 6a (swe) | 292 | 16.4 | 26.7 | +10.3 | <0.0001 | [+6.8, +14.0] |
| Q30 4a Verified (no-SB) | 487 | 19.1 | 25.9 | +6.8 | <0.0001 | [+4.5, +9.0] |
| Q30 4a Verified (swe) | 487 | 19.1 | 28.3 | +9.2 | <0.0001 | [+6.8, +11.9] |
| Q30 4a Verified (default) | 487 | 19.1 | 32.9 | +13.8 | <0.0001 | [+10.9, +16.8] |
| Q30/QNext 4a Verified (default) | 487 | 19.1 | 32.9 | +13.8 | <0.0001 | [+10.7, +16.8] |

*Conclusion (A).* Retry-driven scaling is real and large for **Q30** and on **Verified** — every Q30/Verified lift is p<0.0001 with CIs well clear of 0, and the Q30/QNext-mixed rows gain the most (+14–15 pp). **GLM-4a is the exception: +0.7 pp is not significant (p=0.16, CI touches 0)** — GLM barely benefits from retries. QNext-4a is borderline (+2.1 pp, p=0.014). After BH-FDR over the 14-row family, all Q30/Verified rows survive; the two GLM rows do not.

**(B) Across-run ablation at pass@N — paired by shared instance id.** Direction is not forced, so all paired tests apply; instance sets coincide (Lite 292, Verified 487) for every pair shown.

*(B1) SWE-learn vs default-learn — same backbone / attempts / temp / step:*

| Pair | N | default % | swe % | Δ (def−swe) pp | t p_raw | p_FDR | bootstrap 95% CI |
|---|---|---|---|---|---|---|---|
| GLM 4a | 292 | 37.7 | 34.9 | +2.7 | 0.218 | 0.515 | [−1.7, +7.2] |
| QNext 4a | 292 | 40.1 | 39.7 | +0.3 | 0.862 | 0.862 | [−3.4, +4.1] |
| Q30 4a (Lite) | 292 | 18.5 | 20.2 | −1.7 | 0.412 | 0.515 | [−5.8, +2.4] |
| Q30 6a (Lite) | 292 | 27.1 | 26.7 | +0.3 | 0.318 | 0.515 | [+0.0, +1.0] |
| Q30 4a (Verified) | 487 | 32.9 | 28.3 | +4.5 | 0.003 | 0.013 | [+1.6, +7.4] |

*(B2) skillbook vs no-skillbook at pass@N:*

| Pair | N | skillbook % | no-SB % | Δ pp | t p_raw | p_FDR | bootstrap 95% CI |
|---|---|---|---|---|---|---|---|
| Verified default vs no-SB | 487 | 32.9 | 25.9 | +7.0 | <0.0001 | <0.0001 | [+3.9, +10.1] |
| Verified SWE vs no-SB | 487 | 28.3 | 25.9 | +2.5 | 0.090 | 0.090 | [−0.4, +5.3] |
| Lite default vs no-SB | 292 | 18.5 | 26.0 | −7.5 | 0.002 | 0.004 | [−12.3, −3.1] |
| Lite SWE vs no-SB | 292 | 20.2 | 26.0 | −5.8 | 0.006 | 0.008 | [−9.9, −1.7] |

*Conclusion (B).*
- **Learn mode (SWE vs default):** no significant difference on Lite across four backbone/attempt settings (all p_FDR ≥ 0.5); **only on Verified does default beat SWE (+4.5 pp, p_FDR = 0.013)**. The SWE-optimized reflector does not reliably help and is a net negative once off Lite.
- **Skillbook vs no-SB:** the comparison **reverses sign between benchmarks**. On Verified the default skillbook **significantly helps (+7.0 pp, p_FDR < 0.0001)** while SWE only trends above no-SB (+2.5 pp, ns); on Lite the sign flips and **no-SB resolves significantly more** than either skillbook condition (default −7.5 pp, SWE −5.8 pp, both FDR-significant).

---

### Table R1.3 — Aggregated no-skillbook test-time scaling (split025 val, n=113)
**Goal:** Quantify pure test-time scaling with no skillbook — empty-skillbook resolution scaled to 60 attempts per instance (12 runs × 5) — which doubles as the stable, low-variance Δ reference baseline for R2–R5.
*QNext global skillbook aggregate (`val_skillbook_aggregated_split025_vpk5_qwen3next_global_default`, n=4×5) is the skillbook counterpart, kept as a commented reference.*

| Run | pass@1 (avg) % | pass@5 % | resolved-any (pass@60) % |
|---|---|---|---|
| `val_baseline_aggregated_split025_vpk5_qwen3` | 20.4 | 38.7 | 50.4 |

> **Significance tests** for the split025 val skillbook-vs-baseline comparisons (valSB vs the aggregated valBL60 / each run's own valBL5, paired over the 113 val instances) are in a separate file: **[`7_val_repo_global_stat_tests.md`](7_val_repo_global_stat_tests.md)**.

<!-- QNext global skillbook aggregate (commented reference, n=4×5=20 att/inst): `val_skillbook_aggregated_split025_vpk5_qwen3next_global_default` → pass@1 61.5 / pass@5 73.4 / avg 61.5 / resolved-any 76.1 -->
