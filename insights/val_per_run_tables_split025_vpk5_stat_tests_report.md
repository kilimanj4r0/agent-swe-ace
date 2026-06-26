# Q1 — Does the skillbook beat the empty baseline? (per run)

Paired over the shared 113 val instances. diff = valSB − valBL (positive ⇒ skillbook helps). Two-sided tests; Benjamini–Hochberg FDR applied per test family (12 runs). Bootstrap: 10000 resamples, instance-level, seed=42. α = 0.05.

## 1. Paired Wilcoxon signed-rank (non-parametric)
Tests whether per-instance rate differences are systematically off zero. Zero-difference instances dropped (n_pairs reported). r = matched-pairs rank-biserial effect size in [−1, +1].

| run | n_pairs | W | p_raw | p_FDR | r |
|---|---:|---:|---:|---:|---:|
| repos_split025_swe | 57 | 751.5 | 0.5539 | 0.7879 | +0.091 |
| repos_split025_default | 56 | 743.0 | 0.6565 | 0.7879 | -0.069 |
| global_split025_swe | 57 | 536.0 | 0.0212 | 0.2544 | -0.351 |
| global_split025_default | 56 | 555.5 | 0.0484 | 0.2901 | -0.304 |
| global_split025_retk5_defalut | 56 | 789.0 | 0.9447 | 0.9447 | -0.011 |
| global_split025_retk5_swe | 56 | 693.0 | 0.3939 | 0.7878 | +0.132 |
| global_split025_retk20_defalut | 56 | 664.0 | 0.2761 | 0.7765 | +0.168 |
| global_split025_retk20_swe | 56 | 728.5 | 0.5735 | 0.7879 | +0.087 |
| repos_split025_retk5_default | 56 | 774.5 | 0.8512 | 0.9285 | -0.029 |
| repos_split025_retk5_swe | 56 | 733.5 | 0.6016 | 0.7879 | +0.081 |
| repos_split025_retk20_default | 56 | 661.0 | 0.2654 | 0.7765 | +0.172 |
| repos_split025_retk20_swe | 56 | 676.5 | 0.3236 | 0.7765 | +0.152 |

## 2. Paired t-test (parametric)
Tests the mean per-instance rate difference; reports 95% CI and Cohen's dz (signed, paired = mean_diff / sd_diff).

| run | mean_diff | 95% CI | t | p_raw | p_FDR | Cohen's dz |
|---|---:|---:|---:|---:|---:|---:|
| repos_split025_swe | +0.0277 | [-0.0295, +0.0850] | +0.96 | 0.3390 | 0.4737 | +0.090 |
| repos_split025_default | -0.0024 | [-0.0476, +0.0429] | -0.10 | 0.9180 | 0.9180 | -0.010 |
| global_split025_swe | -0.0590 | [-0.1149, -0.0031] | -2.09 | 0.0389 | 0.4666 | -0.197 |
| global_split025_default | -0.0431 | [-0.0966, +0.0105] | -1.59 | 0.1138 | 0.4737 | -0.150 |
| global_split025_retk5_defalut | +0.0047 | [-0.0455, +0.0549] | +0.19 | 0.8524 | 0.9180 | +0.018 |
| global_split025_retk5_swe | +0.0242 | [-0.0273, +0.0757] | +0.93 | 0.3540 | 0.4737 | +0.088 |
| global_split025_retk20_defalut | +0.0348 | [-0.0113, +0.0810] | +1.49 | 0.1378 | 0.4737 | +0.141 |
| global_split025_retk20_swe | +0.0242 | [-0.0244, +0.0728] | +0.99 | 0.3264 | 0.4737 | +0.093 |
| repos_split025_retk5_default | +0.0030 | [-0.0514, +0.0573] | +0.11 | 0.9145 | 0.9180 | +0.010 |
| repos_split025_retk5_swe | +0.0260 | [-0.0295, +0.0814] | +0.93 | 0.3553 | 0.4737 | +0.087 |
| repos_split025_retk20_default | +0.0384 | [-0.0161, +0.0928] | +1.40 | 0.1653 | 0.4737 | +0.131 |
| repos_split025_retk20_swe | +0.0277 | [-0.0242, +0.0797] | +1.06 | 0.2927 | 0.4737 | +0.099 |

## 3. McNemar (resolved-at-least-once: rate > 0)
Pairs each instance as solved-yes/no under each condition. `BL_only` = baseline solved it but skillbook didn't; `SB_only` = the reverse. Exact two-sided sign test on discordants.

| run | both | BL_only | SB_only | neither | p_raw | p_FDR |
|---|---:|---:|---:|---:|---:|---:|
| repos_split025_swe | 30 | 27 | 0 | 56 | <0.0001 | <0.0001 |
| repos_split025_default | 27 | 30 | 0 | 56 | <0.0001 | <0.0001 |
| global_split025_swe | 26 | 31 | 0 | 56 | <0.0001 | <0.0001 |
| global_split025_default | 25 | 32 | 0 | 56 | <0.0001 | <0.0001 |
| global_split025_retk5_defalut | 31 | 26 | 0 | 56 | <0.0001 | <0.0001 |
| global_split025_retk5_swe | 30 | 27 | 0 | 56 | <0.0001 | <0.0001 |
| global_split025_retk20_defalut | 32 | 25 | 0 | 56 | <0.0001 | <0.0001 |
| global_split025_retk20_swe | 29 | 28 | 0 | 56 | <0.0001 | <0.0001 |
| repos_split025_retk5_default | 29 | 28 | 0 | 56 | <0.0001 | <0.0001 |
| repos_split025_retk5_swe | 29 | 28 | 0 | 56 | <0.0001 | <0.0001 |
| repos_split025_retk20_default | 31 | 26 | 0 | 56 | <0.0001 | <0.0001 |
| repos_split025_retk20_swe | 30 | 27 | 0 | 56 | <0.0001 | <0.0001 |

## 4. Bootstrap 95% CI on the mean difference
Resamples the 113 instances; CI excluding 0 ≈ significant at 0.05.

| run | mean_diff | 95% CI | excludes 0? |
|---|---:|---:|:--:|
| repos_split025_swe | +0.0277 | [-0.0276, +0.0841] | no |
| repos_split025_default | -0.0024 | [-0.0475, +0.0420] | no |
| global_split025_swe | -0.0590 | [-0.1146, -0.0041] | yes |
| global_split025_default | -0.0431 | [-0.0954, +0.0108] | no |
| global_split025_retk5_defalut | +0.0047 | [-0.0435, +0.0547] | no |
| global_split025_retk5_swe | +0.0242 | [-0.0251, +0.0757] | no |
| global_split025_retk20_defalut | +0.0348 | [-0.0102, +0.0807] | no |
| global_split025_retk20_swe | +0.0242 | [-0.0223, +0.0730] | no |
| repos_split025_retk5_default | +0.0030 | [-0.0498, +0.0566] | no |
| repos_split025_retk5_swe | +0.0260 | [-0.0280, +0.0822] | no |
| repos_split025_retk20_default | +0.0384 | [-0.0152, +0.0929] | no |
| repos_split025_retk20_swe | +0.0277 | [-0.0229, +0.0783] | no |

## At a glance
FDR-corrected significance (`*` < 0.05). Wilcoxon and t-test are the trustworthy ones here.

| run | mean_diff | Wilcoxon | t-test | McNemar | boot CI excl 0 |
|---|---:|:--:|:--:|:--:|:--:|
| repos_split025_swe | +0.0277 |  |  | * | no |
| repos_split025_default | -0.0024 |  |  | * | no |
| global_split025_swe | -0.0590 |  |  | * | yes |
| global_split025_default | -0.0431 |  |  | * | no |
| global_split025_retk5_defalut | +0.0047 |  |  | * | no |
| global_split025_retk5_swe | +0.0242 |  |  | * | no |
| global_split025_retk20_defalut | +0.0348 |  |  | * | no |
| global_split025_retk20_swe | +0.0242 |  |  | * | no |
| repos_split025_retk5_default | +0.0030 |  |  | * | no |
| repos_split025_retk5_swe | +0.0260 |  |  | * | no |
| repos_split025_retk20_default | +0.0384 |  |  | * | no |
| repos_split025_retk20_swe | +0.0277 |  |  | * | no |

## Caveats
- **valBL is the same 60-attempt baseline in every run** — it's the shared control, so each run gets an identical reference.
- **valSB has only 5 attempts** → per-instance rates are coarse (multiples of 0.2), producing many ties/zeros in the Wilcoxon (handled, n_pairs reported) and a coarse signal for the t-test.
- **McNemar is k-asymmetric**: baseline's `resolved_any` draws on 60 attempts vs the skillbook's 5, so `BL_only` is structurally inflated. Read it as 'which instances flip solvability', not as a fair skill-vs-baseline verdict — prefer Wilcoxon/t-test/bootstrap for that.
- Direction matters: a *negative* mean_diff / negative effect means the skillbook **hurt** that run.
- FDR is corrected *within* each test family (12 runs); cross-family comparisons are not additionally corrected.

