# Per-Repo Matplotlib Run Comparison

**Qwen3-Coder-Next-FP8** (`run_20260416_131855`) vs **GLM-4.7-Flash** (`run_20260416_180743`)

| Metric | Qwen3-Coder-Next-FP8 | GLM-4.7-Flash |
|---|---|---|
| Train instances | 19 | 19 |
| Train resolved | 1 (5.3%) | 1 (5.3%) |
| Skills learned | 41 | 49 |
| Val baseline (0 skills) | 0/4 (0%) | 0/4 (0%) |
| Val skillbook | 0/4 (0%) | 0/4 (0%) |
| Skillbook improvement | +0.0% | +0.0% |
| Duration | ~1.1h | ~4.7h |
| Baseline reused | 19/19 | 19/19 |

## Insights

- **Identical resolution**: Both models solve only `matplotlib-23964` — the one "easy" matplotlib instance. Every other instance (18 train + 4 val) is unresolved. The models are equivalent on this repo.

- **Skillbook transfer failure**: Neither run shows any skillbook benefit. Val baseline and val skillbook both score 0/4 — the learned skills (41 and 49 respectively) provide zero transfer to unseen instances.

- **GLM learns more skills but no payoff**: GLM produces 49 skills vs Qwen's 41 (+20%), yet this doesn't translate to any improvement. Skill count alone isn't predictive of value.

- **GLM is ~4x slower**: 4.7h vs 1.1h for the same work (8 actual instances after baseline reuse). At ~35 min/instance for GLM vs ~8 min/instance for Qwen, the cost-efficiency gap is large — especially with zero quality difference.

- **Matplotlib is extremely hard for both models**: 1/23 overall (4.3%) vs ~18% on the full benchmark. Matplotlib issues appear to require domain-specific knowledge (plotting, rendering pipelines) that neither model possesses at this scale.

- **Per-repo skillbook with small N is ineffective**: 19 training instances yield 40-50 skills, but the skill diversity and quality aren't sufficient to help even on 4 validation instances from the same repo. The signal-to-noise ratio in learned skills is likely very low when the base resolution rate is only 5%.
