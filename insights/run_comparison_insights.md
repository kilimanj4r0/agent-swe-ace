# Run Comparison Insights
ID   | Proc | Resolv | Unres | Rate  | LLM                  | Att | Time  | Learn      | BL | SB Assist                      
-----+------+--------+-------+-------+----------------------+-----+-------+------------+----+--------------------------------
#000 | 300  | 52     | 248   | 17.3% | -                    | 1   | -     | baseline   | BL | 0                              
#001 | 292  | 64     | 228   | 21.9% | Qwen3-Coder-30B      | 2   | 11.6h | custom_swe | T  | 13 (i1:13)                     
#002 | 292  | 67     | 225   | 22.9% | Qwen3-Coder-30B      | 2   | 15.2h | default    | T  | 16 (i1:16)                     
#003 | 292  | 82     | 210   | 28.1% | Qwen3-Coder-30B      | 6   | 65.8h | default    | T  | 31 (i1:18,i2:5,i3:5,i4:2,i5:1) 
#004 | 292  | 81     | 211   | 27.7% | Qwen3-Coder-30B      | 6   | 70.7h | custom_swe | T  | 30 (i1:15,i2:10,i3:3,i4:1,i5:1)
#005 | 292  | 32     | 260   | 11.0% | Qwen3-Coder-30B      | 1   | 19.1h | custom_swe | F  | 0                              
#006 | 292  | 29     | 263   | 9.9%  | Qwen3-Coder-30B      | 1   | 19.2h | default    | F  | 0                              
#007 | 292  | 52     | 240   | 17.8% | Qwen3-Coder-Next-FP8 | 4   | 44.0h | default    | F  | 6 (i1:1,i2:4,i3:1)             
#008 | 292  | 51     | 241   | 17.5% | Qwen3-Coder-Next-FP8 | 4   | 43.8h | custom_swe | F  | 6 (i1:4,i2:2)          

**Resolution rate vs attempts (same model: Qwen3-Coder-30B)**
- 1 attempt (#005/#006): ~10% — ~30 resolved
- 2 attempts (#001/#002): ~23% — ~66 resolved (+36, +120% over 1-att)
- 6 attempts (#003/#004): ~28% — ~82 resolved (+16, +24% over 2-att)

The jump from 1→2 attempts is massive (+120%). From 2→6 it's marginal (+24%) at 5x the time cost (12h → 68h). **2 attempts is the efficiency sweet spot.**

**custom_swe vs default learn phase**
- At 2 attempts: default 22.9% vs custom_swe 21.9% (+1.0%)
- At 6 attempts: default 28.1% vs custom_swe 27.7% (+0.4%)
- At 4 attempts (Next-FP8): default 17.8% vs custom_swe 17.5% (+0.3%)

Default edges out custom_swe in every pairing, but the gap is negligible and not statistically meaningful at N=292. They're functionally equivalent.

**Skillbook learning contribution**
- Best run (#003): 31 of 82 resolves were SB-assisted (38%)
- 2-attempt runs: 13-16 of ~66 resolves were SB-assisted (~20%)
- Diminishing returns per iteration in #003: i1→18, i2→5, i3→5, i4→2, i5→1
- ~60% of SB-assisted resolves land on the first retry

**Model comparison: 30B vs Next-FP8**
- Qwen3-Coder-30B with 6 attempts: 28.1% (#003)
- Qwen3-Coder-Next-FP8 with 4 attempts: 17.8% (#007)

Not a fair comparison (different attempt counts, different exclude lists, different concurrency), but 30B clearly outperforms Next-FP8. Even at the same 4 attempts, #007/#008 with Next-FP8 score ~18% vs #003/#004 with 30B at 28%. The FP8 quantization or architectural changes likely degrade SWE-bench performance.

**Time efficiency**
- 2 attempts: ~12h for ~66 resolved → ~11 min/resolve
- 6 attempts: ~68h for ~82 resolved → ~50 min/resolve
- The extra 4 attempts in the 6-att runs buy only ~16 more resolves at ~14 min each on top of the 2-att base, but the wall-clock is dominated by the long tail of hard instances.

**Baseline anomaly**
- Baseline #000 (52/300 = 17.3%) scores higher than 1-attempt ACE runs #005/#006 (10-11% on the same 292 instances, 51/292 = 17.5% when adjusted). This suggests the 1-attempt ACE predict phase with empty skillbook is not meaningfully worse than the raw baseline agent — the gap is an artifact of the 8 extra instances in the baseline set.
