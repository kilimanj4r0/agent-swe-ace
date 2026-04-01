# Token Usage Analysis for SWE-bench

## Context Window Requirements (SWE-bench Lite)

| Context Size | Coverage | Recommendation |
|--------------|----------|----------------|
| 16K | 64.1% | Too small - fails ~1/3 of instances |
| **32K** | **98.8%** | **Minimum viable** |
| **64K** | **100%** | **Recommended** |

## Key Metrics (337 instances analyzed)

### Aggregate Statistics
- **Total prompt tokens**: 118.2M
- **Total completion tokens**: 2.2M
- **Avg prompt tokens/instance**: 350,654
- **Avg completion tokens/instance**: 6,525
- **Avg API calls/instance**: 33.4

### Context Growth
- **Initial context**: ~1,500 tokens
- **Final context**: ~14,000 tokens
- **Growth factor**: 9.24x (median: 8.43x)

### By Exit Status

| Status | Count | Avg Prompt | Avg Max Context | Avg API Calls |
|--------|-------|------------|-----------------|---------------|
| Submitted | 293 | 274K | 13,310 | 29.2 |
| LimitsExceeded | 40 | 946K | 24,046 | 67.5 |
| RetryError | 4 | 2.8K | 1,953 | 1.5 |

### Max Context Distribution (All Instances)
- **P50**: 13,374 tokens
- **P75**: 18,779 tokens
- **P90**: 23,412 tokens
- **P95**: 27,097 tokens
- **P99**: 33,606 tokens
- **Max**: 50,307 tokens

### Max Context Distribution (Submitted Only)
- **P50**: 12,083 tokens
- **P95**: 23,666 tokens
- **P99**: 28,829 tokens
- **Max**: 37,265 tokens

## SWE-bench Verified vs Lite

**Note**: The above analysis is for SWE-bench **Lite** (300 instances).

**SWE-bench Verified** (500 instances) is a larger, more challenging subset. While no public token analysis exists for Verified specifically:
- Verified likely has similar or slightly higher context requirements
- The paper describes tasks requiring "extremely long contexts"
- Multi-file changes and complex reasoning increase context needs

## Recommendation

For SWE-bench issue solving:
- **Minimum**: 32K context window (covers 98.8% of successful runs)
- **Recommended**: 64K context window (100% coverage with headroom)
- Consider 128K for future-proofing and larger repositories

## Files
- Analysis script: `scripts/analyze_token_usage.py`
- Per-instance CSV: `token_usage_analysis.csv`
