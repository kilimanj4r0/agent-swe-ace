#!/usr/bin/env python3
"""Generate visualizations.ipynb for the aggregated val-baseline folder.

Builds a Jupyter notebook (via nbformat) that reads stats/*.json from the
aggregated data dir and renders the full pass@k analysis: overall curve,
marginal gains, success histogram, per-repo curves/heatmap/bars, per-instance
sorted bar, the per-instance × per-attempt matrix, and test-time-compute scaling.

The notebook is max-k-agnostic: every figure derives `max_k` (= runs × 5 attempts
per instance) from the data, so it works for 40, 60, or any aggregated depth.

Usage:
    uv run python scripts/build_viz_notebook.py
    uv run python scripts/build_viz_notebook.py \
        --data-dir data/val_baseline_aggregated_split025_vpk5
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

# Each cell is (type, source). Markdown cells describe; code cells plot.
CELLS: list[tuple[str, str]] = [
    ("md", """\
# Val-Baseline pass@k Analysis — split025 (val_pass_k=5, aggregated)

This notebook visualizes the **val baseline** results aggregated across all
`completed*split025*vpk5` runs. Every run shares the **identical 113 val
instances**, and each instance receives **multiple independent fresh attempts**
(N runs × 5) under a uniform val-baseline condition (empty skillbook, no learning,
Qwen3-Coder-30B). Aggregating across runs yields N×5 independent attempts per
instance, enabling a pass@k curve up to k = N×5.

**Methodology**
- `avg` = resolved attempts / total attempts (== pass@1).
- `pass@k` = `1 − C(n−c, k)/C(n, k)` (standard combinatorial / HumanEval estimator),
  n = attempts per instance, c = successes, macro-averaged over the 113 instances.
- `pass@max` = fraction of instances solved at least once across all attempts.

Source data: `stats/{overall,per_repo,per_instance}.json` in the folder below.
"""),

    ("code", """\
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

%matplotlib inline

# Resolve the aggregated data dir (this notebook lives inside it).
_candidates = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/root/makharev/agent-swe-ace/data/val_baseline_aggregated_split025_vpk5"),
]
DATA_DIR = next((c for c in _candidates if (c / "stats" / "overall.json").exists()),
                _candidates[-1])
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

overall = json.loads((DATA_DIR / "stats" / "overall.json").read_text())
per_repo = json.loads((DATA_DIR / "stats" / "per_repo.json").read_text())
per_instance = json.loads((DATA_DIR / "stats" / "per_instance.json").read_text())
runs_index = json.loads((DATA_DIR / "runs_index.json").read_text())

ks = np.array(sorted(int(k) for k in overall["pass_at_k"]))
pak = np.array([overall["pass_at_k"][str(k)] for k in ks])
max_k = int(ks[-1])                       # N runs × 5 attempts per instance
ann_ks = [k for k in (1, 5, 10, 20, max_k) if k <= max_k]   # k's to annotate on curves
REPOS = sorted(per_repo)
REPO_COLOR = {r: plt.get_cmap("tab10")(i % 10) for i, r in enumerate(REPOS)}

mpl.rcParams.update({"figure.dpi": 110, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print("saved", path.relative_to(DATA_DIR))


print(f"DATA_DIR = {DATA_DIR}")
print(f"instances={overall['n_instances']}  runs={overall['n_runs']}  "
      f"attempts/instance={max_k}  repos={len(REPOS)}")
"""),

    ("md", """\
## Headline metrics
"""),
    ("code", """\
kpis = {
    "total attempts": overall["total_attempts"],
    "resolved attempts": overall["total_resolved_attempts"],
    "avg (=pass@1)": overall["avg"],
    "pass@5": overall["pass_at_k"]["5"],
    "pass@10": overall["pass_at_k"]["10"],
    f"pass@{max_k} (any)": overall["resolved_any_rate"],
    f"instances solved >=1/{max_k}": overall["resolved_any"],
    "instances NEVER solved": overall["n_instances"] - overall["resolved_any"],
}
for k, v in kpis.items():
    if isinstance(v, float):
        print(f"  {k:<26} {v*100:6.2f}%")
    else:
        print(f"  {k:<26} {v}")
"""),

    ("md", """\
## 1. Overall pass@k curve

Monotonic but sharply concave: most of the gain is in the first ~10 attempts.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ks, pak * 100, "-o", ms=3, lw=2, color="#1f77b4", label="pass@k")
for k in ann_ks:
    ax.annotate(f"p@{k}={pak[k-1]*100:.1f}%", (k, pak[k-1]*100),
                textcoords="offset points", xytext=(6, -14), fontsize=9)
# Average resolved rate over all attempts (== p@1 == total_solves/total_attempts)
avg = overall["avg"] * 100
ax.axhline(avg, color="#d62728", ls="--", lw=1.5,
           label=f"avg resolved rate = {avg:.1f}%")
ax.set_xlabel("k  (number of attempts)")
ax.set_ylabel("pass@k  (% instances solved)")
ax.set_xlim(0, max_k + 1)
ax.set_title(f'Overall pass@k  (n={overall["n_instances"]} instances, {max_k} attempts each)')
ax.legend(loc="lower right")
savefig(fig, "01_overall_pass_at_k.png")
"""),

    ("md", """\
## 2. Marginal gain per attempt (diminishing returns)

Each additional attempt adds less than the last. The early gains (pass@1→pass@5)
far outweigh the late ones (pass@10→pass@max).
"""),
    ("code", """\
gain = np.diff(pak) * 100  # percentage-point gain per extra attempt
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(ks[1:], gain, color="#ff7f0e")
ax.set_xlabel("attempt number k")
ax.set_ylabel("Δ pass@k  (pp)")
ax.set_title("Marginal pass@k gain per additional attempt")
savefig(fig, "02_marginal_gain.png")
"""),

    ("md", """\
## 3. Per-instance success distribution

Strikingly bimodal: a large block of instances is **never solved in any attempt**
(red, c=0), while the rest spread across the range. This hard floor is what caps
the pass@k ceiling well below 100%.
"""),
    ("code", """\
hist = overall["success_histogram"]
xs = np.arange(0, max_k + 1)
counts = np.array([hist.get(str(x), 0) for x in xs])
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = ["#d62728" if x == 0 else "#2ca02c" for x in xs]
ax.bar(xs, counts, color=colors)
ax.annotate(f"{counts[0]} instances\\nnever solved (c=0)", (0, counts[0]),
            textcoords="offset points", xytext=(12, -8), fontsize=10, color="#d62728")
ax.set_xlabel(f"successful attempts out of {max_k}  (c)")
ax.set_ylabel("number of instances")
ax.set_title("Per-instance success-count distribution")
ax.set_xticks(range(0, max_k + 1, 5))
savefig(fig, "03_success_histogram.png")
"""),

    ("md", """\
## 4. pass@k by repository

Wide variance: scikit-learn / xarray / sympy climb high; matplotlib / sphinx / django
stay low (dominated by never-solvable instances).
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(10.5, 5.5))
for r in REPOS:
    rpak = np.array([per_repo[r]["pass_at_k"][str(k)] for k in ks])
    ax.plot(ks, rpak * 100, "-o", ms=2.5, lw=1.8,
            label=f"{r} ({per_repo[r]['n_instances']})", color=REPO_COLOR[r])
    n = per_repo[r]["n_instances"]
    c1 = round(per_repo[r]["pass_at_k"]["1"] * n)             # resolved count at p@1
    cmax = round(per_repo[r]["pass_at_k"][str(max_k)] * n)    # resolved count at p@max
    ax.annotate(f"{c1}", (ks[0], rpak[0] * 100),
                xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7, color=REPO_COLOR[r])
    ax.annotate(f"{cmax}", (ks[-1], rpak[-1] * 100),
                xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                fontsize=7, color=REPO_COLOR[r])
ax.plot(ks, pak * 100, "--", color="black", lw=2, alpha=0.7,
        label=f"OVERALL ({overall['n_instances']})")
ax.set_xlabel("k")
ax.set_ylabel("pass@k (%)")
ax.set_title(f"pass@k by repository  (legend: n issues; left # = p@1, right # = p@{max_k} resolved)")
ax.legend(fontsize=8, ncol=2, loc="upper left")
ax.set_xlim(-3, max_k + 5)
savefig(fig, "04_per_repo_pass_at_k.png")
"""),

    ("md", """\
## 5. Per-repo pass@k heatmap

Same data as a heatmap — saturating rows (yellow) = repos where extra attempts
keep helping; flat-low rows = repos stuck on never-solvable instances.
"""),
    ("code", """\
mat = np.array([[per_repo[r]["pass_at_k"][str(k)] * 100 for k in ks] for r in REPOS])
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(mat, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(REPOS))); ax.set_yticklabels(REPOS)
ticks = list(range(5, max_k + 1, 5))
ax.set_xticks([k - 1 for k in ticks]); ax.set_xticklabels(ticks)
ax.set_xlabel("k")
ax.set_title("pass@k heatmap by repository (%)")
fig.colorbar(im, ax=ax, label="pass@k (%)")
ax.grid(False)
savefig(fig, "05_per_repo_heatmap.png")
"""),

    ("md", f"""\
## 6. Per-repo bars (avg / p@5 / p@10 / p@max)

Repos sorted by pass@max (solvable ceiling). The gap between avg and pass@max
measures how much sampling variance hides — large for sympy/sklearn, tiny for
matplotlib/sphinx.
"""),
    ("code", """\
pmax_key = str(max_k)
order = sorted(REPOS, key=lambda r: per_repo[r]["pass_at_k"][pmax_key])
rows = []
for r in order:
    rows.append({
        "repo": r,
        "n_inst": per_repo[r]["n_instances"],
        "avg": per_repo[r]["avg"] * 100,
        "p@5": per_repo[r]["pass_at_k"]["5"] * 100,
        "p@10": per_repo[r]["pass_at_k"]["10"] * 100,
        f"p@{max_k}": per_repo[r]["pass_at_k"][pmax_key] * 100,
    })
df_repo = pd.DataFrame(rows)
cols = ["avg", "p@5", "p@10", f"p@{max_k}"]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(df_repo)); w = 0.2
for i, col in enumerate(cols):
    ax.bar(x + (i - 1.5) * w, df_repo[col], w, label=col)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\\n(n={n})" for r, n in zip(df_repo.repo, df_repo.n_inst)], fontsize=8)
ax.set_ylabel("% resolved")
ax.set_title(f"Per-repo resolution: {' / '.join(cols)}  (sorted by p@{max_k})")
ax.legend()
savefig(fig, "06_per_repo_bars.png")
df_repo.round(1)
"""),

    ("md", """\
## 7. Per-instance success rate (sorted, colored by repo)

Each bar is one instance's success rate over all attempts. Sorted ascending.
The flat-zero run on the left is the never-solved set — the target surface for
skillbook learning.
"""),
    ("code", """\
inst_items = sorted(per_instance.items(), key=lambda kv: kv[1]["successes"])
names = [k for k, _ in inst_items]
rates = [v["successes"] / max_k * 100 for _, v in inst_items]
rcolors = [REPO_COLOR[per_instance[n]["repo"]] for n in names]

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(range(len(names)), rates, color=rcolors, edgecolor="none")
ax.axhline(overall["avg"] * 100, color="black", ls="--", lw=1,
           label=f"avg {overall['avg']*100:.1f}%")
ax.set_xlim(-1, len(names))
ax.set_xlabel("instance (sorted by success rate)")
ax.set_ylabel(f"success rate over {max_k} attempts (%)")
ax.set_title("Per-instance resolution — color = repository")
handles = [Patch(facecolor=REPO_COLOR[r], label=r) for r in REPOS]
ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left")
savefig(fig, "07_per_instance_sorted.png")
"""),

    ("md", """\
## 8. Attempt matrix — every attempt × every instance

The full picture: 113 instances (sorted by # successes) × all attempts
(`r00_iter0 … rNN_iter4`). Blue = solved. Horizontal lines separate repositories.

Read it as: the bottom band (all-white) is the **never-solvable** set; the top
band is the **reliably-solvable** set; the middle is genuinely stochastic.
"""),
    ("code", """\
RUN_TAGS = sorted(runs_index)
M = np.zeros((len(names), max_k), dtype=int)
for i, n in enumerate(names):
    flat = []
    for t in RUN_TAGS:
        flat.extend(per_instance[n]["per_run"].get(t, [False] * 5))
    M[i] = [int(bool(b)) for b in flat[:max_k]]

fig, ax = plt.subplots(figsize=(13, 9))
ax.imshow(M, aspect="auto", interpolation="nearest",
          cmap=ListedColormap(["#f2f2f2", "#1f77b4"]))
ax.set_xlabel(f"attempt index  (r00_iter0 … {RUN_TAGS[-1]}_iter4)")
ax.set_ylabel("instance  (sorted by # successes)")
ax.set_title("Attempt matrix — blue = solved")
prev = None
for i, n in enumerate(names):
    r = per_instance[n]["repo"]
    if prev is not None and r != prev:
        ax.axhline(i - 0.5, color="black", lw=0.6)
    prev = r
ax.set_yticks([])
ax.set_xticks(range(0, max_k, 5))
ax.set_xticklabels([f"#{i}" for i in range(0, max_k, 5)], fontsize=8)
ax.grid(False)
savefig(fig, "08_attempt_matrix.png")
"""),

    ("md", """\
## 9. Test-time compute scaling

Each attempt **is** test-time compute. The pass@k curve (sec. 1) is therefore a
**test-time-scaling curve**: spending more attempts ≈ more tokens → higher pass rate.
Here we quantify per-attempt compute (tokens / wall-time / steps) and re-express
pass@k against cumulative test-time compute.

This is **parallel-sampling (best-of-N)** scaling — independent fresh attempts with
*oracle* selection (pass@k = solved if *any* attempt is correct). That makes pass@k
the **ceiling** of best-of-N: a real selector would cost extra compute on top.
"""),
    ("code", """\
df = pd.read_csv(DATA_DIR / "analysis" / "per_attempt.csv")
df["resolved"] = df["resolved"].astype(str).eq("True")

print(f"per-attempt compute (n={len(df)} attempts):")
print(f"  total tokens : median={df['total_tokens'].median():,.0f}  mean={df['total_tokens'].mean():,.0f}  "
      f"(mean inflated by runaway attempts)")
print(f"  wall seconds : median={df['wall_seconds'].median():,.0f}s  mean={df['wall_seconds'].mean():,.0f}s")
print(f"  agent steps  : median={df['n_steps'].median():,.0f}  mean={df['n_steps'].mean():,.0f}")
print("\\nmedian total tokens by exit_status:")
print(df.groupby("exit_status")["total_tokens"].median().sort_values().apply(lambda x: f"{x:,.0f}").to_string())
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for es, color in [("Submitted", "#1f77b4"), ("LimitsExceeded", "#ff7f0e"),
                  ("ContextWindowExceeded", "#d62728")]:
    sub = df.loc[df["exit_status"] == es, "total_tokens"]
    if len(sub):
        ax.hist(sub, bins=np.logspace(3.5, 8, 60), alpha=0.6,
                label=f"{es} (n={len(sub)})", color=color)
ax.set_xscale("log")
ax.set_xlabel("total tokens per attempt (log)")
ax.set_ylabel("number of attempts")
ax.set_title("Per-attempt compute — heavy tail from LimitsExceeded / ContextWindowExceeded (never resolved)")
ax.legend()
savefig(fig, "09_compute_distribution.png")
"""),
    ("code", """\
mean_tok = df["total_tokens"].mean()
cum_tokens = ks * mean_tok  # expected cumulative compute over k independent attempts
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(cum_tokens / 1e6, pak * 100, "-o", ms=3, lw=2, color="#1f77b4")
for k in [k for k in (1, 2, 5, 10, 20, max_k) if k <= max_k]:
    ax.annotate(f"k={k}", (cum_tokens[k - 1] / 1e6, pak[k - 1] * 100),
                textcoords="offset points", xytext=(6, -11), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("cumulative test-time compute  (M tokens = k × mean per-attempt tokens)")
ax.set_ylabel("pass@k  (% instances solved)")
ax.set_title(f"Test-time scaling: pass rate vs cumulative compute\\n"
             f"(mean {mean_tok:,.0f} tok/attempt; median {df['total_tokens'].median():,.0f})")
savefig(fig, "10_test_time_scaling.png")
"""),

    ("code", """\
from IPython.display import display, Markdown
p1, p5, p10 = overall["avg"], overall["pass_at_k"]["5"], overall["pass_at_k"]["10"]
pmax = overall["resolved_any_rate"]
never = overall["n_instances"] - overall["resolved_any"]
med_tok = df["total_tokens"].median()
med_min = df["wall_seconds"].median() / 60
md = f'''## Takeaways (auto-computed from current data)

- **Sampling alone saturates well below 100%.** pass@1={p1*100:.1f}% → pass@10={p10*100:.1f}% → pass@{max_k}={pmax*100:.1f}% ({overall['resolved_any']}/{overall['n_instances']} solvable at least once). The curve is essentially flat after ~pass@10.
- **A hard floor of {never} instances is never solved in any of {max_k} attempts** — pure repeated sampling cannot reach them; this is the target surface for skillbook learning.
- **Per attempt:** median ~{med_tok:,.0f} tokens / ~{med_min:.1f} min; mean is inflated by runaway attempts (fig 9).
- **Repo variance is large** (figs 4/6): scikit-learn / xarray / sympy are highly stochastic; matplotlib / sphinx are dominated by never-solvable instances.
- Figures also saved as PNGs under `figures/`.
'''
display(Markdown(md))
"""),
]


def build(data_dir: Path, out_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for kind, src in CELLS:
        src = textwrap.dedent(src)
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    out_path.write_text(nbf.writes(nb))
    print(f"Wrote {out_path} ({len(nb.cells)} cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default_data = Path("data/val_baseline_aggregated_split025_vpk5")
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output notebook path (default: <data-dir>/visualizations.ipynb)")
    args = ap.parse_args()
    out = args.out or (args.data_dir / "visualizations.ipynb")
    build(args.data_dir.resolve(), out)


if __name__ == "__main__":
    main()
