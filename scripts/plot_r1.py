#!/usr/bin/env python3
"""R1 (within-instance / test-time scaling) figures.

Reads the single-phase per-instance skillbook runs cited in
``runs_results/2_R1_within_instance.md``, computes per-instance resolution and
skill-count metrics, and writes six figures to ``results_figures/R1/``.

Every figure is built through the ``@styled`` decorator, which applies the
unified look from :func:`apply_style` and owns the figure lifecycle (create,
save, close). **To change the look of all figures, edit :func:`apply_style`
and the palettes directly below it — nothing else.**

Usage::

    uv run python scripts/plot_r1.py
    uv run python scripts/plot_r1.py --out-dir data/results_figures/R1
    uv run python scripts/plot_r1.py --only skill_size_dist passk_scaling
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as pe

ROOT = Path(__file__).resolve().parents[1]

# Reuse the validated stat helpers so the CIs / p-values match the R1 report.
sys.path.insert(0, str(ROOT / "scripts"))
from q1_stat_tests import boot_ci, ttest_paired  # noqa: E402

OUT_DIR = ROOT / "results_figures" / "R1"
SEED = 20260630  # deterministic bootstrap


# ===========================================================================
# Unified style — the single place to edit the look of every figure.
# ===========================================================================

# Backbone palette (single source of truth — used by every plot).
BACKBONE_COLORS = {
    "Q30": "#1f77b4",       # blue
    "QNext": "#2ca02c",     # green
    "GLM": "#ff7f0e",       # orange
    "Q30/QNext": "#9467bd",  # purple
}
LEARN_ORDER = ["default", "swe", "no-sb"]
LEARN_DASH = {"default": "-", "swe": "--", "no-sb": ":"}
# Learn-mode palette for the headline figure (the default-vs-SWE comparison).
LEARN_COLORS = {"default": "#4C72B0", "swe": "#DD8452"}


def apply_style() -> None:
    """Apply the unified matplotlib/seaborn style.

    Idempotent — called before every figure via :func:`styled`. **Edit the
    project's plot look here and only here.**
    """
    sns.set_theme(style="whitegrid", context="talk", rc={
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def styled(filename: str, figsize: tuple[float, float] = (9, 5),
           tight_rect: list[float] | None = None):
    """Decorator every plot runs through.

    Creates the figure, applies :func:`apply_style`, hands a clean ``fig`` to
    the wrapped function, then tightens, saves and closes. Style is controlled
    centrally in :func:`apply_style`; the wrapped function only draws.

    ``tight_rect`` is forwarded to ``Figure.tight_layout(rect=...)`` to reserve
    figure margin (e.g. ``[0, 0.05, 1, 1]`` leaves 5% at the bottom for a
    ``fig.text`` caption that tight_layout would otherwise overlap).
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, out_dir: Path = OUT_DIR, dpi: int = 160, fmt: str = "png", **kwargs):
            apply_style()
            fig = plt.figure(figsize=figsize)
            func(fig, *args, **kwargs)
            fig.tight_layout(rect=tight_rect)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = filename.rsplit(".", 1)[0]
            path = out_dir / f"{stem}.{fmt}"
            save_kwargs = {"bbox_inches": "tight"}
            if fmt != "svg":                       # dpi is irrelevant for vector
                save_kwargs["dpi"] = dpi
            fig.savefig(path, format=fmt, **save_kwargs)
            plt.close(fig)
            print(f"saved {path.relative_to(ROOT)}")
            return path
        return wrapper
    return deco


# ===========================================================================
# R1 run registry (metadata from runs_results/2_R1_within_instance.md)
# ===========================================================================

@dataclass(frozen=True)
class Run:
    dir: str
    benchmark: str          # "Lite" | "Verified"
    backbone: str           # "Q30" | "QNext" | "GLM" | "Q30/QNext"
    learn: str              # "default" | "swe" | "no-sb"
    attempts: int           # max attempts configured
    has_skillbook: bool = True

    @property
    def path(self) -> Path:
        return ROOT / "data" / self.dir

    @property
    def label(self) -> str:
        return f"{self.backbone} · {self.learn} · {self.attempts}a"


R1_RUNS: list[Run] = [
    # ---- Lite (R1.1) ----
    Run("run_20260426_210831_completed_baseline", "Lite", "Q30", "no-sb", 1, False),
    Run("run_20260404_150133_completed_qwen3_1a_swe", "Lite", "Q30", "swe", 1),
    Run("run_20260404_150204_completed_qwen3_1a_default", "Lite", "Q30", "default", 1),
    Run("run_20260414_015144_completed_glm_4a_swe", "Lite", "GLM", "swe", 4),
    Run("run_20260414_015225_completed_glm_4a_default", "Lite", "GLM", "default", 4),
    Run("run_20260415_020217_completed_qwen3next_4a_default", "Lite", "QNext", "default", 4),
    Run("run_20260415_020540_completed_qwen3next_4a_swe", "Lite", "QNext", "swe", 4),
    Run("run_20260526_133345_completed_qwen3_4a_no_skillbook", "Lite", "Q30", "no-sb", 4, False),
    Run("run_20260426_211426_completed_qwen3_4a_swe", "Lite", "Q30", "swe", 4),
    Run("run_20260426_211500_completed_qwen3_4a_default", "Lite", "Q30", "default", 4),
    Run("run_20260525_133304_completed_qwen3_qwen3next_4a_default", "Lite", "Q30/QNext", "default", 4),
    Run("run_20260402_235422_completed_qwen3_6a_default", "Lite", "Q30", "default", 6),
    Run("run_20260402_235456_completed_qwen3_6a_swe", "Lite", "Q30", "swe", 6),
    # ---- Verified (R1.2) ----
    Run("run_20260521_154504_completed_qwen3_4a_no_skillbook_verified", "Verified", "Q30", "no-sb", 4, False),
    Run("run_20260520_123809_completed_qwen3_4a_swe_verified", "Verified", "Q30", "swe", 4),
    Run("run_20260520_144216_completed_qwen3_4a_default_verified", "Verified", "Q30", "default", 4),
    Run("run_20260524_160825_completed_qwen3_qwen3next_4a_default_verified", "Verified", "Q30/QNext", "default", 4),
]

# Figure-1 headline runs: the main per-instance-skillbook runs behind R1's
# significance analysis — (a) Lite: GLM + QNext (default & SWE);
# (b) Verified: Q30 (default & SWE).
HEADLINE_RUNS: list[Run] = [
    Run("run_20260414_015225_completed_glm_4a_default",            "Lite",     "GLM",   "default", 4),
    Run("run_20260414_015144_completed_glm_4a_swe",                "Lite",     "GLM",   "swe",     4),
    Run("run_20260415_020217_completed_qwen3next_4a_default",      "Lite",     "QNext", "default", 4),
    Run("run_20260415_020540_completed_qwen3next_4a_swe",          "Lite",     "QNext", "swe",     4),
    Run("run_20260520_144216_completed_qwen3_4a_default_verified", "Verified", "Q30",   "default", 4),
    Run("run_20260520_123809_completed_qwen3_4a_swe_verified",     "Verified", "Q30",   "swe",     4),
]


# ===========================================================================
# Data layer (pure functions — unit-tested in test_plot_r1.py)
# ===========================================================================

def _iter_key(path: Path) -> int:
    m = re.search(r"iter_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_benchmark_dir(run_dir: Path) -> Path | None:
    """Find the benchmark-scoped subdir (e.g. ``princeton-nlp__SWE-bench_Lite``);
    fall back to None (flat layout)."""
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and "__" in child.name:
            return child
    return None


def load_resolved_per_attempt(run_dir: Path) -> dict[str, list[bool]]:
    """instance_id -> [resolved_iter0, resolved_iter1, ...].

    One bool per ``results/<inst>/iter_N.json`` file (only attempts that were
    actually run exist; resolved instances stop early)."""
    bench = find_benchmark_dir(run_dir)
    res_root = (bench or run_dir) / "results"
    out: dict[str, list[bool]] = {}
    if not res_root.exists():
        return out
    for inst_dir in sorted(p for p in res_root.iterdir() if p.is_dir()):
        iters = []
        for f in sorted(inst_dir.glob("iter_*.json"), key=_iter_key):
            iters.append(bool(_load_json(f).get("resolved", False)))
        if iters:
            out[inst_dir.name] = iters
    return out


def load_skill_counts(run_dir: Path) -> dict[str, list[int]]:
    """instance_id -> [skill_count_iter1, ...] for instances with a skillbook
    (only instances that triggered Learn have files)."""
    bench = find_benchmark_dir(run_dir)
    sb_root = (bench or run_dir) / "skillbooks"
    out: dict[str, list[int]] = {}
    if not sb_root.exists():
        return out
    for inst_dir in sorted(p for p in sb_root.iterdir() if p.is_dir()):
        counts = []
        for f in sorted(inst_dir.glob("iter_*.json"), key=_iter_key):
            o = _load_json(f)
            counts.append(int(o.get("skill_count", len(o.get("skills", []) or []))))
        if counts:
            out[inst_dir.name] = counts
    return out


def resolved_within_k(attempts: list[bool], k: int) -> bool:
    """pass@k: resolved in any of the first ``k`` attempts (1-indexed)."""
    return any(attempts[:k])


def first_resolved_at(attempts: list[bool]) -> int | None:
    """1-indexed attempt at which the instance first resolves, else None."""
    for i, r in enumerate(attempts):
        if r:
            return i + 1
    return None


def passk_curve(run_resolved: dict[str, list[bool]], max_k: int) -> np.ndarray:
    """Fraction of instances resolved within k, for k = 1..max_k."""
    n = len(run_resolved)
    counts = np.zeros(max_k)
    for attempts in run_resolved.values():
        fr = first_resolved_at(attempts)
        if fr is not None and fr <= max_k:
            counts[fr - 1:] += 1
    return counts / n if n else counts


def per_instance_delta(run_resolved: dict[str, list[bool]], max_k: int):
    """Return (resolved_any, resolved_iter0) as int arrays.

    Their per-instance difference is the within-run test-time-scaling lift; its
    mean equals pass@N − pass@1. Used for the paired t-test and bootstrap CI.
    """
    any_ = np.array([int(any(a[:max_k])) for a in run_resolved.values()])
    i0 = np.array([int(a[0]) if a else 0 for a in run_resolved.values()])
    return any_, i0


def final_skill_sizes(skill_counts: dict[str, list[int]]) -> dict[str, int]:
    """instance_id -> skill_count at the last available iteration."""
    return {inst: cs[-1] for inst, cs in skill_counts.items()}


def deployed_skill_counts(skill_counts: dict[str, list[int]], attempts: int) -> dict[str, list[int]]:
    """Keep only skillbook iterations that were actually deployed (shown to a
    prediction call).

    In an ``attempts``-attempt run the book learned AFTER the final attempt —
    skillbook file ``iter_{attempts}`` — has no subsequent prediction to use it,
    so it is never deployed (verified: it has no matching trajectory file).
    Deployed books are therefore ``iter_1 .. iter_{attempts-1}`` — the first
    attempt uses the empty initial book. Dropping the trailing unused iteration
    aligns runs whose code persisted it (newer: QNext, Q30) with those that did
    not (older: GLM), so all runs are compared on the books the agent saw.
    """
    keep = max(attempts - 1, 0)
    return {inst: counts[:keep] for inst, counts in skill_counts.items()}


def first_resolved_buckets(run_resolved: dict[str, list[bool]]) -> dict[str, int]:
    """Count instances first resolved at attempt 1, 2, 3, '4+' (collapsed),
    plus 'never' (resolved in no attempt)."""
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0, "never": 0}
    for attempts in run_resolved.values():
        fr = first_resolved_at(attempts)
        if fr is None:
            buckets["never"] += 1
        elif fr >= 4:
            buckets["4+"] += 1
        else:
            buckets[str(fr)] += 1
    return buckets


# ===========================================================================
# Plot 1 — skillbook-size distribution across instances
# ===========================================================================

@styled("01_skillbook_size_dist.png", figsize=(12, 6.5))
def plot_skill_size_dist(fig, runs: list[Run], skill_sizes: dict[str, dict[str, int]]):
    """Box + per-instance strip of the final per-instance skillbook size, per
    run. Only multi-attempt per-instance runs with a skillbook are shown."""
    panels = {"Lite": 1, "Verified": 2}
    for bench, pos in panels.items():
        ax = fig.add_subplot(1, 2, pos)
        bench_runs = [r for r in runs
                      if r.benchmark == bench and r.has_skillbook and r.attempts >= 2]
        bench_runs = [r for r in bench_runs if skill_sizes.get(r.dir)]
        if not bench_runs:
            ax.axis("off")
            ax.set_title(f"{bench}\n(no per-instance skillbook runs)")
            continue
        order = [r.label for r in sorted(bench_runs, key=lambda r: np.median(list(skill_sizes[r.dir].values())))]
        rows = []
        for r in bench_runs:
            for v in skill_sizes[r.dir].values():
                rows.append({"run": r.label, "size": v, "backbone": r.backbone})
        df = pd.DataFrame(rows)
        palette = {r.label: BACKBONE_COLORS[r.backbone] for r in bench_runs}
        sns.boxplot(data=df, y="run", x="size", order=order, hue="run",
                    palette=palette, orient="h", fliersize=0, linewidth=1.2,
                    ax=ax, legend=False)
        sns.stripplot(data=df, y="run", x="size", order=order, color="black",
                      alpha=0.25, size=2.5, jitter=0.18, orient="h", ax=ax)
        for y, lbl in enumerate(order):
            r = next(r for r in bench_runs if r.label == lbl)
            vals = list(skill_sizes[r.dir].values())
            med = int(np.median(vals))
            n = len(vals)
            txt = ax.text(med, y, f"  med {med}  (n={n})", va="center", ha="left",
                          fontsize=8.5, color="black", fontweight="bold")
            txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])
        ax.set_title(f"{bench} — final per-instance skillbook size\n"
                     f"(only instances that triggered Learn; 1 attempt has no book)")
        ax.set_xlabel("# skills in the instance skillbook (final iteration)")
        ax.set_ylabel("")
    fig.suptitle("R1 · Skillbook-size distribution across instances",
                 fontsize=15, fontweight="bold", y=1.02)


# ===========================================================================
# Plot 1b — headline skillbook-size distribution (Figure 1a / 1b)
# ===========================================================================

@styled("01b_skillbook_size_dist_headline.png", figsize=(11, 5.8))
def plot_skill_size_headline(fig, _runs, skill_sizes: dict[str, dict[str, int]]):
    """Figure 1 — per-instance skillbook size for the headline runs:
    (a) Lite (GLM, QNext) and (b) Verified (Q30), grouped by backbone, colored
    by learn mode (default vs SWE). Seaborn violin (inner box) + jittered strip.

    ``_runs`` is unused — the run set is fixed by :data:`HEADLINE_RUNS`.
    Tells the default-vs-SWE story: on Verified the SWE reflector produces
    markedly larger per-instance books (median 12 vs 7).
    """
    panels = {"a": "Lite", "b": "Verified"}
    ylim = (0, 26)
    for tag, bench in panels.items():
        ax = fig.add_subplot(1, 2, ord(tag) - ord("a") + 1)
        sub = [r for r in HEADLINE_RUNS if r.benchmark == bench]
        rows = [{"backbone": r.backbone, "learn": r.learn, "size": v}
                for r in sub for v in skill_sizes.get(r.dir, {}).values()]
        df = pd.DataFrame(rows)
        order = sorted(df["backbone"].unique()) if len(df) else []

        sns.violinplot(data=df, x="backbone", y="size", hue="learn",
                       order=order, hue_order=["default", "swe"],
                       palette=LEARN_COLORS, split=False, inner="box", cut=0,
                       density_norm="width", linewidth=1.1, saturation=0.9, ax=ax)
        sns.stripplot(data=df, x="backbone", y="size", hue="learn",
                      order=order, hue_order=["default", "swe"],
                      palette=LEARN_COLORS, dodge=True, alpha=0.28, size=2.3,
                      jitter=0.22, linewidth=0, ax=ax, legend=False)

        # Two-line x tick labels: backbone (line 1), instance count per learn mode (line 2).
        ticklabels = []
        for bb in order:
            ndef = int(((df["backbone"] == bb) & (df["learn"] == "default")).sum())
            nswe = int(((df["backbone"] == bb) & (df["learn"] == "swe")).sum())
            ticklabels.append(f"{bb}\ndefault: {ndef}, swe: {nswe}")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(ticklabels)

        # Single clean legend (default vs swe) from the violin handles.
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), title="learn mode",
                  loc="upper left", frameon=True, facecolor="white",
                  framealpha=0.9, edgecolor="0.8", fontsize=10, title_fontsize=10)

        ax.set_title(f"({tag}) {bench}", fontsize=12.5)
        ax.set_ylabel("# skills in the final deployed book")
        ax.set_ylim(ylim)
    fig.suptitle("Per-instance skillbook size (default vs SWE)",
                 fontsize=15, fontweight="bold", y=1.03)


# ===========================================================================
# Plot 2b — headline skill growth across attempts (Figure 2)
# ===========================================================================

@styled("02b_skill_growth_headline.png", figsize=(11, 5.8))
def plot_skill_growth_headline(fig, _runs, skill_counts_all: dict[str, dict[str, list[int]]]):
    """Skill count across retries for the headline runs — seaborn lineplot of
    the mean with a central-50% (25–75 %ile) band per (backbone, learn).
    (a) Lite (GLM, QNext) and (b) Verified (Q30). Y-axis shared across panels.
    Only DEPLOYED skillbook iterations are counted (iter_1 .. iter_{attempts-1};
    the trailing book learned after the final attempt is never shown to the
    agent, so it is dropped). The population shrinks each retry as instances
    resolve and stop learning, so later bands are over fewer survivors."""
    panels = {"a": "Lite", "b": "Verified"}
    max_iter = max(r.attempts for r in HEADLINE_RUNS) - 1  # last deployed index
    ylim = (0, 15)  # shared across both panels, cropped to the bulk of the data
    for tag, bench in panels.items():
        ax = fig.add_subplot(1, 2, ord(tag) - ord("a") + 1)
        sub = [r for r in HEADLINE_RUNS if r.benchmark == bench]
        rows = []
        for r in sub:
            for counts in skill_counts_all.get(r.dir, {}).values():
                for i, c in enumerate(counts, start=1):
                    rows.append({"iter": i, "backbone": r.backbone,
                                 "learn": r.learn, "size": c})
        df = pd.DataFrame(rows)
        order = sorted(df["backbone"].unique()) if len(df) else []
        sns.lineplot(data=df, x="iter", y="size",
                     hue="backbone", hue_order=order, palette=BACKBONE_COLORS,
                     style="learn", style_order=["default", "swe"],
                     errorbar=("pi", 50), err_kws={"alpha": 0.16, "linewidth": 0},
                     markers=True, markersize=6, lw=2.0, ax=ax)
        ax.set_title(f"({tag}) {bench}", fontsize=12.5)
        ax.set_xlabel("skillbook iteration")
        ax.set_ylabel("skills per instance")
        ax.set_xticks(range(1, max_iter + 1))
        ax.set_ylim(ylim)
        ax.legend(loc="lower right", frameon=True, facecolor="white",
                  framealpha=0.9, edgecolor="0.8", fontsize=8, title_fontsize=8)
    fig.suptitle("Skill count across retries (default vs SWE)",
                 fontsize=15, fontweight="bold", y=1.03)

@styled("02_skill_growth_by_attempt.png", figsize=(12, 6))
def plot_skill_growth(fig, runs: list[Run], skill_counts_all: dict[str, dict[str, list[int]]]):
    """Mean skill_count per iteration (within-instance accumulation). The
    population shrinks per iter as instances resolve and stop learning."""
    panels = {"Lite": 1, "Verified": 2}
    for bench, pos in panels.items():
        ax = fig.add_subplot(1, 2, pos)
        bench_runs = [r for r in runs
                      if r.benchmark == bench and r.has_skillbook and r.attempts >= 2]
        any_drawn = False
        for r in bench_runs:
            sc = skill_counts_all.get(r.dir, {})
            if not sc:
                continue
            max_iter = max(len(c) for c in sc.values())
            xs, means, ns = [], [], []
            for it in range(1, max_iter + 1):
                vals = [c[it - 1] for c in sc.values() if len(c) >= it]
                if vals:
                    xs.append(it)
                    means.append(float(np.mean(vals)))
                    ns.append(len(vals))
            if xs:
                c = BACKBONE_COLORS[r.backbone]
                ax.plot(xs, means, marker="o", ms=4, lw=1.8, color=c,
                        linestyle=LEARN_DASH.get(r.learn, "-"),
                        label=f"{r.label}  (n→{ns[-1]})")
                any_drawn = True
        if not any_drawn:
            ax.axis("off")
            ax.set_title(f"{bench}\n(no skillbook runs)")
            continue
        ax.set_title(f"{bench} — mean skill count per attempt")
        ax.set_xlabel("attempt (skillbook iteration)")
        ax.set_ylabel("mean # skills per instance")
        ax.set_xticks(range(1, 7))
        ax.legend(fontsize=7.5, loc="lower right",
                  frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8")
    fig.suptitle("R1 · Skill accumulation across retries (per-instance book grows)",
                 fontsize=15, fontweight="bold", y=1.02)


# ===========================================================================
# Plot 3 — pass@k scaling curves
# ===========================================================================

@styled("03_passk_scaling.png", figsize=(13, 6))
def plot_passk_scaling(fig, runs: list[Run], resolved_all: dict[str, dict[str, list[bool]]]):
    """pass@k = fraction of instances resolved within the first k attempts."""
    panels = {"Lite": 1, "Verified": 2}
    for bench, pos in panels.items():
        ax = fig.add_subplot(1, 2, pos)
        bench_runs = [r for r in runs if r.benchmark == bench]
        max_k = max((r.attempts for r in bench_runs), default=1)
        for r in bench_runs:
            res = resolved_all.get(r.dir, {})
            if not res:
                continue
            curve = passk_curve(res, max_k) * 100
            ks = np.arange(1, max_k + 1)
            c = BACKBONE_COLORS[r.backbone]
            ax.plot(ks, curve, marker="o", ms=4.5, lw=2.0, color=c,
                    linestyle=LEARN_DASH.get(r.learn, "-"),
                    label=f"{r.label} → {curve[-1]:.1f}%")
        ax.set_title(f"{bench} — pass@k (test-time scaling)")
        ax.set_xlabel("k  (attempts allowed)")
        ax.set_ylabel("pass@k  (% instances solved)")
        ax.set_xticks(range(1, max_k + 1))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("R1 · Within-instance scaling: pass@1 → pass@N",
                 fontsize=15, fontweight="bold", y=1.02)


# ===========================================================================
# Plot 4 — lift vs pass@1, with significance
# ===========================================================================

def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


@styled("04_lift_vs_pass1.png", figsize=(12, 6.5))
def plot_lift_vs_pass1(fig, runs: list[Run], resolved_all: dict[str, dict[str, list[bool]]]):
    """Δ = pass@N − pass@1 (retry-driven lift), colored by backbone, annotated
    with the within-run paired-t significance (*** / ** / * / ns)."""
    panels = {"Lite": 1, "Verified": 2}
    for bench, pos in panels.items():
        ax = fig.add_subplot(1, 2, pos)
        bench_runs = [r for r in runs if r.benchmark == bench]
        items = []
        for r in bench_runs:
            res = resolved_all.get(r.dir, {})
            if not res:
                continue
            curve = passk_curve(res, r.attempts)
            any_, i0 = per_instance_delta(res, r.attempts)
            # Skip the t-test when there is no lift at all (e.g. 1-attempt runs:
            # pass@N == pass@1 ⇒ x == y ⇒ degenerate, would warn).
            t = ttest_paired(any_, i0) if np.any(any_ - i0) else {"p": np.nan}
            items.append((r, (curve[-1] - curve[0]) * 100, t.get("p")))
        items.sort(key=lambda it: it[1])
        labels = [r.label for r, _, _ in items]
        deltas = [d for _, d, _ in items]
        colors = [BACKBONE_COLORS[r.backbone] for r, _, _ in items]
        sigs = [_sig_marker(p) for _, _, p in items]
        y = np.arange(len(labels))
        ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.5)
        for yi, (d, s) in enumerate(zip(deltas, sigs)):
            ax.text(d + 0.15, yi, f"+{d:.1f} {s}", va="center", ha="left", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(f"{bench} — retry lift Δ(pass@N − pass@1)\n"
                     f"(paired t on per-instance lift: *** p<.001, ** p<.01, * p<.05)")
        ax.set_xlabel("lift (percentage points)")
        ax.axvline(0, color="black", lw=0.8)
    fig.suptitle("R1 · Retry-driven lift over the single-attempt floor",
                 fontsize=15, fontweight="bold", y=1.02)


# ===========================================================================
# Plot 5 — first-resolved-attempt distribution
# ===========================================================================

@styled("05_first_resolved_attempt.png", figsize=(13, 6.5))
def plot_first_resolved(fig, runs: list[Run], resolved_all: dict[str, dict[str, list[bool]]]):
    """Where the solves come from: how many instances are first resolved at
    attempt 1 / 2 / 3 / 4+ (or never). Visualizes the i1 / i2+ columns of R1."""
    bucket_colors = {"1": "#2ca02c", "2": "#1f77b4", "3": "#9467bd",
                     "4+": "#ff7f0e", "never": "#bcbcbc"}
    panels = {"Lite": 1, "Verified": 2}
    for bench, pos in panels.items():
        ax = fig.add_subplot(1, 2, pos)
        bench_runs = [r for r in runs if r.benchmark == bench]
        bench_runs = [r for r in bench_runs if resolved_all.get(r.dir)]
        bench_runs.sort(key=lambda r: sum(first_resolved_buckets(resolved_all[r.dir]).get(b, 0)
                                          for b in ("2", "3", "4+")))
        labels = [r.label for r in bench_runs]
        y = np.arange(len(labels))
        left = np.zeros(len(labels))
        for bucket in ("1", "2", "3", "4+", "never"):
            widths = []
            for r in bench_runs:
                b = first_resolved_buckets(resolved_all[r.dir])
                total = sum(b.values())
                widths.append(100 * b.get(bucket, 0) / total if total else 0)
            widths = np.array(widths)
            ax.barh(y, widths, left=left, color=bucket_colors[bucket],
                    edgecolor="white", linewidth=0.4, label=bucket)
            for yi, w in zip(y, widths):
                if w >= 10:
                    ax.text(left[yi] + w / 2, yi, f"{w:.0f}", va="center",
                            ha="center", fontsize=8, color="white", fontweight="bold")
            left += widths
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 100)
        ax.set_title(f"{bench} — attempt at first resolve (% of instances)")
        ax.set_xlabel("% of instances")
        ax.legend(title="first solved", fontsize=8, title_fontsize=9, loc="lower right",
                  frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8")
    fig.suptitle("R1 · Where the solves come from (i1 vs i2+)",
                 fontsize=15, fontweight="bold", y=1.02)


# ===========================================================================
# Plot 6 — bootstrap-CI forest plot of within-run lift
# ===========================================================================

@styled("06_lift_bootstrap_forest.png", figsize=(12, 6.5))
def plot_forest(fig, runs: list[Run], resolved_all: dict[str, dict[str, list[bool]]]):
    """Within-run lift Δ = resolved_any − iter0 with a 10 000-resample instance
    bootstrap 95% CI (reuses q1_stat_tests.boot_ci)."""
    panels = {"Lite": 1, "Verified": 2}
    for pi, (bench, pos) in enumerate(panels.items()):
        ax = fig.add_subplot(1, 2, pos)
        bench_runs = [r for r in runs if r.benchmark == bench]
        items = []
        for i, r in enumerate(bench_runs):
            res = resolved_all.get(r.dir, {})
            if not res:
                continue
            any_, i0 = per_instance_delta(res, r.attempts)
            ci = boot_ci(any_, i0, seed=SEED + pi * 1000 + i)
            items.append((r, ci["mean"] * 100, ci["ci_lo"] * 100, ci["ci_hi"] * 100))
        items.sort(key=lambda it: it[1])
        labels = [r.label for r, *_ in items]
        y = np.arange(len(labels))
        for yi, (r, mean, lo, hi) in enumerate(items):
            c = BACKBONE_COLORS[r.backbone]
            ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(mean, yi, "o", color=c, ms=8, markeredgecolor="black", markeredgewidth=0.5)
        ax.axvline(0, color="black", lw=1.0, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(f"{bench} — within-run lift Δ (95% bootstrap CI)")
        ax.set_xlabel("lift (percentage points)")
    fig.suptitle("R1 · Retry lift with instance-bootstrap 95% CIs",
                 fontsize=15, fontweight="bold", y=1.02)


# ===========================================================================
# Driver
# ===========================================================================

PLOTS = {
    "skill_size_dist": plot_skill_size_dist,
    "skill_size_headline": plot_skill_size_headline,
    "skill_growth_headline": plot_skill_growth_headline,
    "skill_growth": plot_skill_growth,
    "passk_scaling": plot_passk_scaling,
    "lift_vs_pass1": plot_lift_vs_pass1,
    "first_resolved": plot_first_resolved,
    "forest": plot_forest,
}


def _load_all(runs: list[Run]):
    resolved_all, skill_counts_all, skill_sizes_all = {}, {}, {}
    for r in runs:
        if not r.path.exists():
            print(f"  WARN: missing run dir {r.path}", file=sys.stderr)
            continue
        resolved_all[r.dir] = load_resolved_per_attempt(r.path)
        if r.has_skillbook:
            # Keep only deployed iterations (drop the unused trailing book
            # learned after the final attempt) so all runs compare like-for-like.
            sc = deployed_skill_counts(load_skill_counts(r.path), r.attempts)
            skill_counts_all[r.dir] = sc
            skill_sizes_all[r.dir] = final_skill_sizes(sc)
    return resolved_all, skill_counts_all, skill_sizes_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--only", nargs="+", choices=list(PLOTS),
                    help="subset of plots to generate")
    ap.add_argument("--format", choices=["png", "svg", "both"], default="png",
                    help="output format (svg = vector for publication)")
    args = ap.parse_args()

    selected = args.only or list(PLOTS)
    fmts = ["png", "svg"] if args.format == "both" else [args.format]
    runs = [r for r in R1_RUNS if r.path.exists()]
    print(f"Loading {len(runs)}/{len(R1_RUNS)} runs …")
    resolved_all, skill_counts_all, skill_sizes_all = _load_all(runs)

    for key in selected:
        fn = PLOTS[key]
        for fmt in fmts:
            if key in ("skill_size_dist", "skill_size_headline"):
                fn(runs, skill_sizes_all, out_dir=args.out_dir, fmt=fmt)
            elif key in ("skill_growth", "skill_growth_headline"):
                fn(runs, skill_counts_all, out_dir=args.out_dir, fmt=fmt)
            else:
                fn(runs, resolved_all, out_dir=args.out_dir, fmt=fmt)
    print(f"\nDone — {len(selected)} figure(s) × {len(fmts)} format(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
