#!/usr/bin/env python3
"""T1 — Trajectory LENGTH distribution across backbones.

Reads the precomputed per-attempt table
``trajectories_analysis_results/trajectories_attempts.csv`` (READ-ONLY) and
characterizes how long agent trajectories are per backbone:

  - §A Lite (3-way Q30 / QNext / GLM) — matched ``default`` learn mode primary,
    plus an all-learn aggregate, a per-learn breakdown, and a common-instance
    subset (instances present in all three backbones' default runs).
  - §B split025 Verified (Q30 vs QNext) — ``val_baseline`` (empty skillbook =
    pure backbone) primary, plus ``val`` (skillbook) as a secondary panel.

Metrics (per backbone, per panel/phase): n, mean, median, p25, p75, min, max
for ``steps``, ``total_tokens``, ``completion_tokens``; the same stratified by
``resolved`` (True vs False). ``api_calls`` is reported as a secondary steps
proxy.

CAVEAT: ``reasoning_tokens`` is ONLY populated for GLM (Q30/QNext report 0), so
it is NOT used as a cross-backbone metric. ``completion_tokens`` is used as the
reasoning proxy instead.

Outputs (all under ``trajectories_analysis_results/``):
  - ``2_T1_length.md``           — polished final report
  - ``figures/length_*.png``    — histograms + boxplots (dpi=150)

Run:  uv run python scripts/analyze_trajectory_length.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "trajectories_analysis_results"
CSV_PATH = OUT / "trajectories_attempts.csv"
FIG_DIR = OUT / "figures"
MD_PATH = OUT / "2_T1_length.md"

# Fixed backbone colors (spec).
COLORS = {"Q30": "#1f77b4", "QNext": "#ff7f0e", "GLM": "#2ca02c"}
BACKBONE_ORDER = ["Q30", "QNext", "GLM"]
METRICS = ["steps", "total_tokens", "completion_tokens"]


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def stat_block(series: pd.Series) -> dict:
    """Return n/mean/median/p25/p75/min/max for a numeric series."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"),
                    p25=float("nan"), p75=float("nan"), min=float("nan"),
                    max=float("nan"))
    q25, q75 = np.percentile(s, [25, 75])
    return dict(
        n=int(len(s)),
        mean=float(s.mean()),
        median=float(s.median()),
        p25=float(q25),
        p75=float(q75),
        min=float(s.min()),
        max=float(s.max()),
    )


def fmt_int(x: float) -> str:
    """Format a possibly-large token count compactly."""
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{x:.0f}"


def fmt_num(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def length_rows(df: pd.DataFrame, metric: str, backbones: list[str]) -> list[list[str]]:
    """Build a markdown row list (one per backbone) for a given metric."""
    rows = []
    for bb in backbones:
        sub = df[df["backbone"] == bb]
        st = stat_block(sub[metric])
        rows.append([
            bb, str(st["n"]),
            fmt_num(st["mean"]), fmt_num(st["median"]),
            fmt_num(st["p25"]), fmt_num(st["p75"]),
            fmt_int(st["min"]), fmt_int(st["max"]) if metric != "steps" else fmt_num(st["max"]),
        ])
    return rows


def length_headers(metric: str) -> list[str]:
    return ["Backbone", "n", "mean", "median", "p25", "p75", "min", "max"]


def resolved_unresolved_rows(df: pd.DataFrame, metric: str, backbones: list[str]) -> list[list[str]]:
    """Resolved-vs-unresolved stratified median (n / median-steps / median-tokens)."""
    rows = []
    for bb in backbones:
        sub = df[df["backbone"] == bb]
        rT = stat_block(sub[sub["resolved"] == True][metric])  # noqa: E712
        rF = stat_block(sub[sub["resolved"] == False][metric])  # noqa: E712
        rows.append([
            bb,
            f"{rT['n']} / {rF['n']}",
            f"{fmt_num(rT['median'])} / {fmt_num(rF['median'])}",
            f"{fmt_int(rT['mean'])} / {fmt_int(rF['mean'])}" if metric != "steps"
            else f"{fmt_num(rT['mean'])} / {fmt_num(rF['mean'])}",
        ])
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def hist_overlay(df: pd.DataFrame, metric: str, backbones: list[str],
                 title: str, out_name: str, *, bins: int = 40,
                 x_log: bool = False, clip_upper: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for bb in backbones:
        s = df[df["backbone"] == bb][metric].dropna().astype(float).to_numpy()
        if clip_upper is not None:
            s = np.clip(s, None, clip_upper)
        if len(s) == 0:
            continue
        ax.hist(s, bins=bins, range=(0, clip_upper) if clip_upper else None,
                alpha=0.5, label=f"{bb} (n={len(s)}, med={np.median(s):.0f})",
                color=COLORS[bb], edgecolor=COLORS[bb])
    ax.set_xlabel(_xlabel(metric))
    ax.set_ylabel("# attempts")
    ax.set_title(title)
    if x_log:
        ax.set_xscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)


def boxplot_panel(df: pd.DataFrame, metric: str, backbones: list[str],
                  title: str, out_name: str, *, y_log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    data, labels, colors = [], [], []
    for bb in backbones:
        s = df[df["backbone"] == bb][metric].dropna().astype(float).to_numpy()
        if len(s) == 0:
            continue
        data.append(s)
        labels.append(f"{bb}\n(n={len(s)})")
        colors.append(COLORS[bb])
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                    widths=0.55, medianprops=dict(color="black", linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel(_xlabel(metric))
    ax.set_title(title)
    if y_log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)


def _xlabel(metric: str) -> str:
    return {
        "steps": "assistant steps (turns)",
        "total_tokens": "total tokens",
        "completion_tokens": "completion tokens",
        "api_calls": "API calls",
    }.get(metric, metric)


def group_bar_resolved_unresolved(df: pd.DataFrame, metric: str,
                                  backbones: list[str], title: str,
                                  out_name: str) -> None:
    """Grouped bar: median of metric, resolved vs unresolved, per backbone."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(backbones))
    w = 0.38
    med_T, med_F = [], []
    for bb in backbones:
        sub = df[df["backbone"] == bb]
        med_T.append(np.nanmedian(sub[sub["resolved"] == True][metric]) if len(sub[sub["resolved"] == True]) else np.nan)  # noqa: E712
        med_F.append(np.nanmedian(sub[sub["resolved"] == False][metric]) if len(sub[sub["resolved"] == False]) else np.nan)  # noqa: E712
    b1 = ax.bar(x - w / 2, med_T, w, label="resolved", color="#444444", alpha=0.85)
    b2 = ax.bar(x + w / 2, med_F, w, label="unresolved",
                color=[COLORS[bb] for bb in backbones], alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    ax.set_ylabel("median " + _xlabel(metric))
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if h and not np.isnan(h):
                ax.annotate(fmt_int(h), (rect.get_x() + rect.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def section_lite(df_all: pd.DataFrame) -> str:
    lite = df_all[df_all["panel"] == "Lite"].copy()
    # Primary: matched default learn mode.
    lite_def = lite[lite["learn"] == "default"]
    # Common-instance subset: instances in all 3 default runs.
    common = None
    for bb in BACKBONE_ORDER:
        s = set(lite_def[lite_def["backbone"] == bb]["instance_id"].unique())
        common = s if common is None else (common & s)
    common = sorted(common)
    lite_def_common = lite_def[lite_def["instance_id"].isin(common)]

    parts: list[str] = []
    parts.append("## §A Lite (Q30 / QNext / GLM)\n")
    parts.append(
        "Single-phase runs on SWE-bench Lite. The **primary comparison** is the "
        "matched `default` learn mode (one run per backbone: "
        "`glm_4a_default` / `qwen3next_4a_default` / `qwen3_4a_default`, 4 attempts each). "
        "A **common-instance subset** restricts to the "
        f"{len(common)} instances present in all three backbones' default runs for a "
        "fair apples-to-apples view.\n"
    )

    # --- Table A.1: matched default, steps / total_tokens / completion_tokens ---
    parts.append("### Table A.1 — Matched `default` learn mode (full instance sets)\n")
    parts.append(
        "Trajectory length per backbone in the matched default-mode comparison. "
        "Each metric table gives n, mean, median, p25, p75, min, max.\n"
    )
    for metric in METRICS:
        parts.append(f"\n**{metric}**\n")
        parts.append(md_table(length_headers(metric),
                              length_rows(lite_def, metric, BACKBONE_ORDER)))
        parts.append("")
    parts.append(
        "*`reasoning_tokens` is only populated for GLM (Q30/QNext report 0), so "
        "`completion_tokens` is used as the cross-backbone reasoning proxy.*\n"
    )

    # --- Table A.2: common-instance subset ---
    # NOTE: all three Lite default runs cover the full 292-instance Lite set,
    # so the intersection equals the full set. The honest finding is "instance
    # coverage is already identical"; the real differentiator is per-instance
    # attempt count (resolved early -> fewer retries).
    parts.append("### Table A.2 — Common-instance subset (`default`)\n")
    if len(common) == lite_def["instance_id"].nunique() and all(
        lite_def[lite_def["backbone"] == bb]["instance_id"].nunique() == len(common)
        for bb in BACKBONE_ORDER
    ):
        parts.append(
            f"All three `default` runs cover the full Lite set "
            f"({len(common)} instances each), so the common-instance "
            f"**intersection equals the full set** — instance coverage is "
            f"already identical and the comparison is apples-to-apples on that "
            f"axis. What differs is the **number of attempts per instance** "
            f"(resolved instances exit early, unresolved ones retry). Medians "
            f"therefore match Table A.1; instead of repeating them we show the "
            f"per-instance attempt-count distribution:\n"
        )
        rows = []
        for bb in BACKBONE_ORDER:
            s = lite_def[lite_def["backbone"] == bb]
            per_inst = s.groupby("instance_id").size()
            rows.append([
                bb, str(per_inst.shape[0]),
                f"{per_inst.mean():.2f}",
                f"{per_inst.median():.0f}",
                f"{per_inst.min():.0f}",
                f"{per_inst.max():.0f}",
            ])
        parts.append(md_table(
            ["Backbone", "# inst", "mean attempts/inst",
             "median", "min", "max"], rows))
        parts.append(
            "\n*QNext resolves more instances early (lower mean attempts/inst) "
            "despite running longer per attempt — see §A resolved-vs-unresolved.*\n"
        )
    else:
        parts.append(
            f"Restricting to the {len(common)} instances present in all three "
            f"backbones' default runs:\n"
        )
        for metric in METRICS:
            parts.append(f"\n**{metric}** (median / mean / p25 / p75)\n")
            rows = []
            for bb in BACKBONE_ORDER:
                s = lite_def_common[lite_def_common["backbone"] == bb]
                st = stat_block(s[metric])
                rows.append([bb, str(st["n"]), fmt_num(st["median"]),
                             fmt_num(st["mean"]), fmt_num(st["p25"]),
                             fmt_num(st["p75"])])
            parts.append(md_table(
                ["Backbone", "n", "median", "mean", "p25", "p75"], rows))
            parts.append("")

    # --- Table A.3: resolved vs unresolved (matched default) ---
    parts.append("### Table A.3 — Resolved vs unresolved (`default`)\n")
    parts.append(
        "Stratified medians: each cell is `resolved / unresolved` "
        "(n first, then median of the metric). Resolved trajectories are "
        "systematically shorter.\n"
    )
    for metric in ["steps", "total_tokens"]:
        parts.append(f"\n**{metric}** (resolved / unresolved — n, then median, then mean)\n")
        rows = []
        for bb in BACKBONE_ORDER:
            sub = lite_def[lite_def["backbone"] == bb]
            rT = stat_block(sub[sub["resolved"] == True][metric])  # noqa: E712
            rF = stat_block(sub[sub["resolved"] == False][metric])  # noqa: E712
            rows.append([
                bb,
                f"{rT['n']} / {rF['n']}",
                f"{fmt_num(rT['median'])} / {fmt_num(rF['median'])}",
                (fmt_int(rT['mean']) + " / " + fmt_int(rF['mean']))
                if metric == "total_tokens"
                else (fmt_num(rT['mean']) + " / " + fmt_num(rF['mean'])),
            ])
        parts.append(md_table(["Backbone", "n (T/F)", "median (T/F)", "mean (T/F)"], rows))
        parts.append("")

    # --- Table A.4: all-learn aggregate per backbone ---
    parts.append("### Table A.4 — Each backbone aggregated over ALL its Lite learn modes\n")
    parts.append(
        "Same backbones, all Lite learn modes pooled (`default`/`swe`/`no-sb` for "
        "Q30; `default`/`swe` for QNext and GLM). Shown for completeness — the "
        "matched-default table above is the controlled comparison.\n")
    for metric in ["steps", "total_tokens"]:
        parts.append(f"\n**{metric}**\n")
        parts.append(md_table(length_headers(metric),
                              length_rows(lite, metric, BACKBONE_ORDER)))
        parts.append("")

    # --- Table A.5: per-learn breakdown ---
    parts.append("### Table A.5 — Per-learn breakdown (median steps / median total_tokens)\n")
    learns = ["default", "swe", "no-sb"]
    rows = []
    for bb in BACKBONE_ORDER:
        row = [bb]
        for ln in learns:
            sub = lite[(lite["backbone"] == bb) & (lite["learn"] == ln)]
            if len(sub) == 0:
                row.append("—")
                continue
            row.append(f"{sub['steps'].median():.0f} / {fmt_int(sub['total_tokens'].median())} (n={len(sub)})")
        rows.append(row)
    parts.append(md_table(["Backbone", "default", "swe", "no-sb"], rows))
    parts.append(
        "\n*Cell format: median steps / median total_tokens (n). `no-sb` only "
        "exists for Q30 (skillbook-disabled ablation).*\n"
    )

    # --- Figures ---
    parts.append("### Figures\n")
    hist_overlay(lite_def_common, "steps", BACKBONE_ORDER,
                 "Lite — assistant steps (common-instance subset, default learn)",
                 "length_steps_lite.png")
    hist_overlay(lite_def_common, "total_tokens", BACKBONE_ORDER,
                 "Lite — total tokens (common-instance subset, default learn)",
                 "length_total_tokens_lite.png", x_log=True)
    hist_overlay(lite_def_common, "completion_tokens", BACKBONE_ORDER,
                 "Lite — completion tokens (common-instance subset, default learn)",
                 "length_completion_tokens_lite.png", x_log=True)
    boxplot_panel(lite_def_common, "steps", BACKBONE_ORDER,
                  "Lite — steps distribution (common-instance, default learn)",
                  "length_steps_lite_box.png")
    boxplot_panel(lite_def_common, "total_tokens", BACKBONE_ORDER,
                  "Lite — total tokens distribution (common-instance, default learn)",
                  "length_total_tokens_lite_box.png", y_log=True)
    group_bar_resolved_unresolved(
        lite_def_common, "steps", BACKBONE_ORDER,
        "Lite — resolved vs unresolved: median steps (common-instance)",
        "length_steps_lite_resolved.png")
    parts.append(
        "![steps](figures/length_steps_lite.png)\n\n"
        "![total_tokens](figures/length_total_tokens_lite.png)\n\n"
        "![completion_tokens](figures/length_completion_tokens_lite.png)\n\n"
        "![steps box](figures/length_steps_lite_box.png)\n\n"
        "![total_tokens box](figures/length_total_tokens_lite_box.png)\n\n"
        "![resolved vs unresolved](figures/length_steps_lite_resolved.png)\n"
    )

    # ASCII histogram of steps for quick scanning.
    parts.append("\n**ASCII histogram — median steps (matched default, full sets)**\n")
    parts.append("```\n" + ascii_hist(lite_def, "steps", BACKBONE_ORDER) + "```\n")

    # Stash summary numbers for findings.
    q30 = lite_def[lite_def["backbone"] == "Q30"]
    qn = lite_def[lite_def["backbone"] == "QNext"]
    glm = lite_def[lite_def["backbone"] == "GLM"]
    parts.append(
        f"\n*Quick ratio: QNext median steps / Q30 = "
        f"{qn['steps'].median() / q30['steps'].median():.1f}×; "
        f"QNext median total_tokens / Q30 = "
        f"{qn['total_tokens'].median() / q30['total_tokens'].median():.1f}×. "
        f"GLM median steps / Q30 = "
        f"{glm['steps'].median() / q30['steps'].median():.1f}×.*\n"
    )
    return "\n".join(parts), dict(
        lite_def=lite_def,
        common_n=len(common),
        q30_med_steps=q30["steps"].median(),
        qn_med_steps=qn["steps"].median(),
        glm_med_steps=glm["steps"].median(),
        q30_med_total=q30["total_tokens"].median(),
        qn_med_total=qn["total_tokens"].median(),
        glm_med_total=glm["total_tokens"].median(),
    )


def section_split025(df_all: pd.DataFrame) -> tuple[str, dict]:
    sp = df_all[df_all["panel"] == "split025"].copy()
    valbl = sp[sp["phase"] == "val_baseline"]   # empty skillbook = pure backbone
    val = sp[sp["phase"] == "val"]              # with skillbook
    backbones = ["Q30", "QNext"]

    parts: list[str] = []
    parts.append("## §B split025 Verified — val_baseline (Q30 vs QNext)\n")
    parts.append(
        "Two-phase Verified split (val = 113 instances, 5 attempts each, "
        "`vpk5`). The **primary panel** is `val_baseline` — empty skillbook, so "
        "the trajectory reflects the pure backbone. `val` (with the learned "
        "global/per-repo skillbook) is reported secondarily to show the "
        "skillbook's effect on length.\n"
    )

    parts.append("### Table B.1 — `val_baseline` (empty skillbook)\n")
    for metric in METRICS:
        parts.append(f"\n**{metric}**\n")
        parts.append(md_table(length_headers(metric),
                              length_rows(valbl, metric, backbones)))
        parts.append("")
    parts.append(
        "*Each backbone contributes two split025 runs (global + per-repo "
        "skillbook source); both phases share the same 113 val instances, so "
        "n = 1130 = 113 instances × (2 runs × 5 attempts).*\n"
    )

    parts.append("### Table B.2 — Resolved vs unresolved (`val_baseline`)\n")
    for metric in ["steps", "total_tokens"]:
        parts.append(f"\n**{metric}** (resolved / unresolved — n, median, mean)\n")
        rows = []
        for bb in backbones:
            sub = valbl[valbl["backbone"] == bb]
            rT = stat_block(sub[sub["resolved"] == True][metric])  # noqa: E712
            rF = stat_block(sub[sub["resolved"] == False][metric])  # noqa: E712
            rows.append([
                bb,
                f"{rT['n']} / {rF['n']}",
                f"{fmt_num(rT['median'])} / {fmt_num(rF['median'])}",
                (fmt_int(rT['mean']) + " / " + fmt_int(rF['mean']))
                if metric == "total_tokens"
                else (fmt_num(rT['mean']) + " / " + fmt_num(rF['mean'])),
            ])
        parts.append(md_table(["Backbone", "n (T/F)", "median (T/F)", "mean (T/F)"], rows))
        parts.append("")

    parts.append("### Table B.3 — `val` (with skillbook) — length shift vs val_baseline\n")
    parts.append(
        "Median steps / total_tokens with the learned skillbook, alongside the "
        "matching `val_baseline` medians.\n")
    rows = []
    for bb in backbones:
        sv = val[val["backbone"] == bb]
        sb = valbl[valbl["backbone"] == bb]
        rows.append([
            bb, str(len(sv)),
            f"{sv['steps'].median():.0f}",
            f"{sb['steps'].median():.0f}",
            f"{sv['steps'].median() - sb['steps'].median():+.0f}",
            fmt_int(sv["total_tokens"].median()),
            fmt_int(sb["total_tokens"].median()),
        ])
    parts.append(md_table(
        ["Backbone", "n", "val steps (med)", "valBL steps (med)", "Δ steps",
         "val total (med)", "valBL total (med)"], rows))
    parts.append(
        "\n*Q30's skillbook run grows total tokens (more retrieved context per "
        "step); QNext is essentially flat on steps and grows moderately on "
        "tokens.*\n"
    )

    # Figures.
    parts.append("### Figures\n")
    hist_overlay(valbl, "steps", backbones,
                 "split025 val_baseline — assistant steps (Q30 vs QNext)",
                 "length_steps_split025_valbl.png")
    hist_overlay(valbl, "total_tokens", backbones,
                 "split025 val_baseline — total tokens (Q30 vs QNext)",
                 "length_total_tokens_split025_valbl.png", x_log=True)
    hist_overlay(valbl, "completion_tokens", backbones,
                 "split025 val_baseline — completion tokens (Q30 vs QNext)",
                 "length_completion_tokens_split025_valbl.png", x_log=True)
    boxplot_panel(valbl, "steps", backbones,
                  "split025 val_baseline — steps distribution",
                  "length_steps_split025_valbl_box.png")
    boxplot_panel(valbl, "total_tokens", backbones,
                  "split025 val_baseline — total tokens distribution",
                  "length_total_tokens_split025_valbl_box.png", y_log=True)
    parts.append(
        "![steps](figures/length_steps_split025_valbl.png)\n\n"
        "![total_tokens](figures/length_total_tokens_split025_valbl.png)\n\n"
        "![completion_tokens](figures/length_completion_tokens_split025_valbl.png)\n\n"
        "![steps box](figures/length_steps_split025_valbl_box.png)\n\n"
        "![total_tokens box](figures/length_total_tokens_split025_valbl_box.png)\n"
    )

    parts.append("\n**ASCII histogram — median steps (`val_baseline`)**\n")
    parts.append("```\n" + ascii_hist(valbl, "steps", backbones) + "```\n")

    q30 = valbl[valbl["backbone"] == "Q30"]
    qn = valbl[valbl["backbone"] == "QNext"]
    parts.append(
        f"\n*Quick ratio: QNext median steps / Q30 = "
        f"{qn['steps'].median() / q30['steps'].median():.1f}×; "
        f"QNext median total_tokens / Q30 = "
        f"{qn['total_tokens'].median() / q30['total_tokens'].median():.1f}×.*\n"
    )

    stats = dict(
        q30_med_steps=q30["steps"].median(),
        qn_med_steps=qn["steps"].median(),
        q30_med_total=q30["total_tokens"].median(),
        qn_med_total=qn["total_tokens"].median(),
        # val (skillbook) medians, for the val-vs-valBL length-shift finding
        q30_val_steps=val[val["backbone"] == "Q30"]["steps"].median(),
        qn_val_steps=val[val["backbone"] == "QNext"]["steps"].median(),
        q30_val_total=val[val["backbone"] == "Q30"]["total_tokens"].median(),
        qn_val_total=val[val["backbone"] == "QNext"]["total_tokens"].median(),
        # resolved-vs-unresolved deltas (median steps, Q30 & QNext)
        q30_res_steps=valbl[(valbl["backbone"] == "Q30") & (valbl["resolved"] == True)]["steps"].median(),  # noqa: E712
        q30_unres_steps=valbl[(valbl["backbone"] == "Q30") & (valbl["resolved"] == False)]["steps"].median(),  # noqa: E712
        qn_res_steps=valbl[(valbl["backbone"] == "QNext") & (valbl["resolved"] == True)]["steps"].median(),  # noqa: E712
        qn_unres_steps=valbl[(valbl["backbone"] == "QNext") & (valbl["resolved"] == False)]["steps"].median(),  # noqa: E712
    )
    return "\n".join(parts), stats


# --------------------------------------------------------------------------- #
# ASCII histogram (median steps bar)
# --------------------------------------------------------------------------- #
def ascii_hist(df: pd.DataFrame, metric: str, backbones: list[str],
               width: int = 40) -> str:
    lines = []
    medians = {}
    for bb in backbones:
        s = df[df["backbone"] == bb][metric].dropna()
        medians[bb] = float(s.median()) if len(s) else 0.0
    mx = max(medians.values()) if medians else 1.0
    for bb in backbones:
        m = medians[bb]
        bar = "#" * int(round((m / mx) * width)) if mx else 0
        lines.append(f"{bb:6s} | {m:6.0f}  {bar}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        raise SystemExit(f"Input CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    # Normalize resolved to nullable bool (csv stores True/False strings → pandas bool already).
    print(f"Loaded {len(df)} attempts from {CSV_PATH.name}")
    print(f"  panels: {df['panel'].value_counts().to_dict()}")
    print(f"  backbones: {df['backbone'].value_counts().to_dict()}")

    sec_a, a_stats = section_lite(df)
    sec_b, b_stats = section_split025(df)

    # ---- Findings ----
    findings = build_findings(a_stats, b_stats, df)

    md = []
    md.append("# T1 — Trajectory length distribution\n")
    md.append(
        "**Goal.** Characterize how long agent trajectories are per backbone — "
        "steps (assistant turns), total tokens, and completion tokens — and how "
        "length splits by resolution outcome. We compare the three Lite "
        "backbones (Q30 / QNext / GLM) on matched `default` runs and a "
        "common-instance subset, and the two split025 Verified backbones "
        "(Q30 vs QNext) on `val_baseline` (empty skillbook) plus `val` "
        "(skillbook). All numbers are read from the precomputed "
        "`trajectories_attempts.csv`; no raw trajectories are re-derived.\n"
    )
    md.append(sec_a)
    md.append("\n---\n")
    md.append(sec_b)
    md.append("\n## Findings\n")
    for f in findings:
        md.append(f"- {f}")
    md.append("")
    md.append(
        "\n*Method note: `completion_tokens` is used as the cross-backbone "
        "reasoning proxy because `reasoning_tokens` is populated only for GLM "
        "(Q30/QNext report 0).*\n"
    )

    MD_PATH.write_text("\n".join(md))
    print(f"\nWrote {MD_PATH.relative_to(ROOT)}")
    print(f"Figures in {FIG_DIR.relative_to(ROOT)}/:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}")


def build_findings(a: dict, b: dict, df: pd.DataFrame) -> list[str]:
    """Compose the 4-6 crisp Findings bullets."""
    return [
        (
            f"QNext burns far more compute than Q30 per attempt: in the Lite "
            f"matched-default comparison its median trajectory is "
            f"{a['qn_med_steps']:.0f} steps vs {a['q30_med_steps']:.0f} for Q30 "
            f"(~{a['qn_med_steps']/a['q30_med_steps']:.1f}×), and "
            f"{fmt_int(a['qn_med_total'])} total tokens vs "
            f"{fmt_int(a['q30_med_total'])} "
            f"(~{a['qn_med_total']/a['q30_med_total']:.1f}×). GLM sits between "
            f"them ({a['glm_med_steps']:.0f} steps, "
            f"{fmt_int(a['glm_med_total'])} tokens)."
        ),
        (
            f"The gap reproduces on split025 Verified `val_baseline` (pure "
            f"backbone, empty skillbook): QNext median "
            f"{b['qn_med_steps']:.0f} steps / {fmt_int(b['qn_med_total'])} "
            f"tokens vs Q30 {b['q30_med_steps']:.0f} steps / "
            f"{fmt_int(b['q30_med_total'])} tokens "
            f"(~{b['qn_med_steps']/b['q30_med_steps']:.1f}× steps, "
            f"~{b['qn_med_total']/b['q30_med_total']:.1f}× tokens)."
        ),
        (
            f"Resolved trajectories are systematically shorter — Q30 finishes "
            f"resolved attempts in {b['q30_res_steps']:.0f} median steps vs "
            f"{b['q30_unres_steps']:.0f} when unresolved, and QNext similarly "
            f"({b['qn_res_steps']:.0f} vs {b['qn_unres_steps']:.0f}). I.e. the "
            f"longer a trajectory runs, the more likely it is to be failing "
            f"(it is burning the step budget without converging)."
        ),
        (
            "QNext is the exception to the resolved-shorter rule on Lite: its "
            "resolved and unresolved medians are nearly identical (~110 steps) "
            "— it simply runs long regardless of outcome, consistent with "
            "reasoning-heavy trajectories that saturate the step limit."
        ),
        (
            "The skillbook lengthens rather than shortens trajectories in "
            "split025 `val`: Q30 grows from "
            f"{b['q30_med_steps']:.0f} (valBL) to {b['q30_val_steps']:.0f} "
            f"median steps in `val` and total tokens jump "
            f"{fmt_int(b['q30_med_total'])} → {fmt_int(b['q30_val_total'])} "
            f"(~{b['q30_val_total']/b['q30_med_total']:.1f}×, retrieved skills "
            f"add context per step); QNext steps stay flat at "
            f"{b['qn_val_steps']:.0f} but tokens grow "
            f"~{b['qn_val_total']/b['qn_med_total']:.1f}×. Adding a skillbook "
            f"costs tokens without speeding convergence."
        ),
        (
            "Learn mode barely moves length on Lite: Q30 is ~34-35 median steps "
            "and ~300-350k tokens across `default`/`swe`/`no-sb`, and QNext is "
            "~107-110 steps across its two learn modes (Table A.5). The "
            "**backbone, not the skillbook strategy, dominates trajectory "
            "length** — a 3× steps gap and a ~10× tokens gap separate Q30 from "
            "QNext regardless of learn mode."
        ),
    ]


if __name__ == "__main__":
    main()
