#!/usr/bin/env python3
"""
§C.1 — in-trajectory behavioral pathologies, LITE panel (Q30 / QNext / GLM).

Reads:
  - trajectories_analysis_results/trajectories_behaviors.csv  (one row per attempt)
  - trajectories_analysis_results/trajectories_attempts.csv   (for steps/total_tokens join)

PRIMARY scope: panel=="Lite", phase is empty (NaN), matched `default` learn mode
  (qwen3_4a_default / qwen3next_4a_default / glm_4a_default) — one run per backbone.
SECONDARY: each backbone pooled over all its Lite learn modes (default/swe [+no-sb for Q30]).

Writes figures to trajectories_analysis_results/figures/behavior_*_lite.png and
prints the markdown subsection (§C.1) to stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_BEH = ROOT / "trajectories_analysis_results" / "trajectories_behaviors.csv"
CSV_ATT = ROOT / "trajectories_analysis_results" / "trajectories_attempts.csv"
FIGDIR = ROOT / "trajectories_analysis_results" / "figures"

BB_ORDER = ["Q30", "QNext", "GLM"]
BB_COLOR = {"Q30": "#1f77b4", "QNext": "#ff7f0e", "GLM": "#2ca02c"}
BB_LABEL = {"Q30": "Qwen3-30B (Q30)", "QNext": "Qwen3-Next (QNext)", "GLM": "GLM-4.5"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    beh = pd.read_csv(CSV_BEH)
    att = pd.read_csv(CSV_ATT)
    # Lite, empty phase (NaN in CSV) — these are the single-attempt trajectory rows
    beh = beh[(beh["panel"] == "Lite") & (beh["phase"].isna())].copy()
    return beh, att


def join_attempts(beh: pd.DataFrame, att: pd.DataFrame) -> pd.DataFrame:
    # restrict attempts to the same Lite / empty-phase scope to keep keys unique
    att = att[(att["panel"] == "Lite") & (att["phase"].isna())].copy()
    keys = ["panel", "run", "instance_id", "iter"]
    att_keys = att[keys + ["steps", "total_tokens"]].drop_duplicates(keys)
    merged = beh.merge(att_keys, on=keys, how="left", validate="many_to_one")
    return merged


def _pct(s: pd.Series) -> float:
    return 100.0 * float(s.mean())


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def fmt_med(s: pd.Series, prec: int = 2) -> str:
    v = s.median()
    return f"{v:.{prec}f}" if not pd.isna(v) else "—"


def fmt_mean(s: pd.Series, prec: int = 2) -> str:
    v = s.mean()
    return f"{v:.{prec}f}" if not pd.isna(v) else "—"


# ----------------------------------------------------------------------------- figures


def fig_repeat_ratio(beh_def: pd.DataFrame) -> Path:
    """Box/violin of cmd_repeat_ratio per backbone (matched default runs)."""
    path = FIGDIR / "behavior_repeat_lite.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = [beh_def[beh_def["backbone"] == bb]["cmd_repeat_ratio"].dropna() for bb in BB_ORDER]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
    for body, bb in zip(parts["bodies"], BB_ORDER):
        body.set_facecolor(BB_COLOR[bb])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)
    # overlay medians as text
    for i, bb in enumerate(BB_ORDER, start=1):
        med = data[i - 1].median()
        ax.text(i, med + 0.02, f"med {med:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(1, len(BB_ORDER) + 1))
    ax.set_xticklabels([BB_LABEL[bb] for bb in BB_ORDER], fontsize=9)
    ax.set_ylabel("cmd_repeat_ratio  (1 − unique/total cmds)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Command repetition ratio — Lite, matched `default` runs")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_format_trap(beh_def: pd.DataFrame) -> Path:
    """Headline bar chart: % attempts with >=1 one_action_warning (+ median warnings overlay)."""
    path = FIGDIR / "behavior_format_trap_lite.png"
    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(len(BB_ORDER))
    pct_warn = [100.0 * (beh_def[beh_def["backbone"] == bb]["n_one_action_warnings"] > 0).mean()
                for bb in BB_ORDER]
    med_warn = [beh_def[beh_def["backbone"] == bb]["n_one_action_warnings"].median()
                for bb in BB_ORDER]
    bars = ax1.bar(x, pct_warn, color=[BB_COLOR[bb] for bb in BB_ORDER], alpha=0.8,
                   edgecolor="black", linewidth=0.6, label="% attempts with ≥1 warning")
    for xi, v in zip(x, pct_warn):
        ax1.text(xi, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([BB_LABEL[bb] for bb in BB_ORDER], fontsize=9)
    ax1.set_ylabel("% attempts with ≥1 one_action warning")
    ax1.set_ylim(0, max(pct_warn) * 1.25)
    ax1.set_title("Format-trap (multi-action rejection) — Lite, matched `default` runs")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, med_warn, color="black", marker="D", markersize=7, linewidth=1.6,
             linestyle="--", label="median warnings/attempt")
    for xi, v in zip(x, med_warn):
        ax2.text(xi, v + max(med_warn) * 0.04 + 0.05, f"{v:.0f}", ha="center",
                 va="bottom", fontsize=9, color="black")
    ax2.set_ylabel("median one_action warnings / attempt")
    ax2.set_ylim(0, max([m + 0.5 for m in med_warn]) * 1.35 if max(med_warn) > 0 else 1)

    lines = [bars, ax2.lines[0]]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_cycle(beh_def: pd.DataFrame) -> Path:
    """Bar chart of has_cycle rate per backbone."""
    path = FIGDIR / "behavior_cycle_lite.png"
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = np.arange(len(BB_ORDER))
    cycle = [100.0 * beh_def[beh_def["backbone"] == bb]["has_cycle"].mean() for bb in BB_ORDER]
    bars = ax.bar(x, cycle, color=[BB_COLOR[bb] for bb in BB_ORDER], alpha=0.8,
                  edgecolor="black", linewidth=0.6)
    for xi, v in zip(x, cycle):
        ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([BB_LABEL[bb] for bb in BB_ORDER], fontsize=9)
    ax.set_ylabel("% attempts with a detected A↔B cycle (period 2–4)")
    ax.set_ylim(0, max(cycle) * 1.3 if max(cycle) > 0 else 1)
    ax.set_title("Cyclic behavior rate — Lite, matched `default` runs")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_explore(beh_def: pd.DataFrame) -> Path:
    """Box of explore_ratio per backbone."""
    path = FIGDIR / "behavior_explore_lite.png"
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    data = [beh_def[beh_def["backbone"] == bb]["explore_ratio"].dropna() for bb in BB_ORDER]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
    for body, bb in zip(parts["bodies"], BB_ORDER):
        body.set_facecolor(BB_COLOR[bb])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)
    for i, bb in enumerate(BB_ORDER, start=1):
        med = data[i - 1].median()
        ax.text(i, med + 0.02, f"med {med:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(1, len(BB_ORDER) + 1))
    ax.set_xticklabels([BB_LABEL[bb] for bb in BB_ORDER], fontsize=9)
    ax.set_ylabel("explore_ratio  (fraction of read-only cmds)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Read-only exploration share — Lite, matched `default` runs")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- markdown


def per_bb_row(sub: pd.DataFrame, bb: str) -> dict:
    return {
        "bb": bb,
        "n": len(sub),
        "repeat_med": sub["cmd_repeat_ratio"].median(),
        "repeat_mean": sub["cmd_repeat_ratio"].mean(),
        "n_repeated_med": sub["n_repeated_cmds"].median(),
        "max_consec_med": sub["max_consec_repeat"].median(),
        "consec_dup_med": sub["n_consec_dup_pairs"].median(),
        "cycle_pct": _pct(sub["has_cycle"]),
        "format_trap_pct": _pct(sub["n_one_action_warnings"] > 0),
        "warn_mean": sub["n_one_action_warnings"].mean(),
        "warn_med": sub["n_one_action_warnings"].median(),
        "explore_med": sub["explore_ratio"].median(),
        "n_edits_med": sub["n_edits"].median(),
        "dup_asst_pct": _pct(sub["n_dup_asst_msgs"] > 0),
        "resolved_pct": _pct(sub["resolved"]),
    }


def table_default(rows: list[dict]) -> str:
    out = []
    out.append("| Backbone | n | rep_ratio (med/mean) | n_repeated cmds (med) | max_consec_rep (med) | consec dup pairs (med) | cycle % |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {BB_LABEL[r['bb']]} | {r['n']} | "
            f"{r['repeat_med']:.2f} / {r['repeat_mean']:.2f} | "
            f"{r['n_repeated_med']:.0f} | {r['max_consec_med']:.0f} | "
            f"{r['consec_dup_med']:.0f} | {fmt_pct(r['cycle_pct'])} |"
        )
    return "\n".join(out)


def table_format_trap(rows: list[dict]) -> str:
    out = []
    out.append("| Backbone | n | % attempts with ≥1 warning | mean warnings/attempt | median warnings/attempt |")
    out.append("|---|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {BB_LABEL[r['bb']]} | {r['n']} | "
            f"{fmt_pct(r['format_trap_pct'])} | {r['warn_mean']:.2f} | {r['warn_med']:.0f} |"
        )
    return "\n".join(out)


def table_explore(rows: list[dict]) -> str:
    out = []
    out.append("| Backbone | n | explore_ratio (med) | n_edits (med) | dup_asst_msgs % (>0) | resolved % |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {BB_LABEL[r['bb']]} | {r['n']} | "
            f"{r['explore_med']:.2f} | {r['n_edits_med']:.0f} | "
            f"{fmt_pct(r['dup_asst_pct'])} | {fmt_pct(r['resolved_pct'])} |"
        )
    return "\n".join(out)


def table_pooled(beh_all: pd.DataFrame) -> str:
    """Secondary: pooled over all Lite learn modes per backbone."""
    out = []
    out.append("| Backbone | learn modes | n | rep_ratio (med) | cycle % | format-trap % | explore_ratio (med) |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for bb in BB_ORDER:
        sub = beh_all[beh_all["backbone"] == bb]
        modes = sorted(sub["learn"].unique())
        out.append(
            f"| {BB_LABEL[bb]} | {', '.join(modes)} | {len(sub)} | "
            f"{sub['cmd_repeat_ratio'].median():.2f} | "
            f"{fmt_pct(_pct(sub['has_cycle']))} | "
            f"{fmt_pct(_pct(sub['n_one_action_warnings'] > 0))} | "
            f"{sub['explore_ratio'].median():.2f} |"
        )
    return "\n".join(out)


def table_outcome(beh_def: pd.DataFrame) -> str:
    """Resolved vs unresolved comparison per backbone."""
    out = []
    out.append("| Backbone | outcome | n | rep_ratio (med) | format-trap % | max_consec_rep (med) |")
    out.append("|---|---|---:|---:|---:|---:|")
    for bb in BB_ORDER:
        sub = beh_def[beh_def["backbone"] == bb]
        for outcome, lab in [(True, "resolved"), (False, "unresolved")]:
            s = sub[sub["resolved"] == outcome]
            if len(s) == 0:
                continue
            out.append(
                f"| {BB_LABEL[bb]} | {lab} | {len(s)} | "
                f"{s['cmd_repeat_ratio'].median():.2f} | "
                f"{fmt_pct(_pct(s['n_one_action_warnings'] > 0))} | "
                f"{s['max_consec_repeat'].median():.0f} |"
            )
    return "\n".join(out)


def corr_block(beh_def_joined: pd.DataFrame) -> str:
    """Spearman-ish Pearson correlation of pathology metrics with trajectory length."""
    out = []
    out.append("| Backbone | corr(repeat_ratio, steps) | corr(repeat_ratio, tokens) | corr(n_warnings, steps) |")
    out.append("|---|---:|---:|---:|")
    sub = beh_def_joined.dropna(subset=["steps", "total_tokens"])
    for bb in BB_ORDER:
        s = sub[sub["backbone"] == bb]
        c_rs = s[["cmd_repeat_ratio", "steps"]].corr().iloc[0, 1]
        c_rt = s[["cmd_repeat_ratio", "total_tokens"]].corr().iloc[0, 1]
        c_ws = s[["n_one_action_warnings", "steps"]].corr().iloc[0, 1]
        out.append(
            f"| {BB_LABEL[bb]} | {c_rs:+.2f} | {c_rt:+.2f} | {c_ws:+.2f} |"
        )
    return "\n".join(out)


def build_subsection(beh_def: pd.DataFrame, beh_all: pd.DataFrame,
                     beh_def_joined: pd.DataFrame,
                     figs: dict[str, Path]) -> str:
    rows = [per_bb_row(beh_def[beh_def["backbone"] == bb], bb) for bb in BB_ORDER]

    md: list[str] = []
    md.append("### §C.1 Lite (Q30 / QNext / GLM) — in-trajectory pathologies")
    md.append("")
    md.append(
        "Per-attempt behavioral signals (command repetition, A↔B cycling, multi-action "
        "*format-trap* rejections, and read-only exploration share) on the Lite panel. "
        "PRIMARY scope is the matched `default` learn runs — one per backbone "
        "(`qwen3_4a_default` n=979, `qwen3next_4a_default` n=817, `glm_4a_default` n=807). "
        "A secondary pooled table further down aggregates each backbone over all its Lite learn modes."
    )
    md.append("")

    md.append("**Repetition & cycling — matched `default` runs.**")
    md.append("")
    md.append(table_default(rows))
    md.append("")
    md.append(
        "*cmd_repeat_ratio = 1 − unique_cmds/total_cmds (0 = no repeats, 1 = all identical). "
        "has_cycle flags a period-2–4 loop (e.g. A→B→A→B). max_consec_rep is the longest run of "
        "identical consecutive commands; n_consec_dup_pairs counts immediate retries cmd[i]==cmd[i+1].*"
    )
    md.append("")
    md.append(f"![command repetition ratio](figures/{figs['repeat'].name})")
    md.append("")
    md.append(f"![cycle rate](figures/{figs['cycle'].name})")
    md.append("")

    md.append("**Format-trap (multi-action rejection) — the headline.**")
    md.append("")
    md.append(table_format_trap(rows))
    md.append("")
    md.append(
        "*n_one_action_warnings counts harness rejections of the form "
        "“Please always provide EXACTLY ONE action in triple backticks.” "
        "— i.e. the model emitted an assistant message with ≥2 code blocks (or no code block), "
        "which the harness rejects and re-prompts. This is the known qwen3 format-rejection pathology: "
        "Q30 fires it on ~65% of attempts (median 1 warning, mean 12); GLM ~22% (median 0); "
        "QNext is essentially immune (~3%, median 0).*"
    )
    md.append("")
    md.append(f"![format trap](figures/{figs['format_trap'].name})")
    md.append("")
    md.append(
        "*Concrete callout (Q30, `django__django-16816` iter 2, 200 warnings in one attempt):* "
        "an assistant message containing two fenced blocks — a `bash` block *and* a stray "
        "`</format_example>`-containing block — triggers “Please always provide EXACTLY ONE "
        "action in triple backticks.” The agent then re-emits the same two-block structure, "
        "burning the turn budget without executing anything."
    )

    md.append("")
    md.append("**Exploration vs action & stuck loops — matched `default` runs.**")
    md.append("")
    md.append(table_explore(rows))
    md.append("")
    md.append(
        "*explore_ratio is a regex proxy (fraction of cmds matching cat/ls/grep/find/head/sed/"
        "find/etc.) and is a soft signal, not a precise classifier. n_dup_asst_msgs counts attempts "
        "with ≥2 consecutive identical full assistant messages (a hard stuck-loop signal).*"
    )
    md.append("")
    md.append(f"![explore ratio](figures/{figs['explore'].name})")
    md.append("")

    md.append("**Outcome correlation — resolved vs unresolved attempts.**")
    md.append("")
    md.append(table_outcome(beh_def))
    md.append("")
    md.append(
        "*Repetition and format-trap concentrate in UNRESOLVED attempts — resolved trajectories are "
        "shorter and cleaner. Because Lite `default` resolve rates are low (Q30 ≈6%, GLM ≈14%, "
        "QNext ≈14%), the resolved columns rest on few attempts and should be read as directional only.*"
    )
    md.append("")
    md.append("**Correlation of pathology metrics with trajectory length** (Pearson, matched `default`):")
    md.append("")
    md.append(corr_block(beh_def_joined))
    md.append(
        "*Longer trajectories (more steps / more tokens) correlate positively with command "
        "repetition; format-trap warnings co-occur with long Q30 attempts (the rejection loop inflates step count).*"
    )

    md.append("")
    md.append("**Secondary: pooled over all Lite learn modes per backbone.**")
    md.append("")
    md.append(table_pooled(beh_all))
    md.append(
        "*Aggregating across learn modes (default/swe [+no-skillbook for Q30]) leaves the backbone "
        "ordering unchanged — Q30 highest format-trap, GLM highest repetition/cycling, QNext cleanest.*"
    )
    md.append("")
    md.append("**Lite behavioral findings**")
    md.append("")
    md.append(
        "- **Format-trap is Q30-specific and severe.** ~65% of Q30 `default` attempts fire at least one "
        "“EXACTLY ONE action” rejection (median 1, mean **12** per attempt); the worst single Q30 attempt "
        "accumulates 200 such rejections. GLM sits at ~22% (median 0), QNext at ~3% (median 0). "
        "This is the known qwen3-30B multi-code-block format-rejection pathology; QNext (qwen3next) is immune, "
        "consistent with its slips being max_tokens-truncated reasoning loops rather than format failures."
    )
    md.append(
        "- **GLM repeats and cycles the most commands.** GLM has the highest cmd_repeat_ratio "
        "(median ≈0.17 vs Q30 ≈0.08 vs QNext ≈0.04) and the highest cycle rate (~14% vs ~7% vs <1%). "
        "Its most-repeated commands are reproduce-script runs (`python reproduce_issue.py`). "
        "Across all three, max_consec_repeat median = 1, so repetition is scattered, not stuck-on-one-command."
    )
    md.append(
        "- **QNext is the cleanest but under-acts.** Lowest repetition, near-zero cycling, near-zero format-trap "
        "— yet its explore_ratio median (≈0.50) is comparable to GLM and it resolves only marginally more "
        "often (~14%). Clean behavior ≠ high resolution."
    )
    md.append(
        "- **Exploration share is stable across backbones** (explore_ratio median ≈0.50–0.57). Half of executed "
        "commands are read-only (cat/grep/find/sed/head); no backbone over-explores dramatically, so explore_ratio "
        "is not a discriminator here."
    )
    md.append(
        "- **Pathology concentrates in unresolved attempts.** Median cmd_repeat_ratio and format-trap rate are "
        "both higher among unresolved trajectories, and these metrics correlate positively with step/token counts "
        "(Q30 format-trap warnings track step count most tightly). Long, repetitive, format-rejecting attempts "
        "are wasted budget, not productive search."
    )
    md.append(
        "- **Stuck identical-message loops are rare.** <4% of attempts in any backbone show a repeated consecutive "
        "full assistant message (n_dup_asst_msgs > 0); the dominant failure mode is scattered repetition and "
        "format rejection, not a single frozen message."
    )
    md.append("")
    return "\n".join(md)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    beh, att = load()

    # PRIMARY: matched default runs
    beh_def = beh[beh["learn"] == "default"].copy()

    # SECONDARY: all Lite learn modes
    beh_all = beh.copy()

    # join steps/tokens for correlation block
    beh_def_joined = join_attempts(beh_def, att)

    figs = {
        "repeat": fig_repeat_ratio(beh_def),
        "format_trap": fig_format_trap(beh_def),
        "cycle": fig_cycle(beh_def),
        "explore": fig_explore(beh_def),
    }

    md = build_subsection(beh_def, beh_all, beh_def_joined, figs)
    print(md)

    # sanity echo to stderr
    print("\n--- sanity (matched default) ---", file=sys.stderr)
    for bb in BB_ORDER:
        sub = beh_def[beh_def["backbone"] == bb]
        print(
            f"{bb}: n={len(sub)} repeat_med={sub['cmd_repeat_ratio'].median():.3f} "
            f"cycle={_pct(sub['has_cycle']):.1f}% trap={_pct(sub['n_one_action_warnings']>0):.1f}% "
            f"explore_med={sub['explore_ratio'].median():.3f} max_consec_med={sub['max_consec_repeat'].median():.0f}",
            file=sys.stderr,
        )
    for p in figs.values():
        print(f"wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
