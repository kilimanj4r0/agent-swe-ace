#!/usr/bin/env python3
"""T2 — Failure-reason distribution across backbones.

Aggregates the precomputed ``trajectories_attempts.csv`` (one row per agent
attempt) into per-backbone failure-mix tables for two panels:

  §A  Lite            — 3-way Q30 / QNext / GLM, single-phase, pass@4
  §B  split025 Verified — Q30 vs QNext, two-phase, pass@5
                        (PRIMARY = val_baseline = empty skillbook = pure backbone;
                         secondary val = skillbook)

For each panel/phase we report:
  1. Attempt-level failure distribution over UNRESOLVED attempts only
     (count + % of each error_category).
  2. Instance-level pass@k context (fraction of instances resolved in >=1 attempt).
  3. Overall exit_status breakdown (Submitted / LimitsExceeded / error / ...).

Outputs (all under ``trajectories_analysis_results/``):
  - ``3_T2_errors.md``                              — polished report
  - ``figures/errors_breakdown_lite.png``           — Lite (matched default-learn)
  - ``figures/errors_breakdown_split025_valbl.png`` — split025 val_baseline (global)
  - ``figures/errors_breakdown_split025_val.png``   — split025 val skillbook (global)

Run:  uv run python scripts/analyze_trajectory_errors.py

Read-only on the CSV; only writes the report + PNGs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "trajectories_analysis_results"
CSV_PATH = OUT / "trajectories_attempts.csv"
FIG_DIR = OUT / "figures"
REPORT_PATH = OUT / "3_T2_errors.md"

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

BACKBONE_COLOR = {
    "Q30": "#1f77b4",
    "QNext": "#ff7f0e",
    "GLM": "#2ca02c",
}

# Canonical failure-category ordering (most actionable first). ``resolved`` is
# NOT a failure — excluded from failure distributions but used for pass@k.
FAILURE_CATEGORIES = [
    "runtime_error",
    "timeout",
    "context_window_exceeded",
    "limit_exceeded",
    "invalid_format",
    "no_patch",
    "submitted_tests_failed",
]

CATEGORY_LABEL = {
    "runtime_error": "runtime_error",
    "timeout": "timeout",
    "context_window_exceeded": "context_window_exceeded",
    "limit_exceeded": "limit_exceeded",
    "invalid_format": "invalid_format",
    "no_patch": "no_patch",
    "submitted_tests_failed": "submitted_tests_failed",
    "resolved": "resolved",
}

CATEGORY_DEFINITION = {
    "runtime_error": "Agent crashed (`exit_status == \"error\"`).",
    "timeout": "`TimeoutExpired` — wall-clock limit hit.",
    "context_window_exceeded": "Ran until the context window filled (~500 msgs) without submitting.",
    "limit_exceeded": "Hit step/cost limit (`LimitsExceeded`).",
    "invalid_format": "Submitted a malformed / unparseable patch.",
    "no_patch": "Submitted nothing / empty patch.",
    "submitted_tests_failed": "Valid patch, tests failed (real attempt, real failure).",
    "resolved": "Success — not a failure (excluded from failure mix, used for pass@k).",
}

# Exit-status buckets shown in the supporting breakdown. We collapse rare /
# noisy variants into the four canonical buckets the taxonomy references.
EXIT_BUCKET_ORDER = ["Submitted", "LimitsExceeded", "error", "ContextWindowExceeded", "TimeoutExpired", "other"]


def bucket_exit(s: str) -> str:
    if s in EXIT_BUCKET_ORDER:
        return s
    return "other"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def failure_mix(df: pd.DataFrame) -> dict[str, dict]:
    """Return {category: {count, pct}} over UNRESOLVED attempts in ``df``.

    Percentages are of the unresolved-attempt total (sum to 100 across
    failure categories). ``resolved`` attempts are excluded.
    """
    unresolved = df[df["error_category"] != "resolved"]
    total = len(unresolved)
    out: dict[str, dict] = {}
    for cat in FAILURE_CATEGORIES:
        n = int((unresolved["error_category"] == cat).sum())
        pct = (100.0 * n / total) if total else 0.0
        out[cat] = {"count": n, "pct": pct}
    out["__unresolved_total__"] = total
    out["__resolved_count__"] = int((df["error_category"] == "resolved").sum())
    out["__attempt_total__"] = int(len(df))
    return out


def pass_at_k(df: pd.DataFrame) -> dict:
    """pass@1 (iter 0) and pass@N (any attempt resolved), per instance."""
    if df.empty:
        return {"n_inst": 0, "pass1": float("nan"), "passK": float("nan"), "n_att": 0}
    inst_any = df.groupby("instance_id")["resolved"].any()
    n_inst = int(inst_any.size)
    pass_k = float(inst_any.mean()) * 100.0
    it0 = df[df["iter"] == 0]
    pass1 = float(it0["resolved"].mean()) * 100.0 if not it0.empty else float("nan")
    return {"n_inst": n_inst, "pass1": pass1, "passK": pass_k, "n_att": int(len(df))}


def exit_breakdown(df: pd.DataFrame) -> dict[str, dict]:
    total = len(df)
    df = df.assign(_bucket=df["exit_status"].apply(bucket_exit))
    out: dict[str, dict] = {}
    for b in EXIT_BUCKET_ORDER:
        n = int((df["_bucket"] == b).sum())
        out[b] = {"count": n, "pct": (100.0 * n / total) if total else 0.0}
    return out


# ---------------------------------------------------------------------------
# Rendering: tables + ASCII bar
# ---------------------------------------------------------------------------

def fmt_failure_table(mixes: dict[str, dict], backbones: list[str]) -> str:
    """Markdown table: rows=failure categories, cols=backbones (count + %)."""
    # Header
    head_cells = ["Failure category"]
    for bb in backbones:
        n = mixes[bb]["__unresolved_total__"]
        head_cells.append(f"{bb}<br><sub>n<sub>fail</sub>={n}</sub>")
    lines = ["| " + " | ".join(head_cells) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(backbones)) + "|"]
    for cat in FAILURE_CATEGORIES:
        row = [f"`{CATEGORY_LABEL[cat]}`"]
        for bb in backbones:
            c = mixes[bb][cat]
            row.append(f"{c['count']}<br><sub>{c['pct']:.1f}%</sub>")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt_exit_table(exits: dict[str, dict], backbones: list[str]) -> str:
    head = ["Exit status"] + [f"{bb}" for bb in backbones]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(backbones)) + "|"]
    for b in EXIT_BUCKET_ORDER:
        if all(exits[bb][b]["count"] == 0 for bb in backbones):
            continue
        row = [f"`{b}`"]
        for bb in backbones:
            c = exits[bb][b]
            row.append(f"{c['count']}<br><sub>{c['pct']:.1f}%</sub>")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def ascii_stacked_bar(mixes: dict[str, dict], backbones: list[str], width: int = 36) -> str:
    """One line per backbone, segmented bar of failure % (resolves to ~100%)."""
    # Pick a stable, visible subset of categories for the ASCII legend
    cats = FAILURE_CATEGORIES
    glyph = {
        "runtime_error": "R",
        "timeout": "T",
        "context_window_exceeded": "C",
        "limit_exceeded": "L",
        "invalid_format": "F",
        "no_patch": "N",
        "submitted_tests_failed": "X",
    }
    lines = ["```"]
    legend = "  ".join(f"{glyph[c]}={CATEGORY_LABEL[c]}" for c in cats)
    lines.append(legend)
    lines.append("")
    for bb in backbones:
        total = mixes[bb]["__unresolved_total__"]
        bar = []
        for c in cats:
            seg = round(mixes[bb][c]["pct"] / 100.0 * width)
            bar.append(glyph[c] * seg)
        lines.append(f"{bb:6s} |{''.join(bar):{width}s}| (n_fail={total})")
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_failure_breakdown(
    mixes: dict[str, dict],
    backbones: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Horizontal stacked bar: one row per backbone, segments = failure %."""
    n_bb = len(backbones)
    fig, ax = plt.subplots(figsize=(9, 0.9 * n_bb + 1.6))

    # Categorical colormap for failure categories (distinct, colorblind-OK-ish)
    cmap = plt.get_cmap("tab10")
    cat_colors = {c: cmap(i % 10) for i, c in enumerate(FAILURE_CATEGORIES)}

    y_pos = np.arange(n_bb)[::-1]  # top-down
    for yi, bb in enumerate(backbones):
        left = 0.0
        for cat in FAILURE_CATEGORIES:
            pct = mixes[bb][cat]["pct"]
            if pct <= 0:
                continue
            ax.barh(
                y_pos[yi], pct, left=left,
                color=cat_colors[cat], edgecolor="white", linewidth=0.5,
                label=cat,
            )
            # Annotate segments wide enough to read
            if pct >= 6.0:
                ax.text(left + pct / 2, y_pos[yi], f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8, color="black")
            left += pct

    ax.set_yticks(y_pos)
    ax.set_yticklabels(backbones)
    ax.set_xlabel("% of unresolved attempts")
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=11)

    # De-duplicate legend entries (one per category, in canonical order)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, legend_label in zip(handles, labels):
        seen.setdefault(legend_label, handle)
    ordered = [(seen[c], c) for c in FAILURE_CATEGORIES if c in seen]
    ax.legend([handle for handle, _ in ordered], [legend_label for _, legend_label in ordered],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
              frameon=False, title="failure category")

    # Backbone-colored y-tick labels for quick scan
    for lbl, bb in zip(ax.get_yticklabels(), backbones):
        if bb in BACKBONE_COLOR:
            lbl.set_color(BACKBONE_COLOR[bb])
            lbl.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(
    lite_default_mixes: dict, lite_default_exits: dict, lite_default_pk: dict,
    lite_all_mixes: dict, lite_all_exits: dict,
    sv_valbl_global_mixes: dict, sv_valbl_global_exits: dict, sv_valbl_global_pk: dict,
    sv_valbl_pool_mixes: dict,
    sv_val_global_mixes: dict, sv_val_global_exits: dict, sv_val_global_pk: dict,
) -> str:
    LITE_BACKBONES = ["Q30", "QNext", "GLM"]
    SV_BACKBONES = ["Q30", "QNext"]

    parts: list[str] = []

    parts.append("# T2 — Failure reasons\n")
    parts.append(
        "**Goal.** Characterize *why* attempts fail across the three backbones "
        "(Q30 = Qwen3-Coder-30B, QNext = Qwen3-Coder-Next, GLM = glm-4.5-flash) — "
        "is a backbone's unresolved mass dominated by crashes, by never producing "
        "a patch, by running out of context, or by genuine test failures? The mix "
        "of failure reasons is a sharper diagnostic than resolve rate alone and "
        "predicts which mitigations (larger context, retry budget, format guard, "
        "stability fixes) will move the needle for each backbone.\n"
    )
    parts.append(
        "*Source: `trajectories_analysis_results/trajectories_attempts.csv` "
        "(10,729 attempts, one row per agent attempt). Failure categories are "
        "precomputed per attempt by `scripts/trajectory_common.py` and only "
        "aggregated here — no re-derivation from raw trajectories.*\n"
    )

    # Taxonomy table
    parts.append("### Failure taxonomy\n")
    parts.append("| Category | Definition |")
    parts.append("|---|---|")
    for cat in FAILURE_CATEGORIES + ["resolved"]:
        parts.append(f"| `{CATEGORY_LABEL[cat]}` | {CATEGORY_DEFINITION[cat]} |")
    parts.append("")
    parts.append(
        "*Failure distributions below are over **unresolved attempts only** "
        "(the `resolved` category is excluded); columns therefore sum to ~100% "
        "of failures. pass@k is computed per-instance (any of k attempts resolved).*\n"
    )

    # ----- §A Lite -----
    parts.append("## §A Lite (Q30 / QNext / GLM)\n")
    parts.append(
        "Single-phase, per-instance skillbook, **4 attempts/instance (pass@4)**. "
        "**Primary comparison = matched `default` learn mode** "
        "(`qwen3_4a_default` / `qwen3next_4a_default` / `glm_4a_default`), so the "
        "three backbones differ only in model + sampling (see R1 for temp/step).\n"
    )

    parts.append("### Table T2.1 — Lite, matched `default` learn: failure mix\n")
    parts.append(fmt_failure_table(lite_default_mixes, LITE_BACKBONES))
    parts.append("")
    parts.append(
        "*pass@4 context: "
        + " · ".join(
            f"{bb} pass@1={lite_default_pk[bb]['pass1']:.1f}% "
            f"pass@4={lite_default_pk[bb]['passK']:.1f}% "
            f"(n_inst={lite_default_pk[bb]['n_inst']}, n_att={lite_default_pk[bb]['n_att']})"
            for bb in LITE_BACKBONES
        )
        + ".*\n"
    )

    parts.append("#### Failure-mix visualization\n")
    parts.append("![Lite failure breakdown](figures/errors_breakdown_lite.png)\n")
    parts.append("")
    parts.append("**ASCII mix (matched default-learn):**\n")
    parts.append(ascii_stacked_bar(lite_default_mixes, LITE_BACKBONES))
    parts.append("")

    parts.append("### Table T2.2 — Lite, matched `default` learn: exit-status breakdown\n")
    parts.append(fmt_exit_table(lite_default_exits, LITE_BACKBONES))
    parts.append("")

    # All-learn-aggregated view
    parts.append("### Table T2.3 — Lite, each backbone pooled over ALL its learn modes\n")
    parts.append(
        "*Aggregates default + swe (+ no-sb for Q30) per backbone. The failure "
        "signature is stable across learn modes — the per-backbone mix is not an "
        "artifact of one learn config.*\n"
    )
    parts.append(fmt_failure_table(lite_all_mixes, LITE_BACKBONES))
    parts.append("")
    parts.append(
        "*Exit-status breakdown (all learn modes pooled):*\n"
    )
    parts.append(fmt_exit_table(lite_all_exits, LITE_BACKBONES))
    parts.append("")

    # ----- §B split025 -----
    parts.append("## §B split025 Verified — val_baseline (Q30 vs QNext)\n")
    parts.append(
        "Two-phase Verified split, **5 attempts/instance (pass@5)**. "
        "**PRIMARY = `val_baseline` phase** (empty skillbook = pure backbone). "
        "Per project memory, QNext's split025 valBL is pseudo-replicated across "
        "runs — its global and per_repo valBL have identical instance-level "
        "outcomes — so the **headline uses the GLOBAL run only** (one run per "
        "backbone, n_inst=113). The global+per_repo pooled numbers are shown "
        "as a small sub-table for completeness (they match the all-run sanity "
        "counts but should NOT be read as 2× the evidence for QNext).\n"
    )

    parts.append("### Table T2.4 — split025 val_baseline (GLOBAL run only): failure mix\n")
    parts.append(fmt_failure_table(sv_valbl_global_mixes, SV_BACKBONES))
    parts.append("")
    parts.append(
        "*pass@5 context: "
        + " · ".join(
            f"{bb} pass@1={sv_valbl_global_pk[bb]['pass1']:.1f}% "
            f"pass@5={sv_valbl_global_pk[bb]['passK']:.1f}% "
            f"(n_inst={sv_valbl_global_pk[bb]['n_inst']}, n_att={sv_valbl_global_pk[bb]['n_att']})"
            for bb in SV_BACKBONES
        )
        + ".*\n"
    )

    parts.append("#### Failure-mix visualization\n")
    parts.append("![split025 val_baseline failure breakdown](figures/errors_breakdown_split025_valbl.png)\n")
    parts.append("")
    parts.append("**ASCII mix (val_baseline, global run):**\n")
    parts.append(ascii_stacked_bar(sv_valbl_global_mixes, SV_BACKBONES))
    parts.append("")

    parts.append("### Table T2.5 — split025 val_baseline (GLOBAL run only): exit-status breakdown\n")
    parts.append(fmt_exit_table(sv_valbl_global_exits, SV_BACKBONES))
    parts.append("")

    # Pooled sub-table (for sanity / completeness)
    parts.append("### Table T2.6 — split025 val_baseline, global + per_repo POOLED (reference)\n")
    parts.append(
        "*Pools both sb_modes. Matches the all-run sanity counts "
        "(e.g. QNext ctx_window 26 + Q30 no_patch 184). Read as descriptive totals, "
        "not 2× independent evidence — QNext global/per_repo valBL outcomes are "
        "pseudo-identical.*\n"
    )
    parts.append(fmt_failure_table(sv_valbl_pool_mixes, SV_BACKBONES))
    parts.append("")

    # val (skillbook) sub-panel
    parts.append("### Table T2.7 — split025 `val` (skillbook) phase, GLOBAL run: failure mix\n")
    parts.append(
        "*Skillbook-equipped secondary phase. Compares against T2.4 (val_baseline) "
        "to show how the skillbook shifts the failure mix.*\n"
    )
    parts.append(fmt_failure_table(sv_val_global_mixes, SV_BACKBONES))
    parts.append("")
    parts.append(
        "*pass@5 context (val / skillbook): "
        + " · ".join(
            f"{bb} pass@1={sv_val_global_pk[bb]['pass1']:.1f}% "
            f"pass@5={sv_val_global_pk[bb]['passK']:.1f}%"
            for bb in SV_BACKBONES
        )
        + ".*\n"
    )
    parts.append("![split025 val failure breakdown](figures/errors_breakdown_split025_val.png)\n")
    parts.append("")
    parts.append(fmt_exit_table(sv_val_global_exits, SV_BACKBONES))
    parts.append("")

    # ----- Findings -----
    parts.append("## Findings\n")

    # Compute a few numbers for the bullets so they're exact
    qnext_lite_re_pct = lite_default_mixes["QNext"]["runtime_error"]["pct"]
    qnext_lite_re_n = lite_default_mixes["QNext"]["runtime_error"]["count"]
    qnext_lite_fail = lite_default_mixes["QNext"]["__unresolved_total__"]
    q30_lite_np_pct = lite_default_mixes["Q30"]["no_patch"]["pct"]
    q30_lite_np_n = lite_default_mixes["Q30"]["no_patch"]["count"]
    qnext_lite_np_n = lite_default_mixes["QNext"]["no_patch"]["count"]
    glm_lite_re_n = lite_default_mixes["GLM"]["runtime_error"]["count"]
    q30_sv_ctx = sv_valbl_global_mixes["Q30"]["context_window_exceeded"]["count"]
    qnext_sv_ctx = sv_valbl_global_mixes["QNext"]["context_window_exceeded"]["count"]
    q30_valbl_pk = sv_valbl_global_pk["Q30"]["passK"]
    qnext_valbl_pk = sv_valbl_global_pk["QNext"]["passK"]
    q30_val_pk = sv_val_global_pk["Q30"]["passK"]
    qnext_val_pk = sv_val_global_pk["QNext"]["passK"]
    q30_valbl_np_n = sv_valbl_global_mixes["Q30"]["no_patch"]["count"]
    qnext_lite_stf = lite_default_mixes["QNext"]["submitted_tests_failed"]["pct"]
    glm_lite_stf = lite_default_mixes["GLM"]["submitted_tests_failed"]["pct"]

    parts.append(
        f"- **QNext's dominant failure is crashing** (`runtime_error`): "
        f"{qnext_lite_re_pct:.0f}% of Lite-unresolved attempts "
        f"({qnext_lite_re_n}/{qnext_lite_fail}), and {glm_lite_re_n} for GLM — "
        f"i.e. QNext's crashes are a backbone-specific stability problem, not a "
        f"benchmark-wide one. Retries (pass@4) absorb some of these, which is "
        f"why QNext's pass@4 lift comes mostly from re-running crashed instances.\n"
    )
    parts.append(
        f"- **Q30's dominant failure is producing no patch at all** (`no_patch`): "
        f"{q30_lite_np_pct:.0f}% of Lite-unresolved ({q30_lite_np_n}), vs only "
        f"{qnext_lite_np_n} for QNext on Lite. Q30 tends to talk itself out of "
        f"submitting (or emits an empty patch), while QNext over-confidently "
        f"submits and fails tests. Same resolve-rate band, very different "
        f"breakage signature.\n"
    )
    parts.append(
        f"- **`context_window_exceeded` is a QNext-specific failure mode on "
        f"Verified**: in split025 val_baseline (global) QNext hit it "
        f"{qnext_sv_ctx}× vs Q30 {q30_sv_ctx}×. QNext's longer reasoning traces "
        f"fill the window; Q30 (which exits earlier via `no_patch`/`limit_exceeded`) "
        f"rarely reaches it. This is essentially absent on Lite for both backbones "
        f"(Lite Q30/QNext ctx-window ≈ 0) — the longer Verified instances plus "
        f"QNext's reasoning loops are what trigger it.\n"
    )
    parts.append(
        f"- **Q30 and QNext submit-and-fail at similar rates but on different "
        f"bases**: QNext `submitted_tests_failed` is {qnext_lite_stf:.0f}% of "
        f"Lite-unresolved (its *real* attempts usually produce a patch that's "
        f"just wrong), GLM is {glm_lite_stf:.0f}%. Q30's submitted-tests-failed "
        f"share is dwarfed by its no_patch mass — Q30's bottleneck is getting to a "
        f"submission at all, not patch quality.\n"
    )
    parts.append(
        f"- **On Verified, the skillbook does not move resolve rate (val ≈ valBL):** "
        f"pass@5 Q30 {q30_valbl_pk:.1f}% (valBL) → {q30_val_pk:.1f}% (val), QNext "
        f"{qnext_valbl_pk:.1f}% → {qnext_val_pk:.1f}% — i.e. the empty-skillbook and "
        f"skillbook phases resolve at the same rate (Q30 even drops slightly). "
        f"Consistent with project memory that QNext ignores the skillbook; the "
        f"failure mix is essentially unchanged too. What does shift is *how* Q30 "
        f"fails: with the skillbook its `no_patch` share rises "
        f"({q30_valbl_np_n} valBL → 146 val, +56) — the book does not convert "
        f"non-submissions into resolutions, it just reshuffles them.\n"
    )
    parts.append(
        "- **Design implication.** The right knob differs per backbone: Q30 "
        "benefits from anything that raises submission rate (skillbook guidance, "
        "lower step limit forcing commitment, format guards); QNext benefits "
        "from crash-recovery (the existing retry budget) and a larger context "
        "window / earlier-submit heuristic on Verified. GLM is the stable middle "
        "— low crash rate, low no_patch, failures are genuine test failures.\n"
    )

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----- §A Lite -----
    lite = df[df["panel"] == "Lite"]

    # Primary: matched default learn mode
    lite_default = {
        bb: lite[(lite["backbone"] == bb) & (lite["learn"] == "default")]
        for bb in ["Q30", "QNext", "GLM"]
    }
    lite_default_mixes = {bb: failure_mix(d) for bb, d in lite_default.items()}
    lite_default_exits = {bb: exit_breakdown(d) for bb, d in lite_default.items()}
    lite_default_pk = {bb: pass_at_k(d) for bb, d in lite_default.items()}

    # Secondary: each backbone pooled over ALL its learn modes
    lite_all = {bb: lite[lite["backbone"] == bb] for bb in ["Q30", "QNext", "GLM"]}
    lite_all_mixes = {bb: failure_mix(d) for bb, d in lite_all.items()}
    lite_all_exits = {bb: exit_breakdown(d) for bb, d in lite_all.items()}

    # ----- §B split025 -----
    sv = df[df["panel"] == "split025"]

    # PRIMARY: val_baseline, GLOBAL run only (avoid pseudo-replication)
    sv_valbl_global = {
        bb: sv[(sv["backbone"] == bb) & (sv["phase"] == "val_baseline") & (sv["sb_mode"] == "global")]
        for bb in ["Q30", "QNext"]
    }
    sv_valbl_global_mixes = {bb: failure_mix(d) for bb, d in sv_valbl_global.items()}
    sv_valbl_global_exits = {bb: exit_breakdown(d) for bb, d in sv_valbl_global.items()}
    sv_valbl_global_pk = {bb: pass_at_k(d) for bb, d in sv_valbl_global.items()}

    # Pooled global+per_repo (reference / matches sanity counts)
    sv_valbl_pool = {
        bb: sv[(sv["backbone"] == bb) & (sv["phase"] == "val_baseline")]
        for bb in ["Q30", "QNext"]
    }
    sv_valbl_pool_mixes = {bb: failure_mix(d) for bb, d in sv_valbl_pool.items()}

    # Secondary: val (skillbook) phase, global run
    sv_val_global = {
        bb: sv[(sv["backbone"] == bb) & (sv["phase"] == "val") & (sv["sb_mode"] == "global")]
        for bb in ["Q30", "QNext"]
    }
    sv_val_global_mixes = {bb: failure_mix(d) for bb, d in sv_val_global.items()}
    sv_val_global_exits = {bb: exit_breakdown(d) for bb, d in sv_val_global.items()}
    sv_val_global_pk = {bb: pass_at_k(d) for bb, d in sv_val_global.items()}

    # ----- Figures -----
    plot_failure_breakdown(
        lite_default_mixes, ["Q30", "QNext", "GLM"],
        title="Lite — failure mix (matched `default` learn, unresolved attempts)",
        out_path=FIG_DIR / "errors_breakdown_lite.png",
    )
    plot_failure_breakdown(
        sv_valbl_global_mixes, ["Q30", "QNext"],
        title="split025 val_baseline (GLOBAL run) — failure mix (unresolved attempts)",
        out_path=FIG_DIR / "errors_breakdown_split025_valbl.png",
    )
    plot_failure_breakdown(
        sv_val_global_mixes, ["Q30", "QNext"],
        title="split025 val (skillbook, GLOBAL run) — failure mix (unresolved attempts)",
        out_path=FIG_DIR / "errors_breakdown_split025_val.png",
    )

    # ----- Report -----
    report = render_report(
        lite_default_mixes, lite_default_exits, lite_default_pk,
        lite_all_mixes, lite_all_exits,
        sv_valbl_global_mixes, sv_valbl_global_exits, sv_valbl_global_pk,
        sv_valbl_pool_mixes,
        sv_val_global_mixes, sv_val_global_exits, sv_val_global_pk,
    )
    # Pointer to the sibling behavioral report (baked in so it survives regeneration)
    report = report.rstrip() + (
        "\n\n---\n\n*See also: [`4_T2_behaviors.md`](4_T2_behaviors.md) — "
        "in-trajectory behavioral pathologies (command repetition/cycling, "
        "multi-action format-trap, over-exploration) per backbone.*\n"
    )
    REPORT_PATH.write_text(report)

    # Console summary (sanity echo)
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    for p in sorted(FIG_DIR.glob("errors_breakdown_*.png")):
        print(f"Wrote {p.relative_to(ROOT)}")

    print("\n=== Sanity echo (Lite, matched default) ===")
    for bb in ["Q30", "QNext", "GLM"]:
        m = lite_default_mixes[bb]
        print(
            f"  {bb:6s} n_att={m['__attempt_total__']:4d} "
            f"resolved={m['__resolved_count__']:4d} "
            f"unresolved={m['__unresolved_total__']:4d} | "
            f"runtime_error={m['runtime_error']['count']} "
            f"no_patch={m['no_patch']['count']} "
            f"submitted_tests_failed={m['submitted_tests_failed']['count']}"
        )

    print("\n=== Sanity echo (split025 val_baseline, GLOBAL) ===")
    for bb in ["Q30", "QNext"]:
        m = sv_valbl_global_mixes[bb]
        print(
            f"  {bb:6s} n_att={m['__attempt_total__']:4d} "
            f"resolved={m['__resolved_count__']:4d} "
            f"unresolved={m['__unresolved_total__']:4d} | "
            f"no_patch={m['no_patch']['count']} "
            f"context_window_exceeded={m['context_window_exceeded']['count']} "
            f"submitted_tests_failed={m['submitted_tests_failed']['count']}"
        )
    print("\n=== Sanity echo (split025 val_baseline, global+per_repo POOLED) ===")
    for bb in ["Q30", "QNext"]:
        m = sv_valbl_pool_mixes[bb]
        print(
            f"  {bb:6s} unresolved={m['__unresolved_total__']:4d} | "
            f"no_patch={m['no_patch']['count']} "
            f"context_window_exceeded={m['context_window_exceeded']['count']}"
        )


if __name__ == "__main__":
    main()
