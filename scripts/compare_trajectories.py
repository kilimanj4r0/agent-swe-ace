#!/usr/bin/env python3
"""Compare trajectories across runs with structural, behavioral, and outcome classification.

Usage:
    uv run python scripts/compare_trajectories.py data/run_A data/run_B [data/run_C ...]
    uv run python scripts/compare_trajectories.py data/run_A data/run_B --iter 0
    uv run python scripts/compare_trajectories.py data/run_A data/run_B --detail <instance_id>
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _find_benchmark_dir(run_dir: Path) -> Path | None:
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def extract_bash_commands(messages: list[dict]) -> list[str]:
    """Extract bash commands from assistant messages."""
    commands = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        # Find bash code blocks
        blocks = re.findall(r"```bash\n(.*?)```", content, re.DOTALL)
        for block in blocks:
            cmd = block.strip()
            if cmd:
                commands.append(cmd)
    return commands


def classify_command(cmd: str) -> str:
    """Classify a bash command into a category."""
    cmd_lower = cmd.strip().lower()
    # Order matters: more specific first
    if cmd_lower.startswith("git diff") or cmd_lower.startswith("git add") or "git commit" in cmd_lower:
        return "git_ops"
    if "submit" in cmd_lower or "complete_task" in cmd_lower:
        return "submit"
    if cmd_lower.startswith("python") or cmd_lower.startswith("python3"):
        if "test" in cmd_lower or "pytest" in cmd_lower:
            return "test_run"
        if "reproduce" in cmd_lower or "script" in cmd_lower:
            return "reproduce"
        return "python_run"
    if "pytest" in cmd_lower:
        return "test_run"
    if cmd_lower.startswith("cat ") or cmd_lower.startswith("head ") or cmd_lower.startswith("tail "):
        return "file_read"
    if cmd_lower.startswith("sed ") or cmd_lower.startswith("awk "):
        return "file_edit"
    if cmd_lower.startswith("find "):
        return "explore"
    if cmd_lower.startswith("grep ") or cmd_lower.startswith("rg "):
        return "search"
    if "cat <<'" in cmd_lower or "cat <<'" in cmd:
        return "file_create"
    if cmd_lower.startswith("ls ") or cmd_lower == "ls":
        return "explore"
    if cmd_lower.startswith("cd "):
        return "nav"
    if cmd_lower.startswith("nl ") or "| sed -n" in cmd_lower:
        return "file_read"
    if cmd_lower.startswith("echo "):
        return "file_create"
    return "other"


def extract_assistant_content(messages: list[dict]) -> list[str]:
    """Extract assistant message contents for similarity."""
    return [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str)
    ]


def classify_exit(submission: str | None, exit_status: str) -> str:
    """Classify the trajectory outcome."""
    if exit_status == "error":
        return "error"
    if exit_status == "LimitsExceeded":
        return "limits_exceeded"
    if exit_status == "Submitted":
        if not submission or submission.strip() in (
            "",
            "On branch main\nnothing to commit, working tree clean",
        ):
            return "empty_patch"
        if submission.startswith("diff --git"):
            return "real_patch"
        return "other_patch"
    return exit_status


def load_trajectory(run_dir: Path, instance: str, iteration: int = 0) -> dict | None:
    """Load a single trajectory and extract features."""
    bench = _find_benchmark_dir(run_dir)
    if bench is None:
        return None
    traj_path = bench / "trajectories" / instance / f"iter_{iteration}.json"
    result_path = bench / "results" / instance / f"iter_{iteration}.json"
    if not traj_path.exists():
        return None

    with open(traj_path) as f:
        d = json.load(f)

    info = d.get("info", {})
    messages = d.get("messages", [])

    # Basic structure
    msg_count = info.get("message_count", len(messages))
    asst_count = info.get("assistant_message_count", sum(1 for m in messages if m.get("role") == "assistant"))
    exit_status = info.get("exit_status", "unknown")
    submission = info.get("submission", "")
    model = info.get("model", info.get("config", {}).get("model", {}).get("model_name", "unknown"))

    # Commands and actions
    commands = extract_bash_commands(messages)
    cmd_categories = Counter(classify_command(c) for c in commands)

    # Outcome
    patch_class = classify_exit(submission, exit_status)
    resolved = None
    if result_path.exists():
        with open(result_path) as f:
            r = json.load(f)
        resolved = r.get("resolved")

    return {
        "instance": instance,
        "msg_count": msg_count,
        "asst_count": asst_count,
        "exit_status": exit_status,
        "patch_class": patch_class,
        "resolved": resolved,
        "model": model,
        "submission": submission,
        "commands": commands,
        "cmd_count": len(commands),
        "cmd_categories": cmd_categories,
        "assistant_contents": extract_assistant_content(messages),
    }


def load_all_trajectories(run_dir: Path, iteration: int = 0) -> dict[str, dict]:
    """Load all trajectories for a run."""
    bench = _find_benchmark_dir(run_dir)
    if bench is None:
        return {}
    traj_dir = bench / "trajectories"
    if not traj_dir.exists():
        return {}

    result = {}
    for inst_dir in sorted(traj_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        t = load_trajectory(run_dir, inst_dir.name, iteration)
        if t is not None:
            result[inst_dir.name] = t
    return result


def run_label(run_dir: Path) -> str:
    """Short label for a run directory."""
    name = run_dir.name
    # Remove run_ prefix and common suffixes
    name = re.sub(r"^run_", "", name)
    return name


def print_structural_summary(runs_data: dict[str, dict[str, dict]]):
    """Print structural summary for each run."""
    labels = list(runs_data.keys())

    print("=" * 90)
    print("TRAJECTORY STRUCTURAL SUMMARY (iter_0)")
    print("=" * 90)

    header = f"{'Metric':<30s}"
    for label in labels:
        header += f" {label:>18s}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Total instances", lambda trajs: str(len(trajs))),
        ("Resolved", lambda trajs: str(sum(1 for t in trajs.values() if t["resolved"]))),
        ("Unresolved", lambda trajs: str(sum(1 for t in trajs.values() if t["resolved"] is False))),
        ("Avg msg count", lambda trajs: f"{sum(t['msg_count'] for t in trajs.values()) / len(trajs):.1f}"),
        ("Avg asst msgs", lambda trajs: f"{sum(t['asst_count'] for t in trajs.values()) / len(trajs):.1f}"),
        ("Avg cmd count", lambda trajs: f"{sum(t['cmd_count'] for t in trajs.values()) / len(trajs):.1f}"),
    ]

    # Exit status distribution
    exit_statuses = set()
    for trajs in runs_data.values():
        for t in trajs.values():
            exit_statuses.add(t["exit_status"])

    for status in sorted(exit_statuses):
        metrics.append(
            (f"  exit={status}", lambda trajs, s=status: str(sum(1 for t in trajs.values() if t["exit_status"] == s)))
        )

    # Patch class distribution
    patch_classes = set()
    for trajs in runs_data.values():
        for t in trajs.values():
            patch_classes.add(t["patch_class"])

    for pc in sorted(patch_classes):
        metrics.append(
            (f"  patch={pc}", lambda trajs, p=pc: str(sum(1 for t in trajs.values() if t["patch_class"] == p)))
        )

    for metric_name, metric_fn in metrics:
        row = f"{metric_name:<30s}"
        for label in labels:
            trajs = runs_data[label]
            val = metric_fn(trajs)
            row += f" {val:>18s}"
        print(row)


def print_action_profile(runs_data: dict[str, dict[str, dict]]):
    """Print action profile (command category distribution) for each run."""
    labels = list(runs_data.keys())

    print("\n" + "=" * 90)
    print("ACTION PROFILE (avg commands per instance by category)")
    print("=" * 90)

    all_cats = set()
    for trajs in runs_data.values():
        for t in trajs.values():
            all_cats.update(t["cmd_categories"].keys())

    header = f"{'Category':<20s}"
    for label in labels:
        header += f" {label:>18s}"
    print(header)
    print("-" * len(header))

    for cat in sorted(all_cats):
        row = f"{cat:<20s}"
        for label in labels:
            trajs = runs_data[label]
            total = sum(t["cmd_categories"].get(cat, 0) for t in trajs.values())
            avg = total / len(trajs) if trajs else 0
            row += f" {avg:>18.2f}"
        print(row)


def print_trajectory_comparison(runs_data: dict[str, dict[str, dict]]):
    """Per-instance trajectory comparison across runs."""
    labels = list(runs_data.keys())

    # Find common instances
    all_instances = set()
    for trajs in runs_data.values():
        all_instances.update(trajs.keys())

    print("\n" + "=" * 90)
    print("PER-INSTANCE TRAJECTORY CLASSIFICATION")
    print("=" * 90)

    # Cross-tabulation: how often do runs agree on patch_class and resolved?
    if len(labels) == 2:
        a, b = labels
        a_trajs, b_trajs = runs_data[a], runs_data[b]

        # Patch class agreement
        common = sorted(set(a_trajs.keys()) & set(b_trajs.keys()))
        patch_agree = sum(1 for i in common if a_trajs[i]["patch_class"] == b_trajs[i]["patch_class"])
        res_agree = sum(
            1 for i in common if a_trajs[i]["resolved"] == b_trajs[i]["resolved"]
        )

        print(f"\n  Common instances: {len(common)}")
        print(f"  Patch class agreement: {patch_agree}/{len(common)} ({100 * patch_agree / len(common):.1f}%)")
        print(f"  Resolved agreement:    {res_agree}/{len(common)} ({100 * res_agree / len(common):.1f}%)")

        # Show disagreement instances
        patch_disagree = [i for i in common if a_trajs[i]["patch_class"] != b_trajs[i]["patch_class"]]
        res_disagree = [i for i in common if a_trajs[i]["resolved"] != b_trajs[i]["resolved"]]

        if patch_disagree:
            print(f"\n  Patch class disagreements ({len(patch_disagree)}):")
            print(f"    {'Instance':<45s} {a:>20s} {b:>20s}")
            for inst in patch_disagree:
                print(
                    f"    {inst:<45s} {a_trajs[inst]['patch_class']:>20s} {b_trajs[inst]['patch_class']:>20s}"
                )

        if res_disagree:
            print(f"\n  Resolution disagreements ({len(res_disagree)}):")
            print(f"    {'Instance':<45s} {a:>20s} {b:>20s}")
            for inst in res_disagree:
                ra = "resolved" if a_trajs[inst]["resolved"] else "unresolved"
                rb = "resolved" if b_trajs[inst]["resolved"] else "unresolved"
                print(f"    {inst:<45s} {ra:>20s} {rb:>20s}")

        # Behavioral similarity: correlation of message counts and command counts
        msg_corr = _simple_correlation(
            [(a_trajs[i]["msg_count"], b_trajs[i]["msg_count"]) for i in common]
        )
        cmd_corr = _simple_correlation(
            [(a_trajs[i]["cmd_count"], b_trajs[i]["cmd_count"]) for i in common]
        )
        print(f"\n  Message count correlation:   {msg_corr:.3f}")
        print(f"  Command count correlation:   {cmd_corr:.3f}")

        # Structural distance: how different are the trajectories?
        msg_diffs = [(a_trajs[i]["msg_count"] - b_trajs[i]["msg_count"]) for i in common]
        cmd_diffs = [(a_trajs[i]["cmd_count"] - b_trajs[i]["cmd_count"]) for i in common]
        print(f"  Avg msg count diff:          {sum(msg_diffs) / len(msg_diffs):.1f}")
        print(f"  Avg cmd count diff:          {sum(cmd_diffs) / len(cmd_diffs):.1f}")

        # Largest structural outliers
        print("\n  Top structural outliers (by msg count diff):")
        diffs = [(abs(a_trajs[i]["msg_count"] - b_trajs[i]["msg_count"]), i) for i in common]
        diffs.sort(reverse=True)
        print(f"    {'Instance':<45s} {'MsgDiff':>8s} {'CmdDiff':>8s} {'PatchA':>14s} {'PatchB':>14s} {'ResA':>8s} {'ResB':>8s}")
        for _, inst in diffs[:20]:
            md = a_trajs[inst]["msg_count"] - b_trajs[inst]["msg_count"]
            cd = a_trajs[inst]["cmd_count"] - b_trajs[inst]["cmd_count"]
            print(
                f"    {inst:<45s} {md:>+8d} {cd:>+8d} {a_trajs[inst]['patch_class']:>14s} {b_trajs[inst]['patch_class']:>14s} "
                f"{str(a_trajs[inst]['resolved']):>8s} {str(b_trajs[inst]['resolved']):>8s}"
            )

    else:
        # Multi-run comparison: show classification table
        print(f"\n  {'Instance':<45s}", end="")
        for label in labels:
            print(f" {label[-18:]:>18s}", end="")
        print()
        for inst in sorted(all_instances):
            row = f"  {inst:<45s}"
            for label in labels:
                t = runs_data[label].get(inst)
                if t:
                    mark = "R" if t["resolved"] else "." if t["resolved"] is False else "?"
                    row += f" {mark:>18s}"
                else:
                    row += f" {'--':>18s}"
            print(row)


def print_pairwise_matrix(runs_data: dict[str, dict[str, dict]]):
    """Print pairwise agreement matrix across all runs."""
    labels = list(runs_data.keys())
    if len(labels) < 2:
        return

    print("\n" + "=" * 90)
    print("PAIRWISE AGREEMENT MATRIX")
    print("=" * 90)

    # Resolution agreement
    print("\n  Resolution agreement (% of common instances with same resolved status):")
    header = f"  {'':>20s}"
    for l in labels:
        header += f" {l[-14:]:>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for a in labels:
        row = f"  {a[-20:]:>20s}"
        for b in labels:
            if a == b:
                row += f" {'--':>14s}"
                continue
            common = set(runs_data[a].keys()) & set(runs_data[b].keys())
            if not common:
                row += f" {'N/A':>14s}"
                continue
            agree = sum(
                1
                for i in common
                if runs_data[a][i]["resolved"] == runs_data[b][i]["resolved"]
            )
            pct = 100 * agree / len(common)
            row += f" {pct:>13.1f}%"
        print(row)

    # Patch class agreement
    print("\n  Patch class agreement (% of common instances with same patch_class):")
    header = f"  {'':>20s}"
    for l in labels:
        header += f" {l[-14:]:>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for a in labels:
        row = f"  {a[-20:]:>20s}"
        for b in labels:
            if a == b:
                row += f" {'--':>14s}"
                continue
            common = set(runs_data[a].keys()) & set(runs_data[b].keys())
            if not common:
                row += f" {'N/A':>14s}"
                continue
            agree = sum(
                1
                for i in common
                if runs_data[a][i]["patch_class"] == runs_data[b][i]["patch_class"]
            )
            pct = 100 * agree / len(common)
            row += f" {pct:>13.1f}%"
        print(row)

    # Avg msg count difference (symmetric)
    print("\n  Avg |msg_count diff| (lower = more similar trajectories):")
    header = f"  {'':>20s}"
    for l in labels:
        header += f" {l[-14:]:>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for a in labels:
        row = f"  {a[-20:]:>20s}"
        for b in labels:
            if a == b:
                row += f" {'--':>14s}"
                continue
            common = set(runs_data[a].keys()) & set(runs_data[b].keys())
            if not common:
                row += f" {'N/A':>14s}"
                continue
            avg_diff = sum(
                abs(runs_data[a][i]["msg_count"] - runs_data[b][i]["msg_count"])
                for i in common
            ) / len(common)
            row += f" {avg_diff:>14.1f}"
        print(row)


def print_detail(runs_data: dict[str, dict[str, dict]], instance: str):
    """Print detailed trajectory info for a specific instance."""
    labels = list(runs_data.keys())

    print(f"\n{'=' * 90}")
    print(f"DETAIL: {instance}")
    print(f"{'=' * 90}")

    for label in labels:
        t = runs_data[label].get(instance)
        if t is None:
            print(f"\n  [{label}] NOT FOUND")
            continue

        print(f"\n  [{label}]")
        print(f"    exit_status:    {t['exit_status']}")
        print(f"    patch_class:    {t['patch_class']}")
        print(f"    resolved:       {t['resolved']}")
        print(f"    msg_count:      {t['msg_count']}")
        print(f"    asst_count:     {t['asst_count']}")
        print(f"    cmd_count:      {t['cmd_count']}")

        if t["cmd_categories"]:
            print(f"    cmd categories: {dict(t['cmd_categories'])}")

        if t["submission"]:
            sub = t["submission"][:300]
            print(f"    submission:     {repr(sub)}")

        if t["commands"]:
            print(f"    commands ({len(t['commands'])}):")
            for ci, cmd in enumerate(t["commands"]):
                cat = classify_command(cmd)
                cmd_short = cmd[:80].replace("\n", "\\n")
                print(f"      [{ci:2d}] ({cat:12s}) {cmd_short}")


def _simple_correlation(pairs: list[tuple[float, float]]) -> float:
    """Pearson correlation coefficient."""
    n = len(pairs)
    if n < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="Compare trajectories across runs")
    parser.add_argument("runs", nargs="+", metavar="RUN_DIR", help="Run directories to compare")
    parser.add_argument("--iter", type=int, default=0, help="Iteration to compare (default: 0)")
    parser.add_argument("--detail", type=str, default=None, help="Show full detail for a specific instance")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    for rd in run_dirs:
        if not rd.exists():
            print(f"Path not found: {rd}", file=sys.stderr)
            sys.exit(1)

    runs_data = {}
    for rd in run_dirs:
        label = run_label(rd)
        trajs = load_all_trajectories(rd, iteration=args.iter)
        runs_data[label] = trajs
        print(f"Loaded {len(trajs)} trajectories from {label}")

    if not any(runs_data.values()):
        print("No trajectories found.", file=sys.stderr)
        sys.exit(1)

    if args.detail:
        print_detail(runs_data, args.detail)
        return

    print_structural_summary(runs_data)
    print_action_profile(runs_data)
    print_pairwise_matrix(runs_data)
    print_trajectory_comparison(runs_data)


if __name__ == "__main__":
    main()
