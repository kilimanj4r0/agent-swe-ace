"""In-trajectory behavioral pathology extractor.

Reads the raw message sequence of each attempt and computes deterministic
behavioral metrics — command repetition/cycling, format-trap (multi-action)
rejections, exploration-vs-action, and stuck duplicate-message loops. Emits one
row per attempt to ``trajectories_analysis_results/trajectories_behaviors.csv``,
joinable to ``trajectories_attempts.csv`` on (panel, run, phase, instance_id, iter).

Reuses the run registry + helpers from ``trajectory_common`` so the analyzed
scope is identical to T1/T2.

Run:  uv run python scripts/trajectory_behaviors.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import trajectory_common as tc  # noqa: E402

OUT = tc.ROOT / "trajectories_analysis_results"
CSV_PATH = OUT / "trajectories_behaviors.csv"

# --- message parsing --------------------------------------------------------
BLOCK_RE = re.compile(r"```(\w*)\n(.*?)\n```", re.S)
ONE_ACTION_RE = re.compile(r"EXACTLY ONE action", re.I)
SUBMIT_RE = re.compile(r"COMPLETE_TASK_AND_SUBMIT|/submit|SUBMIT_PATCH", re.I)
EXPLORE_RE = re.compile(
    r"^\s*(cat|ls|ll|grep|rg|ag|find|head|tail|sed\s+-n|awk|pwd|wc|less|more|"
    r"tree|file|stat|du|df|git\s+(log|status|diff|show|blame|ls-files|rev-parse)|"
    r"echo|which|type|man|history|cd)\b",
    re.I,
)
EDIT_RE = re.compile(
    r"(sed\s+-i|git\s+(apply|checkout|restore|add|commit|stash|reset)|"
    r"\bpatch\b|cat\s*>|>>|\btee\b|\bcp\s|\bmv\s|\brm\s|\bmkdir\b|\btouch\b|"
    r"python.*open\([^)]*,\s*['\"][wa])",
    re.I,
)


def normalize_cmd(c: str) -> str:
    return re.sub(r"\s+", " ", c.strip())


def blocks(content: str) -> list[str]:
    return [m[1] for m in BLOCK_RE.findall(content or "")]


def classify_cmd(cmd: str) -> str:
    if SUBMIT_RE.search(cmd):
        return "submit"
    if EDIT_RE.search(cmd):
        return "edit"
    if EXPLORE_RE.search(cmd):
        return "explore"
    return "other"


def detect_cycle(cmds: list[str]) -> int:
    """Smallest period L in {2,3,4} of an immediately-repeated window, else 0."""
    n = len(cmds)
    for L in (2, 3, 4):
        for i in range(n - 2 * L + 1):
            if cmds[i : i + L] == cmds[i + L : i + 2 * L]:
                return L
    return 0


def analyze_messages(messages: list[dict]) -> dict:
    asst = [m for m in messages if m.get("role") == "assistant"]
    user = [m for m in messages if m.get("role") == "user"]

    # executed command = last code block of each assistant message that has one
    cmds_raw = []
    for m in asst:
        bs = blocks(m.get("content") or "")
        if bs:
            cmds_raw.append(bs[-1])
    cmds = [normalize_cmd(c) for c in cmds_raw]
    n_cmds = len(cmds)

    freq = Counter(cmds)
    n_unique = len(freq)
    n_repeated = sum(1 for _, c in freq.items() if c >= 2)

    max_consec = run = 1
    n_consec_pairs = 0
    for i in range(1, n_cmds):
        if cmds[i] == cmds[i - 1]:
            run += 1
            n_consec_pairs += 1
            max_consec = max(max_consec, run)
        else:
            run = 1
    if n_cmds == 0:
        max_consec = 0

    cycle_len = detect_cycle(cmds)

    kinds = [classify_cmd(c) for c in cmds]
    n_explore = sum(1 for k in kinds if k == "explore")
    n_edit = sum(1 for k in kinds if k == "edit")
    n_submit = sum(1 for k in kinds if k == "submit")

    n_one_action = sum(1 for m in user if ONE_ACTION_RE.search(m.get("content") or ""))

    # consecutive identical assistant messages (full content) -> pure stuck loop
    asst_contents = [m.get("content") or "" for m in asst]
    n_dup_asst = 0
    for i in range(1, len(asst_contents)):
        if asst_contents[i] == asst_contents[i - 1]:
            n_dup_asst += 1

    top_cmd = ""
    if freq:
        top_cmd = freq.most_common(1)[0][0][:120]

    return {
        "n_msgs": len(messages),
        "n_asst_msgs": len(asst),
        "n_cmds": n_cmds,
        "n_unique_cmds": n_unique,
        "cmd_repeat_ratio": round(1 - n_unique / n_cmds, 4) if n_cmds else 0.0,
        "n_repeated_cmds": n_repeated,
        "max_consec_repeat": max_consec,
        "n_consec_dup_pairs": n_consec_pairs,
        "has_cycle": int(cycle_len > 0),
        "cycle_len": cycle_len,
        "explore_ratio": round(n_explore / n_cmds, 4) if n_cmds else 0.0,
        "n_edits": n_edit,
        "n_submit_cmds": n_submit,
        "n_one_action_warnings": n_one_action,
        "n_dup_asst_msgs": n_dup_asst,
        "top_repeated_cmd": top_cmd,
    }


FIELDS = [
    "panel", "backbone", "run", "learn", "phase", "has_skillbook", "sb_mode",
    "instance_id", "iter", "exit_status", "resolved", "error_category",
    "n_msgs", "n_asst_msgs", "n_cmds", "n_unique_cmds", "cmd_repeat_ratio",
    "n_repeated_cmds", "max_consec_repeat", "n_consec_dup_pairs", "has_cycle",
    "cycle_len", "explore_ratio", "n_edits", "n_submit_cmds",
    "n_one_action_warnings", "n_dup_asst_msgs", "top_repeated_cmd",
]


def extract() -> list[dict]:
    rows = []
    for panel, run_name, backbone, learn, phase, has_sb, sb_mode in tc.EXTRACTIONS:
        run_dir = tc.DATA / run_name
        if not run_dir.exists():
            continue
        bench = tc.find_benchmark_dir(run_dir)
        if bench is None:
            continue
        traj_glob = bench / "trajectories" / (phase or "")
        if not traj_glob.exists():
            continue
        for traj in sorted(traj_glob.glob("*/iter_*.json")):
            inst = traj.parent.name
            try:
                j = json.loads(traj.read_text())
            except Exception:
                continue
            info = j.get("info") or {}
            messages = j.get("messages") or []
            result_path = bench / "results" / (phase or "") / inst / traj.name
            if phase is None:
                result_path = bench / "results" / inst / traj.name
            r = tc.load_result(result_path)
            resolved = r["resolved"] if r else None
            cat = tc.classify_error(
                info.get("exit_status"), resolved,
                r["patch_empty"] if r else None,
                r["patch_invalid_format"] if r else None,
                bool(info.get("submission") or ""),
            )
            b = analyze_messages(messages)
            rows.append({
                "panel": panel, "backbone": backbone, "run": run_name, "learn": learn,
                "phase": phase or "", "has_skillbook": has_sb, "sb_mode": sb_mode,
                "instance_id": inst, "iter": int(info.get("iteration", traj.stem.split("_")[-1]) or 0),
                "exit_status": info.get("exit_status") or "",
                "resolved": "" if resolved is None else resolved, "error_category": cat,
                **b,
            })
    return rows


def summarize(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["panel"], r["backbone"], r["phase"]), []).append(r)
    print(f"\nTotal attempts: {len(rows)}")
    hdr = f"{'panel/backbone/phase':40s} {'n':>5s} {'repR':>5s} {'cyc%':>5s} {'maxRun':>7s} {'1act%':>6s} {'1act/run':>9s} {'dupAsst':>7s} {'expl':>5s}"
    print(hdr)
    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        rep = median(r["cmd_repeat_ratio"] for r in rs)
        cyc = 100 * sum(1 for r in rs if r["has_cycle"]) / n
        maxrun = median(r["max_consec_repeat"] for r in rs)
        warn = 100 * sum(1 for r in rs if r["n_one_action_warnings"] > 0) / n
        warnrate = median(r["n_one_action_warnings"] for r in rs)
        dup = median(r["n_dup_asst_msgs"] for r in rs)
        expl = median(r["explore_ratio"] for r in rs)
        label = f"{key[0]}/{key[1]}/{key[2] or '-'}"
        print(f"{label:40s} {n:5d} {rep:5.2f} {cyc:4.0f}% {maxrun:7.0f} {warn:5.0f}% {warnrate:9.0f} {dup:7.0f} {expl:5.2f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = extract()
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {CSV_PATH.relative_to(ROOT)}")
    summarize(rows)


if __name__ == "__main__":
    main()
