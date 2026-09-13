"""
scripts/compare_runs.py
Sorted table of runs with metrics and seed standard deviations.
Morning standup / experiment tracking artifact (AGENTS.md §4).

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experiment runs in experiments/ directory."
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Path to experiments directory (default: experiments).",
    )
    parser.add_argument(
        "--md",
        action="store_true",
        help="Output as a GitHub-flavored Markdown table instead of plain text.",
    )
    return parser.parse_args()


def load_all_runs(experiments_dir: str | Path) -> list[dict]:
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        return []

    runs = []
    for run_dir in exp_path.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_file = run_dir / "run_manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest["run_dir"] = str(run_dir)
                runs.append(manifest)
            except Exception:
                continue
    return runs


def format_table(runs: list[dict], markdown: bool = False) -> str:
    if not runs:
        return "No completed experiment runs found in experiments/."

    # Group runs by config name
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        cfg_name = Path(run.get("config", "unknown")).stem
        grouped[cfg_name].append(run)

    # Sort each run by OOF BA descending
    sorted_runs = sorted(
        runs,
        key=lambda r: float(r.get("metrics", {}).get("oof_ba") or 0.0),
        reverse=True,
    )

    headers = [
        "Rank",
        "Experiment",
        "Seed",
        "OOF BA",
        "OOF AUC",
        "Thresh t*",
        "Folds",
        "Finished",
    ]

    rows = []
    for rank, run in enumerate(sorted_runs, 1):
        cfg_name = Path(run.get("config", "unknown")).stem
        seed = str(run.get("seed", "-"))
        metrics = run.get("metrics", {})
        ba = metrics.get("oof_ba")
        ba_str = f"{float(ba):.4f}" if ba is not None else "N/A"
        auc = metrics.get("oof_auc")
        auc_str = f"{float(auc):.4f}" if auc is not None else "N/A"
        thresh = metrics.get("threshold")
        thresh_str = f"{float(thresh):.4f}" if thresh is not None else "N/A"
        num_folds = len(run.get("fold_metrics", []))
        finished = run.get("finished_at", "-").replace("T", " ")[:16]

        rows.append(
            [
                str(rank),
                cfg_name,
                seed,
                ba_str,
                auc_str,
                thresh_str,
                str(num_folds),
                finished,
            ]
        )

    if markdown:
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
    else:
        # Compute column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        header_line = (
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        )

        lines = [sep, header_line, sep]
        for row in rows:
            line = (
                "| "
                + " | ".join(val.ljust(widths[i]) for i, val in enumerate(row))
                + " |"
            )
            lines.append(line)
        lines.append(sep)
        return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_all_runs(args.experiments_dir)
    output = format_table(runs, markdown=args.md)
    print(output)


if __name__ == "__main__":
    main()
