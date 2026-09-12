"""
Standup artifact: Compare all experiment runs in experiments/
Parses run_manifest.json files, computes cross-seed means and stds,
and prints a ranked leaderboard sorted by OOF Balanced Accuracy.
"""

import os
import glob
import json
from collections import defaultdict
import numpy as np


def find_manifests(exp_dir: str = "experiments"):
    manifests = []
    for root, _, files in os.walk(exp_dir):
        if "run_manifest.json" in files:
            m_path = os.path.join(root, "run_manifest.json")
            try:
                with open(m_path, "r") as f:
                    data = json.load(f)
                    manifests.append((m_path, data))
            except Exception as e:
                print(f"Warning: Failed to parse {m_path}: {e}")
    return manifests


def main():
    manifests = find_manifests()
    if not manifests:
        print("No completed experiment manifests found in experiments/")
        return

    runs = []
    grouped_by_exp = defaultdict(list)

    for path, m in manifests:
        cfg = m.get("config", {})
        exp_cfg = cfg.get("experiment", {})
        exp_name = exp_cfg.get("name", "unknown")
        seed = exp_cfg.get("seed", "?")
        model_cfg = cfg.get("model", {})
        backbone = model_cfg.get("backbone", "?")
        cond = model_cfg.get("conditioning", "none")
        ba = m.get("overall_oof_ba", float("nan"))
        t_star = m.get("optimal_threshold", float("nan"))
        fast = m.get("fast_mode", False)
        run_id = m.get("run_id", os.path.basename(os.path.dirname(path)))

        entry = {
            "run_id": run_id,
            "exp_name": exp_name,
            "backbone": backbone,
            "conditioning": cond,
            "seed": seed,
            "ba": ba,
            "t_star": t_star,
            "fast": "Yes" if fast else "No (5-Fold)",
        }
        runs.append(entry)
        if not np.isnan(ba):
            grouped_by_exp[f"{exp_name} ({backbone}+{cond})"].append(ba)

    # Sort runs by BA descending
    runs.sort(key=lambda x: -1.0 if np.isnan(x["ba"]) else x["ba"], reverse=True)

    print("\n" + "=" * 95)
    print("PAREIDOLIA EXPERIMENT RUNS LEADERBOARD (STANDUP ARTIFACT)")
    print("=" * 95)
    header = f"{'Rank':4s} | {'Run ID':44s} | {'Backbone':12s} | {'Cond':6s} | {'5-Fold':10s} | {'OOF BA':7s} | {'t*':5s}"
    print(header)
    print("-" * 95)

    for rank, r in enumerate(runs, start=1):
        ba_str = f"{r['ba']:.4f}" if not np.isnan(r['ba']) else "  N/A "
        t_str = f"{r['t_star']:.3f}" if not np.isnan(r['t_star']) else " N/A "
        print(
            f"{rank:4d} | {r['run_id']:44s} | {r['backbone']:12s} | {r['conditioning']:6s} | "
            f"{r['fast']:10s} | {ba_str:7s} | {t_str:5s}"
        )

    print("=" * 95)

    if grouped_by_exp:
        print("\nSUMMARY BY RECIPE (Mean +/- Std across runs/seeds):")
        print("-" * 65)
        for name, scores in sorted(grouped_by_exp.items(), key=lambda x: -np.mean(x[1])):
            mean_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0.0
            print(f"{name:45s} : {mean_score:.4f} +/- {std_score:.4f} (N={len(scores)})")
        print("-" * 65)


if __name__ == "__main__":
    main()
