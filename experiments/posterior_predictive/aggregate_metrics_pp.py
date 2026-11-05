"""
Aggregate posterior-predictive metrics across multiple runs and observations.

Reads metrics_obs_{obs_seed}_run_{run_idx}.json from results/pp_guidance/
averages over runs for each observation, then computes mean ± std (ddof=1)
over observations.

Usage:
    python experiments/posterior_predictive/aggregate_metrics_pp.py --task_name oup --task_type id --model_id 0
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["id", "ood"])
    parser.add_argument("--model_id", type=int, required=True)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to results directory (default: <repo root>/results)",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        results_path = Path(__file__).parent.parent.parent / "results"
    else:
        results_path = Path(args.results_dir)

    metrics_dir = (
        results_path / "metrics_pp" / f"{args.task_name}_{args.task_type}" / f"model_{args.model_id}"
    )
    output_dir = metrics_dir
    os.makedirs(output_dir, exist_ok=True)

    metrics_files = sorted(metrics_dir.glob("metrics_obs_*_run_*.json"))
    if not metrics_files:
        print(f"ERROR: No metrics files found in {metrics_dir}")
        return

    print(f"Found {len(metrics_files)} metrics files in {metrics_dir}")

    # {method: {metric: {obs_seed: [value per run, ...]}}}
    per_obs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in metrics_files:
        obs_seed = int(f.stem.split("_")[2])  # metrics_obs_<seed>_run_<i>
        data = json.loads(f.read_text())
        for method, method_metrics in data.items():
            for metric_name, value in method_metrics.items():
                per_obs[method][metric_name][obs_seed].append(value)

    # Average over runs for each observation; mean and std are then taken over
    # these per-observation values.
    all_values = {
        method: {m: [float(np.mean(v)) for v in by_obs.values()] for m, by_obs in metrics.items()}
        for method, metrics in per_obs.items()
    }

    methods = sorted(all_values.keys())
    rows = []
    for method in methods:
        row = {"method": method}
        for metric_name in sorted(all_values[method].keys()):
            vals = np.array(all_values[method][metric_name])
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
            row[metric_name] = f"{mean:.4f} ± {std:.4f}"
            print(f"  {method:25s} {metric_name:10s}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = output_dir / f"summary_pp_{args.task_name}_{args.task_type}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
