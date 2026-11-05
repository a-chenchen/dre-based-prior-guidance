"""
Aggregate metrics across multiple runs and observations.

For each method and metric, averages over runs for each observation, then
computes mean and std (ddof=1) over observations, and saves results to CSV.

Usage:
    python experiments/posterior/aggregate_metrics.py --task_name oup --task_type id --model_id 0 --outdir <path_to_results>
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from multiple runs and observations"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name (e.g., oup, turin, two_moons)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["id", "ood"],
        help="Task type",
    )
    parser.add_argument(
        "--model_id",
        type=int,
        required=True,
        help="Model ID",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to results directory (default: <repo root>/results)",
    )

    parser.add_argument(
        "--exp_tag",
        type=str,
        default="",
        help="Experiment tag subdirectory (must match --exp_tag used in run_guidance.py). Empty = no subdirectory.",
    )

    args = parser.parse_args()

    if args.results_dir is None:
        exp_path = Path(__file__).parent.parent
        results_path = exp_path.parent / "results"
    else:
        results_path = Path(args.results_dir)

    _metrics_base = (
        results_path
        / "metrics"
        / f"{args.task_name}_{args.task_type}"
        / f"model_{args.model_id}"
    )
    metrics_dir = _metrics_base / args.exp_tag if args.exp_tag else _metrics_base

    # Find all metrics files
    metrics_files = sorted(metrics_dir.glob("metrics_obs_*_run_*.json"))

    if len(metrics_files) == 0:
        print(f"ERROR: No metrics files found in {metrics_dir}")
        return

    print(f"Found {len(metrics_files)} metrics files in {metrics_dir}")

    # Parse metrics and organize by method, metric, obs, run
    # Structure: {method: {metric: {obs_seed: {run_idx: value}}}}
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for metrics_file in metrics_files:
        # Parse filename: metrics_obs_1000000_run_0.json
        filename = metrics_file.stem
        parts = filename.split("_")
        # parts = ['metrics', 'obs', '1000000', 'run', '0']
        obs_seed = int(parts[2])
        run_idx = int(parts[4])

        with open(metrics_file, "r") as f:
            metrics_dict = json.load(f)

        for method, method_metrics in metrics_dict.items():
            for metric_name, metric_value in method_metrics.items():
                all_metrics[method][metric_name][obs_seed][run_idx] = metric_value

    # Compute mean and std for each method and metric
    # Structure: {method: {metric: {"mean": float, "std": float}}}
    summary_stats = {}

    methods = sorted(all_metrics.keys())
    if len(methods) == 0:
        print("ERROR: No methods found in metrics files")
        return

    for method in methods:
        summary_stats[method] = {}
        metrics_in_method = all_metrics[method]

        for metric_name in sorted(metrics_in_method.keys()):
            # Average over runs for each observation; mean and std are then
            # taken over these per-observation values.
            all_values = [
                float(np.mean(list(runs.values())))
                for runs in metrics_in_method[metric_name].values()
            ]

            if len(all_values) > 0:
                mean_val = float(np.mean(all_values))
                std_val = float(np.std(all_values, ddof=1))
                summary_stats[method][metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": len(all_values),
                }
            else:
                summary_stats[method][metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "n": 0,
                }

    # Create summary DataFrame
    # Each row is a method, each column is a metric with format "mean (std)"
    df_data = []
    for method in methods:
        row = {"method": method}
        for metric_name in sorted(summary_stats[method].keys()):
            stats = summary_stats[method][metric_name]
            mean_val = stats["mean"]
            std_val = stats["std"]
            if np.isfinite(mean_val) and np.isfinite(std_val):
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[metric_name] = "NaN"
        df_data.append(row)

    df_summary = pd.DataFrame(df_data)

    # Save to CSV
    os.makedirs(metrics_dir, exist_ok=True)
    _tag_suffix = f"_{args.exp_tag}" if args.exp_tag else ""
    csv_path = (
        metrics_dir / f"summary_metrics_{args.task_name}_{args.task_type}{_tag_suffix}.csv"
    )
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary (mean ± std) saved to: {csv_path}")
    print(df_summary.to_string())

    # Also save numeric values (for plotting if needed)
    df_numeric_mean = []
    df_numeric_std = []

    for method in methods:
        row_mean = {"method": method}
        row_std = {"method": method}
        for metric_name in sorted(summary_stats[method].keys()):
            stats = summary_stats[method][metric_name]
            row_mean[metric_name] = stats["mean"]
            row_std[metric_name] = stats["std"]
        df_numeric_mean.append(row_mean)
        df_numeric_std.append(row_std)

    df_numeric_mean = pd.DataFrame(df_numeric_mean)
    df_numeric_std = pd.DataFrame(df_numeric_std)

    mean_path = metrics_dir / f"summary_metrics_mean_{args.task_name}_{args.task_type}{_tag_suffix}.csv"
    std_path = metrics_dir / f"summary_metrics_std_{args.task_name}_{args.task_type}{_tag_suffix}.csv"

    df_numeric_mean.to_csv(mean_path, index=False)
    df_numeric_std.to_csv(std_path, index=False)

    print(f"Mean values saved to: {mean_path}")
    print(f"Std values saved to: {std_path}")

    # Print detailed statistics
    print("\n" + "="*80)
    print("Detailed Statistics (for reference)")
    print("="*80)
    for method in methods:
        print(f"\n{method}:")
        for metric_name in sorted(summary_stats[method].keys()):
            stats = summary_stats[method][metric_name]
            print(
                f"  {metric_name:15s}: {stats['mean']:8.4f} ± {stats['std']:8.4f} (n={stats['n']})"
            )


if __name__ == "__main__":
    main()
