"""Generate and save reference observations and target posterior samples.

For each requested seed this script:
  1. Samples (theta, x) from the task's *target* prior and saves it as JSON.
  2. Runs the task-specific posterior sampler conditioned on that x and saves
     the samples as NPZ.

Saved file layout
-----------------
<save_dir>/
  obs_<seed>.json        # {"theta": [...], "x": [...]}
  ref_<seed>.npz         # array key "samples", shape (num_samples, theta_dim)

Usage examples
--------------
# Generate 5 observations + 10 000 posterior samples for slcp_id:
python experiments/data/gen_reference_samples.py \\
    --task_name slcp \\
    --task_type id \\
    --seeds 0 1 2 3 4 \\
    --num_samples 10000 \\
    --save_dir data/reference/slcp_id

# Quick smoke-test with fewer samples:
python experiments/data/gen_reference_samples.py \\
    --task_name two_moons \\
    --task_type id \\
    --seeds 0 \\
    --num_samples 100
"""

import argparse
import logging
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

# Make `import priorg` work when the script is run directly from a checkout.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from priorg.sim.tasks.task import get_task


DEFAULT_SEEDS = [1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1000010, 1000012, 1000008, 1000009]
SUPPORTED_TASK_NAMES = [
    "two_moons",
    "gaussian_linear_6d",
    "slcp",
    "oup",
    "turin",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reference observations and posterior samples for a priorg task."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=SUPPORTED_TASK_NAMES,
        help="Task name (e.g. slcp, oup, two_moons).",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["id", "ood", "truncated"],
        help="Task type: id or ood.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Integer seeds to generate observations for. Default: 0..9.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10_000,
        help="Number of posterior samples to draw per observation. Default: 10000.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help=(
            "Directory to write files into. "
            "Default: data/reference/<task_name> relative to this script."
        ),
    )
    parser.add_argument(
        "--obs_only",
        action="store_true",
        help="Only generate observations (skip posterior sampling).",
    )
    parser.add_argument(
        "--posterior_only",
        action="store_true",
        help=(
            "Only run posterior sampling (obs_{seed}.json must already exist). "
            "Skips observation generation."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    task_full_name = f"{args.task_name}_{args.task_type}"

    save_dir = Path(
        args.save_dir
        if args.save_dir is not None
        else Path(__file__).parent / "reference" / task_full_name
    )

    log.info(f"Task      : {task_full_name}")
    log.info(f"Seeds     : {args.seeds}")
    log.info(f"Samples   : {args.num_samples}")
    log.info(f"Save dir  : {save_dir}")

    task = get_task(task_full_name)
    log.info(
        f"theta_dim={task.theta_dim}, x_dim={task.x_dim}"
    )

    for seed in args.seeds:
        # ------------------------------------------------------------------
        # 1. Observation
        # ------------------------------------------------------------------
        if not args.posterior_only:
            obs = task.generate_and_save_observation(seed=seed, save_dir=save_dir)
            log.info(
                f"[seed={seed}] observation saved  -> {save_dir}/obs_{seed}.json  "
                f"(theta={obs['theta'][:2]}{'...' if len(obs['theta']) > 2 else ''})"
            )
        else:
            obs = task.get_saved_observation(seed=seed, save_dir=save_dir)
            log.info(f"[seed={seed}] observation loaded from {save_dir}/obs_{seed}.json")

        # ------------------------------------------------------------------
        # 2. Reference posterior
        # ------------------------------------------------------------------
        if not args.obs_only:
            log.info(f"[seed={seed}] running posterior sampler ({args.num_samples} samples)...")
            samples = task.generate_and_save_reference_posterior(
                seed=seed, save_dir=save_dir, num_samples=args.num_samples
            )
            log.info(
                f"[seed={seed}] posterior saved    -> {save_dir}/ref_{seed}.npz  "
                f"shape={samples.shape}"
            )

    log.info("Done.")


if __name__ == "__main__":
    main()
