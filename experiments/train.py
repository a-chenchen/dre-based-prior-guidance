"""
Train a base AllConditionalScoreModel on the reference prior for a given task.

Data is simulated from the reference prior and used to train a joint
score model (structured_random conditioning). The trained model is saved to:
    experiments/models/<task_name>/<task_type>/model_<model_id>.pkl

This is the base model consumed by experiments/posterior/run_guidance.py.
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

# Make `import priorg` work when the script is run directly from a checkout.
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

import jax

from priorg.sim.tasks.task import get_task
from priorg.sim.methods.methods import run_score_transformer


class _Cfg(dict):
    """Dict subclass with attribute access, mimicking OmegaConf DictConfig."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _default_condition_mask_name(args) -> str:
    """Pick the conditioning strategy compatible with the model variant.

    With an observation embedding network the observation is condensed into
    tokens, so it can no longer be conditioned scalar by scalar: the mask must
    keep each observation block either fully observed or fully latent (see
    Gloeckler et al. 2024, Sec. 3.1 and App. A3.2).
    """
    if args.condition_mask_fn is not None:
        return args.condition_mask_fn
    if not args.embed_x:
        return "structured_random"
    if args.predict_x_score:
        return "block_posterior"
    return "theta_random"


def _build_method_cfg(args) -> _Cfg:
    condition_mask_name = _default_condition_mask_name(args)
    if args.sde_name == "vpsde":
        sde_cfg = _Cfg(
            name="vpsde",
            beta_max=args.beta_max,
            beta_min=args.beta_min,
            T_max=1.0,
            T_min=1e-5,
            scale_min=0.001,
        )
    else:
        sde_cfg = _Cfg(
            name="vesde",
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            T_max=1.0,
            T_min=1e-5,
            scale_min=0.001,
        )

    return _Cfg(
        device="cpu",
        sde=sde_cfg,
        model=_Cfg(
            token_dim=args.token_dim,
            condition_token_dim=args.condition_token_dim,
            condition_token_init_scale=0.1,
            condition_token_init_mean=0.0,
            condition_mode="concat",
            time_embedding_dim=128,
            num_heads=4,
            num_layers=6,
            attn_size=10,
            widening_factor=3,
            num_hidden_layers=1,
            skip_connection_attn=True,
            skip_connection_mlp=True,
            layer_norm=True,
            embed_x=args.embed_x,
            num_x_tokens=args.num_x_tokens,
            x_embed_hidden=tuple(args.x_embed_hidden),
            x_embed_chunked=args.x_embed_chunked,
            x_embed_layer_norm=True,
            predict_x_score=args.predict_x_score,
        ),
        train=_Cfg(
            condition_mask_fn=_Cfg(name=condition_mask_name),
            edge_mask_fn=_Cfg(name="none"),
            max_number_steps=args.max_steps,
            min_number_steps=5000,
            total_number_steps_scaling=3,
            training_batch_size=args.batch_size,
            learning_rate=args.lr,
            min_learning_rate=1e-6,
            clip_max_norm=10.0,
            validation_fraction=0.05,
            val_repeat=5,
            val_every=50,
            stop_early_count=args.stop_early_count,
            val_error_ratio=args.val_error_ratio,
            rebalance_loss=False,
            z_score_data=args.z_score_data,
        ),
        posterior=_Cfg(
            sampling_method="sde",
            num_steps=500,
            method="euler_maruyama",
        ),
    )


def train():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Train base score model on the reference prior for a task"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="oup",
        choices=["oup", "turin", "two_moons", "gaussian_linear_6d", "slcp"],
        help="Task name (default: oup)",
    )
    parser.add_argument(
        "--stop_early_count",
        type=int,
        default=5,
        help="Consecutive validation checks with val/train loss above "
             "--val_error_ratio before training stops. Raise it (or set a "
             "large value to effectively disable) if training stops too soon "
             "(default: 5)",
    )
    parser.add_argument(
        "--val_error_ratio",
        type=float,
        default=1.1,
        help="val/train loss ratio that counts as a deterioration. Note the "
             "loss is a masked sum over latent nodes, so conditioning schemes "
             "that leave few nodes latent (e.g. theta_random) give a noisier "
             "ratio and trip this sooner (default: 1.1)",
    )
    parser.add_argument(
        "--token_dim",
        type=int,
        default=40,
        help="Width of every token in the sequence. With --embed_x this is "
             "also the width of each observation token, so the observation "
             "embedding the DRE reuses has dimension token_dim * num_x_tokens "
             "(default: 40)",
    )
    parser.add_argument(
        "--condition_token_dim",
        type=int,
        default=10,
        help="Width of the condition-state token appended to each token "
             "(default: 10)",
    )
    parser.add_argument(
        "--embed_x",
        action="store_true",
        help="Condense the observation into tokens with a jointly-trained "
             "embedding network instead of one token per observation scalar "
             "(Gloeckler et al. 2024, Sec. 3.1 / App. A3.2)",
    )
    parser.add_argument(
        "--num_x_tokens",
        type=int,
        default=1,
        help="Number of tokens the observation is condensed into (default: 1)",
    )
    parser.add_argument(
        "--x_embed_hidden",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden widths of the observation embedding MLP (default: 256 128)",
    )
    parser.add_argument(
        "--x_embed_chunked",
        action="store_true",
        help="Split the observation into num_x_tokens contiguous blocks with "
             "one MLP each, preserving block-level conditioning (partial "
             "posteriors), instead of one MLP over the whole vector",
    )
    parser.add_argument(
        "--predict_x_score",
        action="store_true",
        help="Also decode scores for the observation entries, so observation "
             "blocks may be latent. Required for partial posteriors.",
    )
    parser.add_argument(
        "--condition_mask_fn",
        type=str,
        default=None,
        help="Override the conditioning strategy (default: chosen from "
             "--embed_x / --predict_x_score)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="id",
        choices=["id", "ood", "truncated"],
        help="Task type (default: id)",
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=0,
        choices=list(range(5)),
        help="Model ID used in the saved filename (default: 0)",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=50000,
        help="Number of (theta, x) pairs to simulate from the reference prior (default: 50000)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Maximum training steps (default: 50000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Training batch size (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--sde_name",
        type=str,
        default="vpsde",
        choices=["vpsde", "vesde"],
        help="Diffusion SDE used for training (default: vpsde)",
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.1,
        help="Minimum beta for VPSDE (default: 0.1)",
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        default=10.0,
        help="Maximum beta for VPSDE (default: 10.0)",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=1e-4,
        help="Minimum sigma for VESDE (default: 1e-4)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=15.0,
        help="Maximum sigma for VESDE (default: 15.0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--z_score_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Z-score training data (default: True). Pass --no-z_score_data to disable.",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        default=False,
        help=(
            "Save the simulated training data as data_<model_id>.pkl alongside "
            "the model (needed for DRE training / augmentation in run_guidance.py)."
        ),
    )
    args = parser.parse_args()

    task_full_name = f"{args.task_name}_{args.task_type}"
    logger.info(
        f"Training base model | task={task_full_name}, model_id={args.model_id}, "
        f"num_simulations={args.num_simulations}, max_steps={args.max_steps}, "
        f"batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}, sde={args.sde_name}, "
        f"z_score_data={args.z_score_data}"
    )

    exp_path = Path(__file__).parent
    model_dir = exp_path / "models" / args.task_name / args.task_type
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / f"model_{args.model_id}.pkl"

    if model_path.exists():
        logger.warning(
            "%s already exists and will be OVERWRITTEN. Model variants "
            "(e.g. --embed_x) must use a different --model_id, otherwise they "
            "clobber the baseline that run_guidance.py loads by the same id.",
            model_path,
        )

    task = get_task(task_full_name)
    logger.info(f"Task built: theta_dim={task.theta_dim}, x_dim={task.x_dim}")

    key = jax.random.PRNGKey(args.seed)
    key, key_data, key_train = jax.random.split(key, 3)

    logger.info(f"Simulating {args.num_simulations} samples from reference prior...")
    data = task.get_data(args.num_simulations, key=key_data)
    logger.info(f"  theta: {data['theta'].shape}, x: {data['x'].shape}")

    if args.save_data:
        data_path = model_dir / f"data_{args.model_id}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Training data saved to {data_path}")

    method_cfg = _build_method_cfg(args)

    logger.info("Starting training...")
    model = run_score_transformer(task, data, method_cfg, rng=key_train)
    logger.info("Training complete.")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
