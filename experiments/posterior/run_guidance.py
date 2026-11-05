"""
Posterior experiment: adapt a score model trained under a reference prior p to
a target prior q, and score every method against reference posterior samples.

Pipeline:
  1. Load the pretrained base score model (AllConditionalScoreModel).
  2. Build the Task, which carries the reference prior p and the target prior q.
  3. Train the correction models ONCE per run -- the prior ratio log q/p does
     not depend on the observation:
       - log-ratio models for `guided` (logistic) and `kliep_guided` (KLIEP)
       - the score correction for `wsm_guided`
       - the GMM fit to log q/p for `priorguide`
  4. For each evaluation observation, sample from every selected method, save
     the samples to NPZ, and score them against the reference posterior.

Use --methods to run only a subset.  The metrics file is loaded from disk (if
it exists) and only the selected methods' entries are updated, so results for
other methods are preserved.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Make `import priorg` work when the script is run directly from a checkout.
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

import jax
import jax.numpy as jnp
import numpy as np
import torch

from priorg.sim.tasks.task import get_task
from priorg.sim.methods.guidance_dre import (
    LogRatioModelSpec,
    init_log_ratio_model,
    prior_guide_with_log_ratio_model,
    sample_unguided,
    train_log_ratio_model,
    train_log_ratio_model_kliep,
)
from priorg.sim.methods.guidance_wsm import (
    WSMModelSpec,
    init_wsm_model,
    train_wsm_model,
    prior_guide_with_wsm,
)
from priorg.sim.methods.data_augment import build_augmented_dataset
from priorg.sim.methods.gmm import fit_gmm, fit_gmm_v2
from priorg.sim.methods.guidance_priorguide import prior_guide_theta_prior_only
from priorg.sim.methods.metrics import (
    compute_mmd,
    compute_gskl,
    compute_wasserstein2,
    compute_c2st as _compute_c2st,
    compute_rmse,
)

def compute_c2st(x, y):
    acc = float(_compute_c2st(x, y))
    return max(acc, 1.0 - acc)

observation_seeds = [
    1000000,  # observation 1
    1000001,  # observation 2
    1000002,  # observation 3
    1000003,  # observation 4
    1000004,  # observation 5
    1000005,  # observation 6
    1000010,  # observation 7
    1000012,  # observation 8
    1000008,  # observation 9
    1000009,  # observation 10
]

_DEFAULT_METHODS = [
    "guided",            # ours, logistic NRE log-ratio guidance
    "kliep_guided",      # ours, KLIEP log-ratio guidance
    "wsm_guided",        # weighted score matching correction
    "priorguide",        # PriorGuide (GMM prior-ratio guidance)
    "simformer",         # unguided base model
    "simformer_importance",  # base model + importance resampling by log q/p
]

_ALL_METHODS = list(_DEFAULT_METHODS)


def _extract_gmm_from_distribution(dist):
    base = getattr(dist, "base_dist", None)
    if base is not None and hasattr(base, "loc") and hasattr(base, "scale"):
        loc = jnp.array(base.loc.detach().cpu().numpy())
        scale = jnp.array(base.scale.detach().cpu().numpy())
        cov = jnp.diag(scale**2)
        return jnp.array([0.0]), loc[None, :], cov[None, :, :]

    if hasattr(dist, "loc") and hasattr(dist, "covariance_matrix"):
        loc = jnp.array(dist.loc.detach().cpu().numpy())
        cov = jnp.array(dist.covariance_matrix.detach().cpu().numpy())
        if loc.ndim == 1:
            return jnp.array([0.0]), loc[None, :], cov[None, :, :]
        num_components = loc.shape[0]
        return jnp.full((num_components,), -jnp.log(num_components)), loc, cov

    mixing_dist = getattr(dist, "mixing_distribution", None)
    if mixing_dist is None:
        mixing_dist = getattr(dist, "mixture_distribution", None)
    component_dist = getattr(dist, "component_distribution", None)
    if mixing_dist is not None and component_dist is not None:
        probs = getattr(mixing_dist, "probs", None)
        if probs is None:
            probs = torch.softmax(mixing_dist.logits, dim=-1)
        probs = jnp.array(probs.detach().cpu().numpy())

        comp_base = getattr(component_dist, "base_dist", None)
        if comp_base is not None and hasattr(comp_base, "loc") and hasattr(comp_base, "scale"):
            means = jnp.array(comp_base.loc.detach().cpu().numpy())
            scales = jnp.array(comp_base.scale.detach().cpu().numpy())
            covs = jax.vmap(jnp.diag)(scales**2)
            return jnp.log(probs), means, covs

        if hasattr(component_dist, "loc") and hasattr(component_dist, "covariance_matrix"):
            means = jnp.array(component_dist.loc.detach().cpu().numpy())
            covs = jnp.array(component_dist.covariance_matrix.detach().cpu().numpy())
            return jnp.log(probs), means, covs

    raise NotImplementedError(
        f"Unsupported prior distribution for PriorGuide: {type(dist)}"
    )


def _z_score_gaussian_mixture(model, log_weights, means, covs, latent_node_id):
    if model.z_score_params is None:
        return log_weights, means, covs

    mean_per_node = jnp.asarray(
        model.z_score_params["mean_per_node_id"][latent_node_id]
    ).reshape(-1)
    std_per_node = jnp.asarray(
        model.z_score_params["std_per_node_id"][latent_node_id]
    ).reshape(-1)

    means_z = (means - mean_per_node[None, :]) / std_per_node[None, :]
    covs_z = covs / (std_per_node[None, :, None] * std_per_node[None, None, :])
    return log_weights, means_z, covs_z


def _z_score_observation_if_needed(model, x_o):
    if model.z_score_params is None:
        return x_o

    z_score_fn = model.z_score_params["z_score_fn"]
    cond_node_id = model.node_id[model.condition_mask.astype(bool)]
    return z_score_fn(x_o, cond_node_id)


def _un_z_score_latent_if_needed(model, result):
    if model.z_score_params is None:
        return result

    un_z_score_fn = model.z_score_params["un_z_score_fn"]
    # Use a concrete numpy boolean array so this function is safe inside @jax.jit.
    # model.condition_mask is a JAX array; inside JIT it becomes abstract and
    # cannot be used directly as a boolean index (data-dependent shape).
    latent_mask_bool = ~np.asarray(model.condition_mask, dtype=bool)
    latent_node_id = model.node_id[latent_mask_bool]
    latent_vals = result[latent_mask_bool]
    latent_vals = un_z_score_fn(latent_vals, latent_node_id)
    return result.at[latent_mask_bool].set(latent_vals)


def _importance_resample_by_ratio(samples, log_ratio_fn, theta_dim, key):
    theta_samples = jnp.asarray(samples[:, :theta_dim])
    log_weights = jax.vmap(log_ratio_fn)(theta_samples).reshape(-1)
    finite_mask = jnp.isfinite(log_weights)
    safe_log_weights = jnp.where(finite_mask, log_weights, -jnp.inf)

    def _uniform_weights(_):
        return jnp.full((samples.shape[0],), 1.0 / samples.shape[0])

    def _ratio_weights(weights_):
        return jax.nn.softmax(weights_)

    weights = jax.lax.cond(
        jnp.any(finite_mask),
        _ratio_weights,
        _uniform_weights,
        safe_log_weights,
    )
    indices = jax.random.choice(
        key,
        samples.shape[0],
        shape=(samples.shape[0],),
        replace=True,
        p=weights,
    )
    return jnp.asarray(samples)[indices], weights


def evaluate():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run log-ratio guided diffusion for prior adaptation under uniform training priors"
    )
    parser.add_argument(
        "--sde_name",
        type=str,
        default=None,
        choices=["vpsde", "vesde"],
        help="Optional override check for the model SDE type; defaults to the checkpoint config",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="oup",
        choices=["oup", "turin", "two_moons", "gaussian_linear_6d", "slcp"],
        help="Task name (default: oup)",
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
        help="Model ID (default: 0)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of posterior samples per observation (default: 5000)",
    )
    parser.add_argument("--rho", type=float, default=2.0, help="Time schedule exponent (default: 2.0)")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=25,
        help="Number of reverse diffusion steps (default: 25)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the log-ratio gradient guidance (default: 1.0)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches to split num_samples into (default: 5)",
    )
    # Log-ratio training arguments
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4096,
        help="Batch size for log-ratio model training (default: 4096)",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=12000,
        help="Maximum training epochs for log-ratio model (default: 12000)",
    )
    parser.add_argument(
        "--train_lr",
        type=float,
        default=3e-4,
        help="AdamW learning rate for log-ratio model training (default: 3e-4)",
    )
    parser.add_argument(
        "--train_patience",
        type=int,
        default=0,
        help="Early-stopping patience in epochs; set <= 0 to disable (default: 0)",
    )
    parser.add_argument(
        "--load_cached_params",
        action="store_true",
        default=False,
        help="Load previously saved model params instead of retraining (default: False)",
    )
    parser.add_argument(
        "--reinit_output_head",
        action="store_true",
        default=True,
        help="Randomly reinitialise the output head of the log-ratio model (default: True)",
    )
    parser.add_argument(
        "--log_ratio_model_type",
        type=str,
        default="mlp",
        choices=["transformer", "mlp", "mlp_large", "mlp_embedded"],
        help="Log-ratio model backend: transformer reuses the score backbone, "
             "mlp/mlp_large use random initialization, mlp_embedded reuses the "
             "observation embedding network of an embed_x score model (default: mlp)",
    )
    parser.add_argument(
        "--log_ratio_finetune_embedding",
        action="store_true",
        help="With --log_ratio_model_type mlp_embedded, fine-tune the reused "
             "observation encoder instead of keeping it frozen (default: frozen)",
    )
    parser.add_argument(
        "--log_ratio_mlp_hidden_dim",
        type=int,
        default=256,
        help="Hidden width for the optional MLP log-ratio model (default: 256)",
    )
    parser.add_argument(
        "--log_ratio_mlp_num_layers",
        type=int,
        default=4,
        help="Number of hidden layers for the optional MLP log-ratio model (default: 4)",
    )
    parser.add_argument(
        "--log_ratio_mlp_time_embedding_dim",
        type=int,
        default=4,
        help="Fourier time embedding dimension for the optional MLP log-ratio model (default: 4)",
    )
    parser.add_argument(
        "--log_ratio_mlp_activation",
        type=str,
        default="silu",
        choices=["silu", "gelu"],
        help="Activation for the optional MLP log-ratio model (default: silu)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--priorguide_gmm_version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help=(
            "GMM fitting version for PriorGuide: "
            "'v1' = original fit_gmm (samples from GMM itself), "
            "'v2' = fit_gmm_v2 (samples from q(theta), more stable) (default: v2)"
        ),
    )
    parser.add_argument(
        "--priorguide_gmm_iters",
        type=int,
        default=20000,
        help="Optimisation steps for the PriorGuide GMM fit (default: 20000). "
             "Lower it for smoke tests.",
    )
    parser.add_argument(
        "--priorguide_langevin_steps",
        type=int,
        default=4,
        help="Number of Langevin refinement steps for exact PriorGuide (default: 4)",
    )
    parser.add_argument(
        "--priorguide_langevin_ratio",
        type=float,
        default=0.5,
        help="Langevin step-size ratio for exact PriorGuide (default: 0.5)",
    )
    parser.add_argument(
        "--num_obs",
        type=int,
        default=None,
        help="Evaluate only the first N observation seeds (default: all 10). "
             "Handy for smoke tests.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of independent runs per observation (default: 1)",
    )
    parser.add_argument(
        "--run_start_idx",
        type=int,
        default=0,
        help="Starting run index, used to offset run_idx for SLURM array jobs (default: 0)",
    )
    parser.add_argument(
        "--augment_samples",
        type=int,
        default=420000,
        help=(
            "Number of additional (theta, x) pairs to generate for the guidance "
            "model training dataset by sampling theta from the base model conditioned "
            "on x values drawn from the base training set. 0 = no augmentation (default: 420000)."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=_DEFAULT_METHODS,
        choices=_ALL_METHODS,
        metavar="METHOD",
        help=(
            f"Which methods to run (default: all). "
            f"Choices: {_ALL_METHODS}. "
            "Partial runs load the existing metrics file and update only the "
            "selected methods' entries — other methods are preserved. "
            "'simformer_importance' automatically includes 'simformer'."
        ),
    )

    parser.add_argument(
        "--exp_tag",
        type=str,
        default="",
        help=(
            "Optional experiment tag inserted as a subdirectory in all result paths "
            "(e.g. 'baseline', 'softmax'). Empty string = original paths, no subdirectory."
        ),
    )

    # paras for softmax-in-t relaxation (minimax objective)

    parser.add_argument(
        "--softmax_t",
        action="store_true",
        default=False,
        help="Enable soft-max-in-t relaxation (minimax objective). Default: disabled (plain expectation over t).",
    )
    parser.add_argument(
        "--n_time_samples",
        type=int,
        default=8,
        help="Number of t samples per step when --softmax_t is enabled (default: 8)",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=0.0,
        help="Initial inverse-temperature for soft-max anneal (default: 0.0 = pure mean)",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=8.0,
        help="Final inverse-temperature for soft-max anneal (default: 8.0)",
    )
    parser.add_argument(
        "--beta_warmup_frac",
        type=float,
        default=0.0,
        help="Fraction of epochs held at beta_start before anneal begins (default: 0.0)",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "exp"],
        help="Beta annealing schedule (default: linear)",
    )

    args = parser.parse_args()

    eval_obs_seeds = (
        observation_seeds if args.num_obs is None
        else observation_seeds[: args.num_obs]
    )

    run_methods = set(args.methods)
    # simformer_importance derives from simformer samples, so always include simformer.
    if "simformer_importance" in run_methods:
        run_methods.add("simformer")

    logger.info(
        f"Running log-ratio guide | task={args.task_name}, type={args.task_type}, "
        f"model_id={args.model_id}, num_samples={args.num_samples}, "
        f"num_steps={args.num_steps}, rho={args.rho}, guidance_scale={args.guidance_scale}, "
        f"train_epochs={args.train_epochs}, train_lr={args.train_lr}, seed={args.seed}, "
        f"log_ratio_model_type={args.log_ratio_model_type}, num_runs={args.num_runs}, "
        f"methods={sorted(run_methods)}"
    )

    exp_path = Path(__file__).parent.parent
    results_path = exp_path.parent / "results"

    model_path = exp_path / "models" / args.task_name / args.task_type / f"model_{args.model_id}.pkl"

    def _res(method_dir):
        base = results_path / method_dir / args.task_name / args.task_type / f"model_{args.model_id}"
        return base / args.exp_tag if args.exp_tag else base

    log_ratio_guide_path = _res("guidance")
    os.makedirs(log_ratio_guide_path, exist_ok=True)

    simformer_path = _res("simformer")
    os.makedirs(simformer_path, exist_ok=True)

    simformer_importance_path = _res("simformer_importance")
    os.makedirs(simformer_importance_path, exist_ok=True)

    priorguide_path = _res("priorguide")
    os.makedirs(priorguide_path, exist_ok=True)

    kliep_guide_path = _res("kliep_guidance")
    os.makedirs(kliep_guide_path, exist_ok=True)

    wsm_guide_path = _res("wsm_guidance")
    os.makedirs(wsm_guide_path, exist_ok=True)

    _metrics_base = results_path / "metrics" / f"{args.task_name}_{args.task_type}" / f"model_{args.model_id}"
    metrics_path = _metrics_base / args.exp_tag if args.exp_tag else _metrics_base
    os.makedirs(metrics_path, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load score model
    # ------------------------------------------------------------------
    logger.info(f"Loading score model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if args.sde_name is not None:
        ckpt_sde_name = str(model.sde_init_params.get("name", "")).lower()
        if ckpt_sde_name != args.sde_name:
            raise ValueError(
                f"Checkpoint SDE '{ckpt_sde_name}' does not match requested --sde_name '{args.sde_name}'."
            )

    data_path = exp_path / "models" / args.task_name / args.task_type / f"data_{args.model_id}.pkl"
    if data_path.exists():
        with open(data_path, "rb") as f:
            base_dataset = pickle.load(f)
        logger.info(
            f"Loaded base training dataset from {data_path} "
            f"(theta: {base_dataset['theta'].shape}, x: {base_dataset['x'].shape})"
        )
    else:
        base_dataset = None
        logger.warning(f"Base dataset not found at {data_path}; will sample from task at each epoch.")

    if base_dataset is not None and args.augment_samples > 0:
        aug_cache_path = (
            exp_path / "models" / args.task_name / args.task_type
            / f"augmented_data_{args.model_id}_{args.augment_samples}.pkl"
        )
        if aug_cache_path.exists():
            logger.info(f"Loading cached augmented dataset from {aug_cache_path}")
            with open(aug_cache_path, "rb") as f:
                base_dataset = pickle.load(f)
            logger.info(
                f"Cached augmented dataset: theta={base_dataset['theta'].shape}, "
                f"x={base_dataset['x'].shape}"
            )
        else:
            logger.info(
                f"Augmenting training dataset with {args.augment_samples} model-sampled "
                f"(theta, x) pairs (num_steps={args.num_steps}, rho={args.rho}) ..."
            )
            _theta_dim = int((~model.condition_mask.astype(bool)).sum())
            key_aug = jax.random.PRNGKey(args.seed + 99999)
            base_dataset = build_augmented_dataset(
                base_dataset=base_dataset,
                model=model,
                condition_mask=model.condition_mask,
                theta_dim=_theta_dim,
                n_augment=args.augment_samples,
                key=key_aug,
                num_steps=args.num_steps,
                rho=args.rho,
            )
            logger.info(
                f"Augmented dataset size: theta={base_dataset['theta'].shape}, "
                f"x={base_dataset['x'].shape}"
            )
            with open(aug_cache_path, "wb") as f:
                pickle.dump(base_dataset, f)
            logger.info(f"Augmented dataset cached to {aug_cache_path}")

    # ------------------------------------------------------------------
    # 2. Build task using default reference / target priors
    # ------------------------------------------------------------------
    task = get_task(f"{args.task_name}_{args.task_type}")
    logger.info(f"Task built: theta_dim={task.theta_dim}, x_dim={task.x_dim}")

    log_ratio_fn = task.make_jax_log_density_ratio_fn()
    logger.info("JAX log-ratio function built from task priors.")

    latent_mask_bool = ~model.condition_mask.astype(bool)
    latent_node_id = model.node_id[latent_mask_bool]

    if "priorguide" in run_methods:
        q_dist = task.get_target_prior()
        p_dist = task.get_reference_prior()
        logger.info(f"Fitting GMM (version={args.priorguide_gmm_version}) to log ratio log q(theta)/p(theta) for PriorGuide...")
        _log_ratio_target = lambda theta: q_dist.log_prob(theta) - p_dist.log_prob(theta)
        _q_sampler = lambda batch_size: q_dist.sample((batch_size,))
        if args.priorguide_gmm_version == "v1":
            _gmm_t0 = time.time()
            _, gmm_params = fit_gmm(
                log_target=_log_ratio_target,
                data_sampler=_q_sampler,
                num_components=50,
                num_iters=args.priorguide_gmm_iters,
                learning_rate=1e-3,
                batch_size=2000,
            )
            logger.info(f"PriorGuide GMM fit (v1) training time: {time.time() - _gmm_t0:.1f}s")
            gmm_log_weights = jnp.array(gmm_params["log_weights"].detach().numpy())
            gmm_means = jnp.array(gmm_params["means"].detach().numpy())
            gmm_stds = jnp.array(jnp.exp(jnp.array(gmm_params["log_stds"].detach().numpy())))
        else:
            _gmm_t0 = time.time()
            gmm_model = fit_gmm_v2(
                log_target_fn=_log_ratio_target,
                data_sampler=_q_sampler,
                num_components=50,
                dim=task.theta_dim,
                num_iters=args.priorguide_gmm_iters,
                learning_rate=1e-3,
                batch_size=200,
            )
            logger.info(f"PriorGuide GMM fit (v2) training time: {time.time() - _gmm_t0:.1f}s")
            gmm_log_weights = jnp.array(gmm_model.log_weights.detach().numpy())
            gmm_means = jnp.array(gmm_model.means.detach().numpy())
            gmm_stds = jnp.array((torch.exp(gmm_model.log_stds) + 1e-6).detach().numpy())
        gmm_cov = (gmm_stds ** 2)[..., None] * jnp.eye(gmm_stds.shape[-1])
        (
            target_log_weights_z,
            target_means_z,
            target_covs_z,
        ) = _z_score_gaussian_mixture(
            model,
            gmm_log_weights,
            gmm_means,
            gmm_cov,
            latent_node_id,
        )
        logger.info(
            f"PriorGuide GMM fit complete: {target_means_z.shape[0]} components."
        )


    theta_dim = task.theta_dim
    x_dim = task.x_dim

    for run_idx in range(args.run_start_idx, args.run_start_idx + args.num_runs):
        key = jax.random.PRNGKey(args.seed + run_idx)

        # Save full config so results can always be traced back to their run settings.
        import subprocess
        try:
            _git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            _git_hash = "unknown"
        _config = {**vars(args), "git_commit": _git_hash}
        _config_path = metrics_path / f"config_run_{run_idx}.json"
        with open(_config_path, "w") as _f:
            json.dump(_config, _f, indent=2)
        logger.info(f"[run={run_idx}] Config saved to {_config_path}")

        # ------------------------------------------------------------------
        # 3. Initialise and train log-ratio model (only if needed)
        # ------------------------------------------------------------------
        if "guided" in run_methods:
            guided_params_path = log_ratio_guide_path / f"params_run_{run_idx}.pkl"
            key, key_init = jax.random.split(key)
            log_ratio_spec = init_log_ratio_model(
                model,
                key_init,
                reinit_output_head=args.reinit_output_head,
                model_type=args.log_ratio_model_type,
                mlp_hidden_dim=args.log_ratio_mlp_hidden_dim,
                mlp_num_layers=args.log_ratio_mlp_num_layers,
                mlp_time_embedding_dim=args.log_ratio_mlp_time_embedding_dim,
                mlp_activation=args.log_ratio_mlp_activation,
                freeze_embedding=not args.log_ratio_finetune_embedding,
            )
            if args.load_cached_params and guided_params_path.exists():
                logger.info(f"[run={run_idx}] Loading cached log-ratio model from {guided_params_path}")
                with open(guided_params_path, "rb") as f:
                    cached_params = pickle.load(f)
                log_ratio_spec = LogRatioModelSpec(
                    log_ratio_apply=log_ratio_spec.log_ratio_apply,
                    params=cached_params,
                )
            else:
                if args.log_ratio_model_type == "transformer":
                    logger.info(f"[run={run_idx}] Log-ratio transformer initialised from score model backbone.")
                elif args.log_ratio_model_type == "mlp_embedded":
                    logger.info(
                        f"[run={run_idx}] Log-ratio MLP initialised with the score model's "
                        f"observation encoder "
                        f"({'fine-tuned' if args.log_ratio_finetune_embedding else 'frozen'})."
                    )
                else:
                    logger.info(f"[run={run_idx}] Log-ratio MLP initialised with random parameters.")

                key, key_train = jax.random.split(key)
                logger.info(
                    f"[run={run_idx}] Training log-ratio model for up to {args.train_epochs} epochs "
                    f"(batch_size={args.train_batch_size}, lr={args.train_lr}, patience={args.train_patience})"
                )
                _nre_t0 = time.time()
                log_ratio_spec, loss_history = train_log_ratio_model(
                    task=task,
                    diffusion_model=model,
                    condition_mask=model.condition_mask,
                    model_spec=log_ratio_spec,
                    rng=key_train,
                    batch_size=args.train_batch_size,
                    epochs=args.train_epochs,
                    lr=args.train_lr,
                    patience=args.train_patience,
                    dataset=base_dataset,
                    softmax_t=args.softmax_t,
                    n_time_samples=args.n_time_samples,
                    beta_start=args.beta_start,
                    beta_end=args.beta_end,
                    beta_warmup_frac=args.beta_warmup_frac,
                    beta_schedule=args.beta_schedule,
                )
                logger.info(
                    f"[run={run_idx}] Training complete. Final loss: {loss_history[-1]:.4f} (over {len(loss_history)} epochs)"
                )
                logger.info(f"[run={run_idx}] NRE (guided) training time: {time.time() - _nre_t0:.1f}s")
                loss_save_path = log_ratio_guide_path / f"loss_history_run_{run_idx}.json"
                with open(loss_save_path, "w") as f:
                    json.dump(loss_history, f)
                logger.info(f"[run={run_idx}] Loss history saved to {loss_save_path}")
                with open(guided_params_path, "wb") as f:
                    pickle.dump(log_ratio_spec.params, f)
                logger.info(f"[run={run_idx}] Log-ratio model params saved to {guided_params_path}")

            log_ratio_apply = log_ratio_spec.log_ratio_apply
            log_ratio_params = log_ratio_spec.params

        # ------------------------------------------------------------------
        # 3b. Initialise and train KLIEP log-ratio model (only if needed)
        # ------------------------------------------------------------------
        if "kliep_guided" in run_methods:
            kliep_params_path = kliep_guide_path / f"params_run_{run_idx}.pkl"
            key, key_init = jax.random.split(key)
            kliep_spec = init_log_ratio_model(
                model,
                key_init,
                reinit_output_head=args.reinit_output_head,
                model_type=args.log_ratio_model_type,
                mlp_hidden_dim=args.log_ratio_mlp_hidden_dim,
                mlp_num_layers=args.log_ratio_mlp_num_layers,
                mlp_time_embedding_dim=args.log_ratio_mlp_time_embedding_dim,
                mlp_activation=args.log_ratio_mlp_activation,
                freeze_embedding=not args.log_ratio_finetune_embedding,
            )
            if args.load_cached_params and kliep_params_path.exists():
                logger.info(f"[run={run_idx}] Loading cached KLIEP model from {kliep_params_path}")
                with open(kliep_params_path, "rb") as f:
                    cached_params = pickle.load(f)
                kliep_spec = LogRatioModelSpec(
                    log_ratio_apply=kliep_spec.log_ratio_apply,
                    params=cached_params,
                )
            else:
                if args.log_ratio_model_type == "transformer":
                    logger.info(f"[run={run_idx}] KLIEP log-ratio transformer initialised from score model backbone.")
                else:
                    logger.info(f"[run={run_idx}] KLIEP log-ratio MLP initialised with random parameters.")

                key, key_train = jax.random.split(key)
                logger.info(
                    f"[run={run_idx}] Training KLIEP log-ratio model for up to {args.train_epochs} epochs "
                    f"(batch_size={args.train_batch_size}, lr={args.train_lr}, patience={args.train_patience})"
                )
                kliep_spec, kliep_loss_history = train_log_ratio_model_kliep(
                    task=task,
                    diffusion_model=model,
                    condition_mask=model.condition_mask,
                    model_spec=kliep_spec,
                    rng=key_train,
                    batch_size=args.train_batch_size,
                    epochs=args.train_epochs,
                    lr=args.train_lr,
                    patience=args.train_patience,
                    dataset=base_dataset,
                    softmax_t=args.softmax_t,
                    n_time_samples=args.n_time_samples,
                    beta_start=args.beta_start,
                    beta_end=args.beta_end,
                    beta_warmup_frac=args.beta_warmup_frac,
                    beta_schedule=args.beta_schedule,
                )
                logger.info(
                    f"[run={run_idx}] KLIEP training complete. Final loss: {kliep_loss_history[-1]:.4f} "
                    f"(over {len(kliep_loss_history)} epochs)"
                )
                kliep_loss_save_path = kliep_guide_path / f"loss_history_run_{run_idx}.json"
                with open(kliep_loss_save_path, "w") as f:
                    json.dump(kliep_loss_history, f)
                logger.info(f"[run={run_idx}] KLIEP loss history saved to {kliep_loss_save_path}")
                with open(kliep_params_path, "wb") as f:
                    pickle.dump(kliep_spec.params, f)
                logger.info(f"[run={run_idx}] KLIEP model params saved to {kliep_params_path}")

            kliep_log_ratio_apply = kliep_spec.log_ratio_apply
            kliep_log_ratio_params = kliep_spec.params

        # ------------------------------------------------------------------
        # 3g. Initialise and train WSM correction model (only if needed)
        # ------------------------------------------------------------------
        if "wsm_guided" in run_methods:
            wsm_params_path = wsm_guide_path / f"params_run_{run_idx}.pkl"
            key, key_wsm_init = jax.random.split(key)
            wsm_spec = init_wsm_model(
                model,
                key_wsm_init,
                hidden_dim=args.log_ratio_mlp_hidden_dim,
                num_layers=args.log_ratio_mlp_num_layers,
                time_embedding_dim=args.log_ratio_mlp_time_embedding_dim,
                activation=args.log_ratio_mlp_activation,
            )
            if args.load_cached_params and wsm_params_path.exists():
                logger.info(f"[run={run_idx}] Loading cached WSM model from {wsm_params_path}")
                with open(wsm_params_path, "rb") as f:
                    cached_params = pickle.load(f)
                wsm_spec = WSMModelSpec(
                    wsm_apply=wsm_spec.wsm_apply,
                    params=cached_params,
                )
            else:
                logger.info(f"[run={run_idx}] WSM correction MLP initialised.")
                key, key_wsm_train = jax.random.split(key)
                logger.info(
                    f"[run={run_idx}] Training WSM model for up to {args.train_epochs} epochs "
                    f"(batch_size={args.train_batch_size}, lr={args.train_lr}, patience={args.train_patience})"
                )
                _wsm_t0 = time.time()
                wsm_spec, wsm_loss_history = train_wsm_model(
                    task=task,
                    diffusion_model=model,
                    condition_mask=model.condition_mask,
                    model_spec=wsm_spec,
                    rng=key_wsm_train,
                    batch_size=args.train_batch_size,
                    epochs=args.train_epochs,
                    lr=args.train_lr,
                    patience=args.train_patience,
                    dataset=base_dataset,
                )
                logger.info(
                    f"[run={run_idx}] WSM training complete. Final loss: {wsm_loss_history[-1]:.4f} "
                    f"(over {len(wsm_loss_history)} epochs)"
                )
                logger.info(f"[run={run_idx}] WSM training time: {time.time() - _wsm_t0:.1f}s")
                wsm_loss_save_path = wsm_guide_path / f"loss_history_run_{run_idx}.json"
                with open(wsm_loss_save_path, "w") as f:
                    json.dump(wsm_loss_history, f)
                logger.info(f"[run={run_idx}] WSM loss history saved to {wsm_loss_save_path}")
                with open(wsm_params_path, "wb") as f:
                    pickle.dump(wsm_spec.params, f)
                logger.info(f"[run={run_idx}] WSM model params saved to {wsm_params_path}")
            wsm_apply_fn = wsm_spec.wsm_apply
            wsm_trained_params = wsm_spec.params

        for obs_seed in eval_obs_seeds:
            obs_path = (
                exp_path
                / "data"
                / "reference"
                / f"{args.task_name}_{args.task_type}"
                / f"obs_{obs_seed}.json"
            )
            obs = json.load(open(obs_path, "r"))
            theta_o = jnp.atleast_1d(jnp.array(obs["theta"]))
            x_o = jnp.atleast_1d(jnp.array(obs["x"]))
            x_o_z = _z_score_observation_if_needed(model, x_o)

            @jax.jit
            def sample_once_priorguide(key):
                x_T = model.sde.marginal_mean(
                    jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                ).reshape(-1) + model.sde.marginal_stddev(
                    jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                ).reshape(-1) * jax.random.normal(key, shape=(theta_dim + x_dim,))

                result = prior_guide_theta_prior_only(
                    model=model,
                    key=key,
                    condition_mask=model.condition_mask,
                    x_o=x_o_z,
                    x_T=x_T,
                    theta_prior_log_weights=target_log_weights_z,
                    theta_prior_mean=target_means_z,
                    theta_prior_cov=target_covs_z,
                    prior_dim=theta_dim,
                    num_steps=args.num_steps,
                    rho=args.rho,
                    langevin_steps=args.priorguide_langevin_steps,
                    langevin_ratio=args.priorguide_langevin_ratio,
                )

                return _un_z_score_latent_if_needed(model, result)

            # Base model, sampled with the same reverse-SDE discretisation
            # (num_steps, rho) as the guided methods, with no guidance term.
            if "simformer" in run_methods:
                @jax.jit
                def sample_once_simformer(key):
                    x_T = model.sde.marginal_mean(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) + model.sde.marginal_stddev(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) * jax.random.normal(key, shape=(theta_dim + x_dim,))
                    return sample_unguided(
                        model=model,
                        key=key,
                        condition_mask=model.condition_mask,
                        x_o=x_o,
                        x_T=x_T,
                        num_steps=args.num_steps,
                        rho=args.rho,
                    )

            # Define per-trained-model sample functions only if model was trained.
            if "guided" in run_methods:
                @jax.jit
                def sample_once_ratio_guide(key):
                    x_T = model.sde.marginal_mean(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) + model.sde.marginal_stddev(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) * jax.random.normal(key, shape=(theta_dim + x_dim,))
                    return prior_guide_with_log_ratio_model(
                        model=model,
                        key=key,
                        condition_mask=model.condition_mask,
                        x_o=x_o,
                        x_T=x_T,
                        log_ratio_apply=log_ratio_apply,
                        log_ratio_params=log_ratio_params,
                        num_steps=args.num_steps,
                        rho=args.rho,
                        guidance_scale=args.guidance_scale,
                    )

            if "kliep_guided" in run_methods:
                @jax.jit
                def sample_once_kliep_guide(key):
                    x_T = model.sde.marginal_mean(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) + model.sde.marginal_stddev(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) * jax.random.normal(key, shape=(theta_dim + x_dim,))
                    return prior_guide_with_log_ratio_model(
                        model=model,
                        key=key,
                        condition_mask=model.condition_mask,
                        x_o=x_o,
                        x_T=x_T,
                        log_ratio_apply=kliep_log_ratio_apply,
                        log_ratio_params=kliep_log_ratio_params,
                        num_steps=args.num_steps,
                        rho=args.rho,
                        guidance_scale=args.guidance_scale,
                    )

            if "wsm_guided" in run_methods:
                @jax.jit
                def sample_once_wsm_guide(key):
                    x_T = model.sde.marginal_mean(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) + model.sde.marginal_stddev(
                        jnp.array([model.T_max]), jnp.zeros((theta_dim + x_dim,))
                    ).reshape(-1) * jax.random.normal(key, shape=(theta_dim + x_dim,))
                    return prior_guide_with_wsm(
                        model=model,
                        key=key,
                        condition_mask=model.condition_mask,
                        x_o=x_o,
                        x_T=x_T,
                        wsm_apply=wsm_apply_fn,
                        wsm_params=wsm_trained_params,
                        num_steps=args.num_steps,
                        rho=args.rho,
                        guidance_scale=args.guidance_scale,
                    )

            key, key_obs = jax.random.split(key)

            # Batch collectors — only for methods being run (exclude simformer_importance,
            # which is computed via importance resampling after all batches are collected).
            batches: dict[str, list] = {m: [] for m in run_methods if m != "simformer_importance"}

            for batch in range(args.num_batches):
                batch_size = args.num_samples // args.num_batches
                logger.info(
                    f"obs_seed={obs_seed} | batch {batch + 1}/{args.num_batches}"
                )

                (
                    key_obs,
                    key_ratio_guide,
                    key_simformer,
                    key_priorguide,
                    key_kliep_guide,
                    key_wsm_guide,
                ) = jax.random.split(key_obs, 6)

                if "guided" in run_methods:
                    keys = jax.random.split(key_ratio_guide, batch_size)
                    _t0 = time.time()
                    samples_full = jax.vmap(sample_once_ratio_guide)(keys)
                    samples_full.block_until_ready()
                    logger.info(f"obs_seed={obs_seed} | batch {batch+1} | guided sampling time: {time.time() - _t0:.1f}s ({batch_size} samples)")
                    batches["guided"].append(jnp.array(samples_full[:, :theta_dim]))

                if "simformer" in run_methods:
                    keys_simformer = jax.random.split(key_simformer, batch_size)
                    samples_simformer_full = jax.vmap(sample_once_simformer)(keys_simformer)
                    batches["simformer"].append(jnp.array(samples_simformer_full[:, :theta_dim]))

                if "priorguide" in run_methods:
                    keys_priorguide = jax.random.split(key_priorguide, batch_size)
                    _t0 = time.time()
                    samples_priorguide_full = jax.vmap(sample_once_priorguide)(keys_priorguide)
                    samples_priorguide_full.block_until_ready()
                    logger.info(f"obs_seed={obs_seed} | batch {batch+1} | priorguide sampling time: {time.time() - _t0:.1f}s ({batch_size} samples)")
                    batches["priorguide"].append(jnp.array(samples_priorguide_full[:, :theta_dim]))

                if "kliep_guided" in run_methods:
                    keys_kliep = jax.random.split(key_kliep_guide, batch_size)
                    samples_kliep_full = jax.vmap(sample_once_kliep_guide)(keys_kliep)
                    batches["kliep_guided"].append(jnp.array(samples_kliep_full[:, :theta_dim]))

                if "wsm_guided" in run_methods:
                    keys_wsm = jax.random.split(key_wsm_guide, batch_size)
                    samples_wsm_full = jax.vmap(sample_once_wsm_guide)(keys_wsm)
                    batches["wsm_guided"].append(jnp.array(samples_wsm_full[:, :theta_dim]))

            # Concatenate all batches.
            collected: dict[str, jnp.ndarray] = {
                m: jnp.concatenate(v, axis=0) for m, v in batches.items()
            }

            # Importance resampling for simformer_importance (uses freshly generated simformer).
            if "simformer_importance" in run_methods:
                key_obs, key_si = jax.random.split(key_obs)
                collected["simformer_importance"], si_weights = _importance_resample_by_ratio(
                    collected["simformer"], log_ratio_fn, theta_dim, key_si
                )

            logger.info(
                f"obs_seed={obs_seed} | run={run_idx} | shapes: "
                + ", ".join(f"{m}={collected[m].shape}" for m in sorted(collected))
            )

            # ------------------------------------------------------------------
            # Save NPZ files (only for methods that were run).
            # ------------------------------------------------------------------
            theta_np = np.array(theta_o)
            x_np = np.array(x_o)

            if "guided" in collected:
                jnp.savez(
                    os.path.join(
                        log_ratio_guide_path,
                        f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz",
                    ),
                    samples=collected["guided"],
                    theta=theta_np,
                    x=x_np,
                )

            if "simformer" in collected:
                jnp.savez(
                    os.path.join(
                        simformer_path,
                        f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz",
                    ),
                    samples=collected["simformer"],
                    theta=theta_np,
                    x=x_np,
                )

            if "simformer_importance" in collected:
                jnp.savez(
                    os.path.join(
                        simformer_importance_path,
                        f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz",
                    ),
                    samples=collected["simformer_importance"],
                    original_samples=collected["simformer"],
                    weights=si_weights,
                    theta=theta_np,
                    x=x_np,
                )

            if "priorguide" in collected:
                jnp.savez(
                    os.path.join(
                        priorguide_path,
                        f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz",
                    ),
                    samples=collected["priorguide"],
                    theta=theta_np,
                    x=x_np,
                )

            if "kliep_guided" in collected:
                jnp.savez(
                    os.path.join(
                        kliep_guide_path,
                        f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz",
                    ),
                    samples=collected["kliep_guided"],
                    theta=theta_np,
                    x=x_np,
                )

            if "wsm_guided" in collected:
                jnp.savez(
                    os.path.join(
                        wsm_guide_path,
                        f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz",
                    ),
                    samples=collected["wsm_guided"],
                    theta=theta_np,
                    x=x_np,
                )

            # ------------------------------------------------------------------
            # 6. Compute metrics — partial update: load existing file, overwrite
            #    only the methods that were just run, preserve the rest.
            # ------------------------------------------------------------------
            ref_path = (
                exp_path
                / "data"
                / "reference"
                / f"{args.task_name}_{args.task_type}"
                / f"ref_{obs_seed}.npz"
            )

            if ref_path.exists():
                ref_samples = np.array(np.load(ref_path)["samples"])

                # Load existing metrics so we can do a partial update.
                metrics_file_path = metrics_path / f"metrics_obs_{obs_seed}_run_{run_idx}.json"
                if metrics_file_path.exists():
                    with open(metrics_file_path) as f:
                        metrics = json.load(f)
                else:
                    metrics = {}

                n_cap = min(len(ref_samples), args.num_samples)
                ref_cap = ref_samples[:n_cap]
                ref_cap_t = torch.tensor(ref_cap, dtype=torch.float32)

                # Map from method key to its samples array (numpy).
                method_samples_map = {
                    m: np.array(collected[m]) if m in collected else None
                    for m in _ALL_METHODS
                }

                for method_key, samples_np in method_samples_map.items():
                    if samples_np is None:
                        continue
                    samples_cap = samples_np[:n_cap]
                    samples_cap_t = torch.tensor(samples_cap, dtype=torch.float32)
                    metrics[method_key] = {
                        "mmd":          compute_mmd(samples_cap, ref_cap),
                        "gskl":         compute_gskl(samples_cap, ref_cap),
                        "wasserstein2": compute_wasserstein2(samples_cap, ref_cap),
                        "c2st":         float(compute_c2st(samples_cap_t, ref_cap_t)),
                        "rmse":         float(compute_rmse(theta_np, samples_np)),
                    }

                with open(metrics_file_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                updated_keys = [k for k in method_samples_map if method_samples_map[k] is not None]
                logger.info(
                    f"obs_seed={obs_seed}, run={run_idx} | metrics saved -> {metrics_file_path} "
                    f"(updated: {updated_keys})\n"
                    + "\n".join(
                        f"  {k}: mmd={metrics[k]['mmd']:.4f}, gskl={metrics[k]['gskl']:.4f}, "
                        f"w2={metrics[k]['wasserstein2']:.4f}, c2st={metrics[k]['c2st']:.4f}, "
                        f"rmse={metrics[k]['rmse']:.4f}"
                        for k in updated_keys
                    )
                )
            else:
                logger.warning(
                    f"obs_seed={obs_seed} | reference samples not found at {ref_path}, "
                    "skipping metrics computation."
                )


if __name__ == "__main__":
    evaluate()
