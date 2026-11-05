"""
Run log-ratio guided diffusion for posterior predictive.

Conditions on the first 1/3 of obs dims; treats para + last 2/3 obs as jointly latent.
The log-ratio weight is still r(theta) = p_target(theta) / p_ref(theta) (prior ratio only).
All guidance methods from run_guidance.py are supported.

Pipeline:
  1. Load pretrained score model.
  2. Build Task using default reference / target priors from get_task().
  3. Construct pp_condition_mask: obs[:cond_obs_dim] conditioned, para + obs[cond_obs_dim:] latent.
  4. Initialise a log-ratio model from the score model backbone.
  5. Train the log-ratio model with pp_condition_mask
     (noises para + last 2/3 obs together; target = prior ratio on para only).
  6. For each evaluation observation:
       a. Use x_o[:cond_obs_dim] as conditioning input.
       b. Sample with log-ratio guidance, exact PriorGuide, MC-ratio guide, and Simformer.
       c. Save samples  (shape N x (theta_dim + pred_obs_dim)) to NPZ.

Use --methods to run only a subset of methods.  The metrics file is loaded from
disk (if it exists) and only the selected methods' entries are updated.
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
from priorg.sim.methods.metrics import compute_rmse

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

    # torch.MixtureSameFamily uses "mixing_distribution";
    # pyro.distributions.MixtureSameFamily uses "mixture_distribution".
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


def _gmm_moments(log_weights, means, covs):
    weights = jax.nn.softmax(log_weights)
    mean = jnp.sum(weights[:, None] * means, axis=0)
    centered = means - mean[None, :]
    cov = jnp.sum(
        weights[:, None, None]
        * (covs + centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    return mean, cov


def _z_score_gaussian_mixture(model, log_weights, means, covs, node_id):
    """Z-score prior GMM parameters using the z-score params for the given node_id."""
    if model.z_score_params is None:
        return log_weights, means, covs

    mean_per_node = jnp.asarray(
        model.z_score_params["mean_per_node_id"][node_id]
    ).reshape(-1)
    std_per_node = jnp.asarray(
        model.z_score_params["std_per_node_id"][node_id]
    ).reshape(-1)

    means_z = (means - mean_per_node[None, :]) / std_per_node[None, :]
    covs_z = covs / (std_per_node[None, :, None] * std_per_node[None, None, :])
    return log_weights, means_z, covs_z


def _z_score_covariance(model, cov, node_id):
    if model.z_score_params is None:
        return cov

    std_per_node = jnp.asarray(
        model.z_score_params["std_per_node_id"][node_id]
    ).reshape(-1)
    return cov / (std_per_node[:, None] * std_per_node[None, :])


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
        description="Run log-ratio guided diffusion for posterior predictive"
    )
    parser.add_argument(
        "--sde_name",
        type=str,
        default=None,
        choices=["vpsde", "vesde"],
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="oup",
        choices=["oup", "turin"],
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
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
    )
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_runs", type=int, default=1)
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
            "model training dataset by sampling from the base model conditioned "
            "on x values drawn from the base training set. 0 = no augmentation (default: 420000)."
        ),
    )
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--train_epochs", type=int, default=8000)
    parser.add_argument("--train_lr", type=float, default=3e-4)
    parser.add_argument("--train_patience", type=int, default=0)
    parser.add_argument(
        "--load_cached_params",
        action="store_true",
        default=False,
        help="Load previously saved model params instead of retraining (default: False)",
    )
    parser.add_argument("--reinit_output_head", action="store_true", default=True)
    parser.add_argument(
        "--log_ratio_model_type",
        type=str,
        default="mlp",
        choices=["transformer", "mlp"],
    )
    parser.add_argument("--log_ratio_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--log_ratio_mlp_num_layers", type=int, default=4)
    parser.add_argument("--log_ratio_mlp_time_embedding_dim", type=int, default=4)
    parser.add_argument(
        "--log_ratio_mlp_activation",
        type=str,
        default="silu",
        choices=["silu", "gelu"],
    )
    parser.add_argument("--seed", type=int, default=0)
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
        "--num_obs",
        type=int,
        default=None,
        help="Evaluate only the first N observation seeds (default: all 10). "
             "Handy for smoke tests.",
    )
    parser.add_argument(
        "--priorguide_gmm_iters",
        type=int,
        default=20000,
        help="Optimisation steps for the PriorGuide GMM fit (default: 20000). "
             "Lower it for smoke tests.",
    )
    parser.add_argument("--priorguide_langevin_steps", type=int, default=4)
    parser.add_argument("--priorguide_langevin_ratio", type=float, default=0.5)
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

    args = parser.parse_args()

    eval_obs_seeds = (
        observation_seeds if args.num_obs is None
        else observation_seeds[: args.num_obs]
    )

    run_methods = set(args.methods)
    # simformer_importance derives from simformer samples.
    if "simformer_importance" in run_methods:
        run_methods.add("simformer")

    logger.info(
        f"Posterior predictive | task={args.task_name}, type={args.task_type}, "
        f"model_id={args.model_id}, num_samples={args.num_samples}, "
        f"num_steps={args.num_steps}, rho={args.rho}, guidance_scale={args.guidance_scale}, "
        f"train_epochs={args.train_epochs}, train_lr={args.train_lr}, seed={args.seed}, "
        f"log_ratio_model_type={args.log_ratio_model_type}, "
        f"methods={sorted(run_methods)}"
    )

    exp_path = Path(__file__).parent.parent
    results_path = exp_path.parent / "results"

    model_path = exp_path / "models" / args.task_name / args.task_type / f"model_{args.model_id}.pkl"

    pp_guidance_path = (
        results_path / "pp_guidance" / args.task_name / args.task_type / f"model_{args.model_id}"
    )
    pp_simformer_path = (
        results_path / "pp_simformer" / args.task_name / args.task_type / f"model_{args.model_id}"
    )
    pp_simformer_importance_path = (
        results_path / "pp_simformer_importance" / args.task_name / args.task_type / f"model_{args.model_id}"
    )
    pp_priorguide_path = (
        results_path / "pp_priorguide" / args.task_name / args.task_type / f"model_{args.model_id}"
    )
    pp_kliep_guide_path = (
        results_path / "pp_kliep_guidance" / args.task_name / args.task_type / f"model_{args.model_id}"
    )
    pp_wsm_guide_path = (
        results_path / "pp_wsm_guidance" / args.task_name / args.task_type / f"model_{args.model_id}"
    )
    pp_metrics_path = (
        results_path / "metrics_pp" / f"{args.task_name}_{args.task_type}" / f"model_{args.model_id}"
    )
    for p in [
        pp_guidance_path,
        pp_simformer_path,
        pp_simformer_importance_path,
        pp_priorguide_path,
        pp_kliep_guide_path,
        pp_wsm_guide_path,
        pp_metrics_path,
    ]:
        os.makedirs(p, exist_ok=True)

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
                f"Checkpoint SDE '{ckpt_sde_name}' does not match --sde_name '{args.sde_name}'."
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

    # ------------------------------------------------------------------
    # 2. Build task and posterior predictive condition mask
    # ------------------------------------------------------------------
    task = get_task(f"{args.task_name}_{args.task_type}")
    theta_dim = task.theta_dim
    x_dim = task.x_dim

    # Condition on the first 1/3 of obs dims; last 2/3 obs + para are jointly latent.
    cond_obs_dim = x_dim // 3
    pred_obs_dim = x_dim - cond_obs_dim
    D = theta_dim + x_dim

    pp_condition_mask_bool = np.zeros(D, dtype=bool)
    pp_condition_mask_bool[theta_dim : theta_dim + cond_obs_dim] = True
    pp_condition_mask = jnp.array(pp_condition_mask_bool, dtype=jnp.float32)
    pp_latent_mask_bool = ~pp_condition_mask_bool  # para + last 1/3 obs

    logger.info(
        f"Task: theta_dim={theta_dim}, x_dim={x_dim}, "
        f"cond_obs_dim={cond_obs_dim}, pred_obs_dim={pred_obs_dim}"
    )

    # para node IDs — same as the model's original latent node IDs.
    # Used for z-scoring the prior (prior is on para only).
    para_mask_bool = ~model.condition_mask.astype(bool)
    para_node_id = model.node_id[para_mask_bool]
    pp_cond_node_id = model.node_id[pp_condition_mask_bool]
    pp_latent_node_id = model.node_id[pp_latent_mask_bool]

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
            key_aug = jax.random.PRNGKey(args.seed + 99999)
            base_dataset = build_augmented_dataset(
                base_dataset=base_dataset,
                model=model,
                condition_mask=model.condition_mask,
                theta_dim=theta_dim,
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
    # 3. Prepare prior parameters (for exact PriorGuide, z-scored with para node IDs)
    # ------------------------------------------------------------------
    log_ratio_fn = task.make_jax_log_density_ratio_fn()

    ref_log_weights, ref_means, ref_covs = _extract_gmm_from_distribution(
        task.get_reference_prior()
    )
    _, reference_prior_cov = _gmm_moments(ref_log_weights, ref_means, ref_covs)
    reference_prior_cov_z = _z_score_covariance(model, reference_prior_cov, para_node_id)

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
            para_node_id,
        )
        logger.info(
            f"PriorGuide GMM fit complete: {target_means_z.shape[0]} components."
        )


    # Un-z-score para + pred_obs (pp latent) dims; obs-independent so defined once.
    def _un_z_score_pp_latent(result):
        if model.z_score_params is None:
            return result
        un_z = model.z_score_params["un_z_score_fn"]
        latent_vals = result[pp_latent_mask_bool]
        latent_vals = un_z(latent_vals, pp_latent_node_id)
        return result.at[pp_latent_mask_bool].set(latent_vals)

    # ------------------------------------------------------------------
    # 6. Per-run training and per-observation sampling
    # ------------------------------------------------------------------
    for run_idx in range(args.run_start_idx, args.run_start_idx + args.num_runs):
        key = jax.random.PRNGKey(args.seed + run_idx)
        # ------------------------------------------------------------------
        # 4. Initialise and train log-ratio model (only if needed)
        # ------------------------------------------------------------------
        if "guided" in run_methods:
            guided_params_path = pp_guidance_path / f"params_run_{run_idx}.pkl"
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
                condition_mask=pp_condition_mask,
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
                logger.info(f"[run={run_idx}] Log-ratio model ({args.log_ratio_model_type}) initialised.")
                key, key_train = jax.random.split(key)
                logger.info(
                    f"[run={run_idx}] Training log-ratio model for up to {args.train_epochs} epochs "
                    f"(batch_size={args.train_batch_size}, lr={args.train_lr}, patience={args.train_patience})"
                )
                log_ratio_spec, loss_history = train_log_ratio_model(
                    task=task,
                    diffusion_model=model,
                    condition_mask=pp_condition_mask,
                    model_spec=log_ratio_spec,
                    rng=key_train,
                    batch_size=args.train_batch_size,
                    epochs=args.train_epochs,
                    lr=args.train_lr,
                    patience=args.train_patience,
                    dataset=base_dataset,
                )
                logger.info(
                    f"[run={run_idx}] Training done. Final loss: {loss_history[-1]:.4f} ({len(loss_history)} epochs)"
                )
                loss_save_path = pp_guidance_path / f"loss_history_run_{run_idx}.json"
                with open(loss_save_path, "w") as f:
                    json.dump(loss_history, f)
                logger.info(f"[run={run_idx}] Loss history saved to {loss_save_path}")
                with open(guided_params_path, "wb") as f:
                    pickle.dump(log_ratio_spec.params, f)
                logger.info(f"[run={run_idx}] Log-ratio model params saved to {guided_params_path}")

            log_ratio_apply = log_ratio_spec.log_ratio_apply
            log_ratio_params = log_ratio_spec.params

        # ------------------------------------------------------------------
        # 4b. Initialise and train KLIEP log-ratio model (only if needed)
        # ------------------------------------------------------------------
        if "kliep_guided" in run_methods:
            kliep_params_path = pp_kliep_guide_path / f"params_run_{run_idx}.pkl"
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
                condition_mask=pp_condition_mask,
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
                logger.info(f"[run={run_idx}] KLIEP log-ratio model ({args.log_ratio_model_type}) initialised.")
                key, key_train = jax.random.split(key)
                logger.info(
                    f"[run={run_idx}] Training KLIEP log-ratio model for up to {args.train_epochs} epochs "
                    f"(batch_size={args.train_batch_size}, lr={args.train_lr}, patience={args.train_patience})"
                )
                kliep_spec, kliep_loss_history = train_log_ratio_model_kliep(
                    task=task,
                    diffusion_model=model,
                    condition_mask=pp_condition_mask,
                    model_spec=kliep_spec,
                    rng=key_train,
                    batch_size=args.train_batch_size,
                    epochs=args.train_epochs,
                    lr=args.train_lr,
                    patience=args.train_patience,
                    dataset=base_dataset,
                )
                logger.info(
                    f"[run={run_idx}] KLIEP training done. Final loss: {kliep_loss_history[-1]:.4f} "
                    f"({len(kliep_loss_history)} epochs)"
                )
                kliep_loss_save_path = pp_kliep_guide_path / f"loss_history_run_{run_idx}.json"
                with open(kliep_loss_save_path, "w") as f:
                    json.dump(kliep_loss_history, f)
                logger.info(f"[run={run_idx}] KLIEP loss history saved to {kliep_loss_save_path}")
                with open(kliep_params_path, "wb") as f:
                    pickle.dump(kliep_spec.params, f)
                logger.info(f"[run={run_idx}] KLIEP model params saved to {kliep_params_path}")

            kliep_log_ratio_apply = kliep_spec.log_ratio_apply
            kliep_log_ratio_params = kliep_spec.params

        # ------------------------------------------------------------------
        # 5f. Initialise and train WSM correction model (only if needed)
        # ------------------------------------------------------------------
        if "wsm_guided" in run_methods:
            wsm_params_path = pp_wsm_guide_path / f"params_run_{run_idx}.pkl"
            key, key_init = jax.random.split(key)
            wsm_spec = init_wsm_model(
                model,
                key_init,
                hidden_dim=args.log_ratio_mlp_hidden_dim,
                num_layers=args.log_ratio_mlp_num_layers,
                time_embedding_dim=args.log_ratio_mlp_time_embedding_dim,
                activation=args.log_ratio_mlp_activation,
                condition_mask=pp_condition_mask,
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
                key, key_train = jax.random.split(key)
                logger.info(
                    f"[run={run_idx}] Training WSM correction model for up to {args.train_epochs} epochs "
                    f"(batch_size={args.train_batch_size}, lr={args.train_lr}, patience={args.train_patience})"
                )
                wsm_spec, wsm_loss_history = train_wsm_model(
                    task=task,
                    diffusion_model=model,
                    condition_mask=pp_condition_mask,
                    model_spec=wsm_spec,
                    rng=key_train,
                    batch_size=args.train_batch_size,
                    epochs=args.train_epochs,
                    lr=args.train_lr,
                    patience=args.train_patience,
                    dataset=base_dataset,
                )
                logger.info(
                    f"[run={run_idx}] WSM training done. Final loss: {wsm_loss_history[-1]:.4f} "
                    f"({len(wsm_loss_history)} epochs)"
                )
                wsm_loss_save_path = pp_wsm_guide_path / f"loss_history_run_{run_idx}.json"
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
            theta_o = jnp.atleast_1d(jnp.array(obs["theta"]))   # (theta_dim,)
            x_o = jnp.atleast_1d(jnp.array(obs["x"]))           # (x_dim,)

            # Conditioning input: first 1/3 of obs dims.
            x_o_cond = x_o[:cond_obs_dim]  # (cond_obs_dim,)

            # Pre-z-score for PriorGuide / MC methods (they expect z-scored obs).
            if model.z_score_params is not None:
                z_score_fn = model.z_score_params["z_score_fn"]
                x_o_cond_z = z_score_fn(x_o_cond, pp_cond_node_id)
            else:
                x_o_cond_z = x_o_cond

            def _init_x_T(key):
                """Sample the initial noisy state x_T (full D-dim)."""
                return (
                    model.sde.marginal_mean(
                        jnp.array([model.T_max]), jnp.zeros((D,))
                    ).reshape(-1)
                    + model.sde.marginal_stddev(
                        jnp.array([model.T_max]), jnp.zeros((D,))
                    ).reshape(-1)
                    * jax.random.normal(key, shape=(D,))
                )

            @jax.jit
            def sample_once_priorguide(key):
                result = prior_guide_theta_prior_only(
                    model=model,
                    key=key,
                    condition_mask=pp_condition_mask,
                    x_o=x_o_cond_z,
                    x_T=_init_x_T(key),
                    theta_prior_log_weights=target_log_weights_z,
                    theta_prior_mean=target_means_z,
                    theta_prior_cov=target_covs_z,
                    prior_dim=theta_dim,
                    theta_original_prior_cov=reference_prior_cov_z,
                    num_steps=args.num_steps,
                    rho=args.rho,
                    langevin_steps=args.priorguide_langevin_steps,
                    langevin_ratio=args.priorguide_langevin_ratio,
                )
                return _un_z_score_pp_latent(result)

            # Base model, sampled with the same reverse-SDE discretisation
            # (num_steps, rho) as the guided methods, with no guidance term.
            if "simformer" in run_methods:
                @jax.jit
                def sample_once_simformer(key):
                    return sample_unguided(
                        model=model,
                        key=key,
                        condition_mask=pp_condition_mask,
                        x_o=x_o_cond,
                        x_T=_init_x_T(key),
                        num_steps=args.num_steps,
                        rho=args.rho,
                    )

            # Define per-trained-model sample functions only if model was trained.
            if "guided" in run_methods:
                @jax.jit
                def sample_once_ratio_guide(key):
                    # prior_guide_with_log_ratio_model z-scores x_o_cond internally.
                    return prior_guide_with_log_ratio_model(
                        model=model,
                        key=key,
                        condition_mask=pp_condition_mask,
                        x_o=x_o_cond,
                        x_T=_init_x_T(key),
                        log_ratio_apply=log_ratio_apply,
                        log_ratio_params=log_ratio_params,
                        num_steps=args.num_steps,
                        rho=args.rho,
                        guidance_scale=args.guidance_scale,
                    )

            if "kliep_guided" in run_methods:
                @jax.jit
                def sample_once_kliep_guide(key):
                    return prior_guide_with_log_ratio_model(
                        model=model,
                        key=key,
                        condition_mask=pp_condition_mask,
                        x_o=x_o_cond,
                        x_T=_init_x_T(key),
                        log_ratio_apply=kliep_log_ratio_apply,
                        log_ratio_params=kliep_log_ratio_params,
                        num_steps=args.num_steps,
                        rho=args.rho,
                        guidance_scale=args.guidance_scale,
                    )

            if "wsm_guided" in run_methods:
                @jax.jit
                def sample_once_wsm_guide(key):
                    return prior_guide_with_wsm(
                        model=model,
                        key=key,
                        condition_mask=pp_condition_mask,
                        x_o=x_o_cond,
                        x_T=_init_x_T(key),
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
                logger.info(f"obs_seed={obs_seed} | run={run_idx} | batch {batch + 1}/{args.num_batches}")

                (
                    key_obs,
                    key_ratio,
                    key_simformer,
                    key_priorguide,
                    key_kliep_guide,
                    key_wsm_guide,
                ) = jax.random.split(key_obs, 6)

                if "guided" in run_methods:
                    keys = jax.random.split(key_ratio, batch_size)
                    samples_full = jax.vmap(sample_once_ratio_guide)(keys)  # (N, D)
                    samples_latent = jnp.concatenate(
                        [samples_full[:, :theta_dim], samples_full[:, theta_dim + cond_obs_dim:]],
                        axis=-1,
                    )
                    batches["guided"].append(samples_latent)

                if "simformer" in run_methods:
                    keys_simformer = jax.random.split(key_simformer, batch_size)
                    samples_sf_full = jax.vmap(sample_once_simformer)(keys_simformer)  # (N, D)
                    samples_latent_sf = jnp.concatenate(
                        [samples_sf_full[:, :theta_dim], samples_sf_full[:, theta_dim + cond_obs_dim:]],
                        axis=-1,
                    )
                    batches["simformer"].append(samples_latent_sf)

                if "priorguide" in run_methods:
                    keys_priorguide = jax.random.split(key_priorguide, batch_size)
                    samples_pg_full = jax.vmap(sample_once_priorguide)(keys_priorguide)  # (N, D)
                    samples_latent_pg = jnp.concatenate(
                        [samples_pg_full[:, :theta_dim], samples_pg_full[:, theta_dim + cond_obs_dim:]],
                        axis=-1,
                    )
                    batches["priorguide"].append(samples_latent_pg)

                if "kliep_guided" in run_methods:
                    keys_kliep = jax.random.split(key_kliep_guide, batch_size)
                    samples_kliep_full = jax.vmap(sample_once_kliep_guide)(keys_kliep)  # (N, D)
                    samples_latent_kliep = jnp.concatenate(
                        [samples_kliep_full[:, :theta_dim], samples_kliep_full[:, theta_dim + cond_obs_dim:]],
                        axis=-1,
                    )
                    batches["kliep_guided"].append(samples_latent_kliep)

                if "wsm_guided" in run_methods:
                    keys_wsm = jax.random.split(key_wsm_guide, batch_size)
                    samples_wsm_full = jax.vmap(sample_once_wsm_guide)(keys_wsm)  # (N, D)
                    samples_latent_wsm = jnp.concatenate(
                        [samples_wsm_full[:, :theta_dim], samples_wsm_full[:, theta_dim + cond_obs_dim:]],
                        axis=-1,
                    )
                    batches["wsm_guided"].append(samples_latent_wsm)

            # Concatenate all batches.
            collected: dict[str, jnp.ndarray] = {
                m: jnp.concatenate(v, axis=0) for m, v in batches.items()
            }

            # Importance resampling (uses freshly generated simformer samples).
            if "simformer_importance" in run_methods:
                key_obs, key_si = jax.random.split(key_obs)
                collected["simformer_importance"], si_weights = _importance_resample_by_ratio(
                    collected["simformer"], log_ratio_fn, theta_dim, key_si
                )

            logger.info(
                f"obs_seed={obs_seed} | run={run_idx} | latent shape (para+pred_obs): "
                + ", ".join(f"{m}={collected[m].shape}" for m in sorted(collected))
            )

            # ------------------------------------------------------------------
            # Save NPZ files (only for methods that were run).
            # ------------------------------------------------------------------
            shared_meta = dict(theta=theta_o, x=x_o, cond_obs_dim=cond_obs_dim)

            if "guided" in collected:
                jnp.savez(
                    os.path.join(pp_guidance_path, f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz"),
                    samples=collected["guided"],
                    **shared_meta,
                )
            if "simformer" in collected:
                jnp.savez(
                    os.path.join(pp_simformer_path, f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz"),
                    samples=collected["simformer"],
                    **shared_meta,
                )
            if "simformer_importance" in collected:
                jnp.savez(
                    os.path.join(pp_simformer_importance_path, f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz"),
                    samples=collected["simformer_importance"],
                    original_samples=collected["simformer"],
                    weights=si_weights,
                    **shared_meta,
                )
            if "priorguide" in collected:
                jnp.savez(
                    os.path.join(pp_priorguide_path, f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz"),
                    samples=collected["priorguide"],
                    **shared_meta,
                )
            if "kliep_guided" in collected:
                jnp.savez(
                    os.path.join(pp_kliep_guide_path, f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz"),
                    samples=collected["kliep_guided"],
                    **shared_meta,
                )
            if "wsm_guided" in collected:
                jnp.savez(
                    os.path.join(pp_wsm_guide_path, f"samples_{args.num_samples}_obs_{obs_seed}_run_{run_idx}.npz"),
                    samples=collected["wsm_guided"],
                    **shared_meta,
                )

            # ------------------------------------------------------------------
            # Compute metrics — partial update: load existing file, overwrite
            # only the methods that were just run, preserve the rest.
            # ------------------------------------------------------------------
            pred_obs_truth = np.array(x_o[cond_obs_dim:])

            # Load existing metrics for partial update.
            metrics_file = pp_metrics_path / f"metrics_obs_{obs_seed}_run_{run_idx}.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
            else:
                metrics = {}

            # Map method key → pred_obs samples (the last pred_obs_dim columns).
            for method_key, samples in collected.items():
                pred_obs_samples = np.array(samples[:, theta_dim:])
                metrics[method_key] = {
                    "rmse": float(compute_rmse(pred_obs_truth, pred_obs_samples)),
                }

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            updated_keys = list(collected.keys())
            logger.info(
                f"obs_seed={obs_seed} | run={run_idx} | metrics saved -> {metrics_file} "
                f"(updated: {updated_keys})\n"
                + "\n".join(
                    f"  {k}: rmse={metrics[k]['rmse']:.4f}"
                    for k in updated_keys
                )
            )


if __name__ == "__main__":
    evaluate()
