"""Weighted Score Matching (WSM) guided diffusion for prior adaptation.

Trains an additive correction g(x_t, x_obs, t) ∈ R^{latent_dim} directly in score space.

Training objective (importance-weighted denoising score matching):
    L(g) = E_{t, θ₀, ε}[ w(θ₀) · ||s_base(x_t,t) + g(x_t,x_obs,t) - s*(x_t,θ₀,t)||² ]
where:
    s_base  = score_fn(x_t, t)           [frozen base model score]
    s*      = -ε / std_t                  [per-sample DSM target, latent dims only]
    w(θ₀)  = exp(log q(θ₀)/p(θ₀))       [importance weight]

At sampling time:
    score_guided = score_base + guidance_scale * g(x_t, x_obs, t)
No gradient computation needed (unlike DRE-based methods).
"""

import logging

import jax
import jax.numpy as jnp
from jax.nn import silu, gelu
import haiku as hk
import optax

from priorg.sim.nn.helpers import GaussianFourierEmbedding

_logger = logging.getLogger(__name__)


class WSMCorrectionMLP(hk.Module):
    """MLP outputting additive score correction g ∈ R^{latent_dim}."""

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embedding_dim: int = 4,
        act=silu,
        name: str = "wsm_correction_mlp",
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.act = act

    def __call__(self, t, x, condition_mask, condition_value):
        batch_size = x.shape[0]
        condition_mask = condition_mask.reshape(batch_size, -1)
        latent_indices = jnp.where(condition_mask[0] == 0, size=self.latent_dim)[0]
        observation_indices = jnp.where(condition_mask[0] == 1, size=self.observation_dim)[0]

        latent = jnp.take(x, latent_indices, axis=-1)
        observation = jnp.take(condition_value, observation_indices, axis=-1)
        time_ctx = GaussianFourierEmbedding(self.time_embedding_dim)(t[..., None])

        features = jnp.concatenate([time_ctx, latent, observation], axis=-1)
        hidden_sizes = [self.hidden_dim] * self.num_layers + [self.latent_dim]
        return hk.nets.MLP(hidden_sizes, activation=self.act)(features)  # (batch, latent_dim)


class WSMModelSpec:
    """Container for the trained WSM correction model.

    wsm_apply(params, t, x, condition_mask, condition_value) -> (latent_dim,)
    """

    def __init__(self, wsm_apply, params=None):
        self.wsm_apply = wsm_apply
        self.params = params


def init_wsm_model(
    score_model,
    rng,
    hidden_dim: int = 256,
    num_layers: int = 4,
    time_embedding_dim: int = 4,
    activation: str = "silu",
    condition_mask=None,
):
    """Build and initialise the WSM correction MLP."""
    num_nodes = score_model.model_init_params["num_nodes"]
    if condition_mask is None:
        condition_mask_bool = score_model.condition_mask.astype(bool)
    else:
        condition_mask_bool = jnp.asarray(condition_mask).astype(bool)
    latent_dim = int((~condition_mask_bool).sum())
    observation_dim = int(condition_mask_bool.sum())

    act = silu if activation == "silu" else gelu

    def model_fn(t, x, condition_mask, condition_value):
        return WSMCorrectionMLP(
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embedding_dim=time_embedding_dim,
            act=act,
        )(t, x, condition_mask, condition_value)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(model_fn))

    rng, subkey = jax.random.split(rng)
    dummy_x = jnp.ones((1, num_nodes))
    dummy_cm = condition_mask_bool.reshape(1, num_nodes).astype(jnp.float32)
    dummy_cv = jnp.zeros((1, num_nodes)).at[:, condition_mask_bool].set(1.0)
    params = init_fn(
        subkey,
        jnp.ones((1,)),
        dummy_x,
        dummy_cm,
        dummy_cv,
    )

    def wsm_apply(params, t, x, condition_mask, condition_value):
        """Single-sample wrapper: returns (latent_dim,) correction."""
        out = apply_fn(
            params,
            jnp.atleast_1d(t).reshape(1,),
            x.reshape(1, num_nodes),
            condition_mask.reshape(1, num_nodes),
            condition_value.reshape(1, num_nodes),
        )
        return out.squeeze(0)

    return WSMModelSpec(wsm_apply=wsm_apply, params=params)


def train_wsm_model(
    task,
    diffusion_model,
    condition_mask,
    model_spec: WSMModelSpec,
    rng,
    batch_size: int = 512,
    epochs: int = 5000,
    lr: float = 1e-3,
    patience: int = 0,
    log_density_ratio_fn=None,
    dataset=None,
):
    """Train the WSM correction model.

    Args:
        task:               Prior-shift task (sample_unshifted + log_density_ratio).
        diffusion_model:    AllConditionalScoreModel (provides SDE + z-score info).
        condition_mask:     0/1 array shape (D,).
        model_spec:         WSMModelSpec to train.
        rng:                JAX PRNG key.
        batch_size:         Samples per gradient step.
        epochs / lr / patience: Optimiser settings.
        log_density_ratio_fn: Optional custom log r(θ₀); defaults to task.log_density_ratio.
        dataset:            Optional dict with keys ``theta`` and ``x``.

    Returns:
        (updated_model_spec, loss_history)
    """
    condition_mask = jnp.asarray(condition_mask)
    condition_mask_bool = condition_mask.astype(bool)
    latent_mask = 1 - condition_mask
    latent_mask_bool = latent_mask.astype(bool)
    latent_dim = int(latent_mask_bool.sum())
    latent_indices = jnp.where(latent_mask_bool, size=latent_dim)[0]

    params = model_spec.params
    wsm_apply = model_spec.wsm_apply

    score_fn = diffusion_model._init_score(
        node_id=diffusion_model.node_id,
        condition_mask=condition_mask,
        edge_mask=None,
        meta_data=None,
    )

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(params)

    ratio_fn = log_density_ratio_fn if log_density_ratio_fn is not None else task.log_density_ratio

    @jax.jit
    def update_step(params, opt_state, x_t, x_obs_full, log_ratios, target_score_latent, t):
        def loss_fn(params):
            x_cond = condition_mask * x_obs_full + latent_mask * x_t  # (B, D)

            # Base score (frozen - no gradient through base model)
            base_score_full = jax.vmap(lambda xi: score_fn(t, xi))(x_cond)  # (B, D)
            base_score_latent = jax.lax.stop_gradient(
                jnp.take(base_score_full, latent_indices, axis=-1)
            )  # (B, latent_dim)

            # WSM correction
            def apply_single(x_t_i, x_obs_i):
                x_ci = condition_mask * x_obs_i + latent_mask * x_t_i
                return wsm_apply(params, jnp.atleast_1d(t), x_ci, condition_mask, x_obs_i)

            correction = jax.vmap(apply_single)(x_t, x_obs_full)  # (B, latent_dim)

            # Importance-weighted MSE loss
            iw = jnp.exp(jnp.clip(log_ratios, -50.0, 50.0))  # (B,)
            residual = base_score_latent + correction - target_score_latent  # (B, latent_dim)
            per_sample = jnp.sum(residual ** 2, axis=-1)  # (B,)
            return jnp.mean(iw * per_sample)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    loss_history = []
    best_loss = float("inf")
    counter = 0
    early_stopping_enabled = patience is not None and patience > 0

    for epoch in range(epochs):
        rng, key_t, key_data, key_eps = jax.random.split(rng, 4)

        t_val = float(jax.random.uniform(
            key_t, minval=diffusion_model.T_min, maxval=diffusion_model.T_max
        ))
        t_arr = jnp.array([t_val])

        # Sample from reference prior
        if dataset is None:
            thetas, x_obs = task.sample_unshifted(batch_size, key=key_data)
            thetas = jnp.asarray(thetas)
            x_obs = jnp.asarray(x_obs)
        else:
            theta_all = jnp.asarray(dataset["theta"])
            x_all = jnp.asarray(dataset["x"])
            idx = jax.random.randint(key_data, (batch_size,), 0, theta_all.shape[0])
            thetas = theta_all[idx]
            x_obs = x_all[idx]

        x_full = jnp.concatenate([thetas, x_obs], axis=-1)
        x_obs_full = jnp.zeros_like(x_full)
        x_obs_full = x_obs_full.at[:, condition_mask_bool].set(x_full[:, condition_mask_bool])

        if diffusion_model.z_score_params is not None:
            z_score_fn = diffusion_model.z_score_params["z_score_fn"]
            x_full = z_score_fn(x_full, diffusion_model.node_id)
            x_obs_full = jnp.zeros_like(x_full)
            x_obs_full = x_obs_full.at[:, condition_mask_bool].set(x_full[:, condition_mask_bool])

        # Noise latent dimensions; compute DSM target score
        eps = jax.random.normal(key_eps, shape=x_full.shape)
        mean_t = diffusion_model.sde.marginal_mean(t_arr, x_full)  # (B, D)
        std_t = diffusion_model.sde.marginal_stddev(t_arr, x_full)  # (B, D)

        x_t = x_full.copy()
        x_t = x_t.at[:, latent_mask_bool].set(
            mean_t[:, latent_mask_bool] + std_t[:, latent_mask_bool] * eps[:, latent_mask_bool]
        )

        # DSM target: ∇_xt log p(xt|x0) = -ε / std_t  (latent dims only)
        target_score_latent = -eps[:, latent_mask_bool] / (std_t[:, latent_mask_bool] + 1e-8)

        log_ratios = jnp.asarray(ratio_fn(thetas))  # (B,)

        params, opt_state, loss = update_step(
            params, opt_state, x_t, x_obs_full, log_ratios, target_score_latent, t_arr[0]
        )

        l = float(loss)
        loss_history.append(l)

        if early_stopping_enabled:
            if l < best_loss:
                best_loss = l
                counter = 0
            else:
                counter += 1

        if epoch % 100 == 0:
            _logger.info(f"[wsm] Epoch {epoch}: loss={l:.4f}")

        if early_stopping_enabled and counter >= patience:
            _logger.info(f"[wsm] Early stopping at epoch {epoch}")
            break

    if loss_history:
        _logger.info(f"[wsm] Done. Epoch {len(loss_history) - 1}: loss={loss_history[-1]:.4f}")

    return WSMModelSpec(wsm_apply=wsm_apply, params=params), loss_history


def prior_guide_with_wsm(
    model,
    key,
    condition_mask,
    x_o,
    x_T,
    wsm_apply,
    wsm_params,
    num_steps: int = 500,
    rho: float = 2.0,
    guidance_scale: float = 1.0,
    clipping_range=None,
):
    """Guided reverse diffusion using the trained WSM correction model.

    The guided score is:
        score_guided(x_t, t) = score_base(x_t, t) + guidance_scale * g(x_t, x_obs, t)

    Args:
        model:          AllConditionalScoreModel.
        key:            JAX PRNG key.
        condition_mask: 0/1 array shape (D,).
        x_o:            (obs_dim,) clean observation.
        x_T:            (D,) initial noisy state.
        wsm_apply:      wsm_apply from WSMModelSpec.
        wsm_params:     trained params.
        num_steps:      reverse diffusion steps.
        rho:            time schedule exponent.
        guidance_scale: scalar multiplier for correction term.
        clipping_range: optional (low, high) clip for guided score.

    Returns:
        (D,) array of samples in original (un-z-scored) space.
    """
    condition_mask_bool = jnp.asarray(condition_mask, dtype=bool)
    condition_mask_f = condition_mask_bool.astype(x_T.dtype)
    latent_mask = 1.0 - condition_mask_f
    cond_size = x_o.shape[0]
    cond_idx = jnp.nonzero(condition_mask_bool, size=cond_size)[0]

    if model.z_score_params is not None:
        z_score_fn = model.z_score_params["z_score_fn"]
        cond_node_id = model.node_id[cond_idx]
        x_o = z_score_fn(x_o, cond_node_id)

    ts = jnp.linspace(model.T_min, model.T_max, num_steps) ** rho
    x_o_full = jnp.zeros_like(x_T)
    x_o_full = x_o_full.at[cond_idx].set(x_o)

    latent_dim = x_T.shape[0] - cond_size
    latent_indices = jnp.nonzero(~condition_mask_bool, size=latent_dim)[0]

    score_fn = model._init_score(
        node_id=model.node_id,
        condition_mask=condition_mask_f,
        edge_mask=None,
        meta_data=None,
    )

    def compute_guided_score(x, t):
        x_cond = condition_mask_f * x_o_full + latent_mask * x
        base_score = score_fn(t, x_cond)

        correction = wsm_apply(
            wsm_params, jnp.atleast_1d(t), x_cond, condition_mask_f, x_o_full
        )  # (latent_dim,)

        correction_full = jnp.zeros_like(x_cond).at[latent_indices].set(correction)
        guided = base_score + guidance_scale * correction_full

        if clipping_range is not None:
            guided = jnp.clip(guided, clipping_range[0], clipping_range[1])

        return guided

    def scan_fn(carry, t0):
        key, t1, x1 = carry
        dt = t0 - t1
        key, subkey = jax.random.split(key)
        guided_score = compute_guided_score(x1, t1)

        drift = latent_mask * (
            model.sde.drift(t1, x1) - model.sde.diffusion(t1, x1) ** 2 * guided_score
        )
        diffusion_coef = latent_mask * model.sde.diffusion(t1, x1)

        x0 = (
            x1
            + drift * dt
            + diffusion_coef * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape)
        )
        x0 = condition_mask_f * x_o_full + latent_mask * x0
        return (key, t0, x0), x0

    carry_init = (key, ts[-1], x_T)
    _, traj = jax.lax.scan(scan_fn, carry_init, ts[::-1][1:])
    result = traj[-1]

    if model.z_score_params is not None:
        un_z_score_fn = model.z_score_params["un_z_score_fn"]
        latent_node_id = model.node_id[latent_indices]
        latent_vals = un_z_score_fn(result[latent_indices], latent_node_id)
        result = result.at[latent_indices].set(latent_vals)

    return result
