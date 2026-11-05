import jax
import jax.numpy as jnp
from jax import grad
from jax.nn import gelu, silu

import haiku as hk

from priorg.sim.methods.guidance_priorguide import debug_scan, predict_x0
from priorg.sim.methods.guidance_priorguide import posterior_x0_std
from priorg.sim.nn.embedding_net import XEmbeddingNet, token_condition_mask
from priorg.sim.nn.helpers import GaussianFourierEmbedding
from priorg.sim.nn.tokenizer import ScalarTokenizer
from priorg.sim.nn.transformers import Transformer

# LogRatioModelSpec and the init helpers live in guidance_dre; re-exported here
# because the hk.Module definitions below are what those helpers instantiate.
from priorg.sim.methods.guidance_dre import (
    LogRatioModelSpec,
    init_log_ratio_model,
    init_log_ratio_transformer_from_score_model,
    init_log_ratio_mlp,
    init_log_ratio_mlp_large,
    init_log_ratio_mlp_embedded,
    init_log_ratio_model_from_score_model,
)


class LogRatioTransformer(hk.Module):
    """Transformer backbone with a scalar output head for log-ratio estimation.

    Identical architecture to `scalar_transformer_model` up to and including the
    Transformer, but replaces the per-node `hk.Linear(1)` score head with a
    mean-pool-over-nodes followed by a single `hk.Linear(1)` scalar head.

    This makes the param tree directly compatible with a pretrained score model:
    all sub-module keys (`scalar_tokenizer`, `gaussian_fourier_embedding`,
    `transformer`, `condition_token`) match, only the output head differs
    (`log_ratio_head` vs. `linear`).
    """

    def __init__(
        self,
        num_nodes: int,
        token_dim: int = 40,
        condition_token_dim: int = 10,
        condition_token_init_scale: float = 0.01,
        condition_token_init_mean: float = 0.0,
        condition_mode: str = "concat",
        time_embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        attn_size: int = 5,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act=jax.nn.gelu,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        embed_x: bool = False,
        theta_dim: int = None,
        x_dim: int = None,
        num_x_tokens: int = 1,
        x_embed_hidden=(256, 128),
        x_embed_chunked: bool = False,
        x_embed_layer_norm: bool = True,
        name: str = "log_ratio_transformer",
        **kwargs,  # absorb unused keys from model_init_params (e.g. layer_norm)
    ):
        super().__init__(name=name)
        self.num_nodes = num_nodes
        self.embed_x = embed_x
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.num_x_tokens = num_x_tokens
        self.x_embed_hidden = x_embed_hidden
        self.x_embed_chunked = x_embed_chunked
        self.x_embed_layer_norm = x_embed_layer_norm
        if embed_x and (theta_dim is None or x_dim is None):
            raise ValueError(
                "embed_x=True requires theta_dim and x_dim to be provided."
            )
        self.condition_mode = condition_mode
        self.time_embedding_dim = time_embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor
        self.num_hidden_layers = num_hidden_layers
        self.act = act
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp
        self.cond_init_scale = condition_token_init_scale
        self.cond_init_mean = condition_token_init_mean

        # Mirror the condition_mode token_dim adjustment from scalar_transformer_model.
        if condition_mode == "add":
            token_dim = token_dim + condition_token_dim
            condition_token_dim = token_dim
        elif condition_mode == "none":
            token_dim = token_dim + condition_token_dim
            condition_token_dim = 0
        self.token_dim = token_dim
        self.condition_token_dim = condition_token_dim

    def __call__(self, t, data, data_id, condition_mask):
        _, current_nodes, _ = data.shape
        data_id = data_id.reshape(-1, current_nodes)
        condition_mask = condition_mask.reshape(-1, current_nodes)

        if self.embed_x:
            # Mirrors the embed_x branch of scalar_transformer_model so the
            # parameter tree stays compatible with a pretrained score model.
            tokens = ScalarTokenizer(
                self.token_dim, self.theta_dim + self.num_x_tokens
            )(data_id[:, : self.theta_dim], data[:, : self.theta_dim, :], None)

            x_tokens = XEmbeddingNet(
                token_dim=self.token_dim,
                x_dim=self.x_dim,
                num_x_tokens=self.num_x_tokens,
                hidden_sizes=self.x_embed_hidden,
                chunked=self.x_embed_chunked,
                act=self.act,
                layer_norm=self.x_embed_layer_norm,
            )(data[:, self.theta_dim :, 0])
            x_token_id = hk.get_parameter(
                "x_token_id_embedding",
                shape=[1, self.num_x_tokens, self.token_dim],
                init=hk.initializers.TruncatedNormal(stddev=0.02),
            )
            tokens = jnp.concatenate([tokens, x_tokens + x_token_id], axis=-2)

            condition_mask = token_condition_mask(
                condition_mask,
                self.theta_dim,
                self.x_dim,
                self.num_x_tokens,
                self.x_embed_chunked,
            )
            current_nodes = self.theta_dim + self.num_x_tokens
            time_ctx = GaussianFourierEmbedding(self.time_embedding_dim)(t[..., None])
        else:
            tokens = ScalarTokenizer(self.token_dim, self.num_nodes)(data_id, data)
            time_ctx = GaussianFourierEmbedding(self.time_embedding_dim)(t[..., None])

        if self.condition_mode != "none":
            condition_token = hk.get_parameter(
                "condition_token",
                shape=[1, 1, self.condition_token_dim],
                init=hk.initializers.RandomNormal(self.cond_init_scale, self.cond_init_mean),
            )
            cm = condition_mask.reshape(-1, current_nodes, 1)
            condition_token = cm * condition_token
            if self.condition_mode == "add":
                tokens = tokens + condition_token
            elif self.condition_mode == "concat":
                condition_token = jnp.broadcast_to(
                    condition_token, tokens.shape[:-1] + (self.condition_token_dim,)
                )
                tokens = jnp.concatenate([tokens, condition_token], -1)

        h = Transformer(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            attn_size=self.attn_size,
            widening_factor=self.widening_factor,
            num_hidden_layers=self.num_hidden_layers,
            act=self.act,
            skip_connection_attn=self.skip_connection_attn,
            skip_connection_mlp=self.skip_connection_mlp,
        )(tokens, context=time_ctx)  # (batch, nodes, hidden_dim)

        h_pooled = h.mean(axis=1)  # (batch, hidden_dim) — pool over the node sequence
        return hk.Linear(1, name="log_ratio_head")(h_pooled).squeeze(-1)  # (batch,)


class LogRatioMLP(hk.Module):
    """MLP-based log-ratio estimator over noisy latent state and observation context."""

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embedding_dim: int = 4,
        act=silu,
        name: str = "log_ratio_mlp",
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
        observation_indices = jnp.where(
            condition_mask[0] == 1, size=self.observation_dim
        )[0]

        latent = jnp.take(x, latent_indices, axis=-1)
        observation = jnp.take(condition_value, observation_indices, axis=-1)
        time_ctx = GaussianFourierEmbedding(self.time_embedding_dim)(t[..., None])

        features = jnp.concatenate([time_ctx, latent, observation], axis=-1)
        hidden_sizes = [self.hidden_dim] * self.num_layers + [1]
        out = hk.nets.MLP(hidden_sizes, activation=self.act)(features)
        return out.squeeze(-1)


class LogRatioMLPLarge(hk.Module):
    """MLP log-ratio estimator with a condition encoder for high-dim observations.

    Designed for tasks like Turin where the observation dimension is large.
    A small ConditionEncoder first compresses the raw observation into a
    lower-dimensional embedding before feeding into a wider/deeper MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        cond_emb_dim: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 6,
        time_embedding_dim: int = 4,
        act=silu,
        name: str = "log_ratio_mlp_large",
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.cond_emb_dim = cond_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.act = act

    def __call__(self, t, x, condition_mask, condition_value):
        batch_size = x.shape[0]
        condition_mask = condition_mask.reshape(batch_size, -1)
        latent_indices = jnp.where(condition_mask[0] == 0, size=self.latent_dim)[0]
        observation_indices = jnp.where(
            condition_mask[0] == 1, size=self.observation_dim
        )[0]

        latent = jnp.take(x, latent_indices, axis=-1)
        observation = jnp.take(condition_value, observation_indices, axis=-1)
        time_ctx = GaussianFourierEmbedding(self.time_embedding_dim)(t[..., None])

        # Compress high-dim observation with a 3-layer encoder + LayerNorm.
        h_obs = hk.Linear(256, name="cond_enc_0")(observation)
        h_obs = self.act(h_obs)
        h_obs = hk.Linear(128, name="cond_enc_1")(h_obs)
        h_obs = self.act(h_obs)
        h_obs = hk.Linear(self.cond_emb_dim, name="cond_enc_2")(h_obs)
        h_obs = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name="cond_enc_norm")(h_obs)

        features = jnp.concatenate([time_ctx, latent, h_obs], axis=-1)
        hidden_sizes = [self.hidden_dim] * self.num_layers + [1]
        out = hk.nets.MLP(hidden_sizes, activation=self.act)(features)
        return out.squeeze(-1)


class LogRatioMLPEmbedded(hk.Module):
    """MLP log-ratio estimator that reuses the score model's observation encoder.

    Instead of learning a fresh condition encoder from scratch (as
    ``LogRatioMLPLarge`` does), this model consumes the embedding network that
    was trained end-to-end with the score model under ``embed_x=True``.  The
    ratio model therefore sees exactly the same summary of the observation that
    the diffusion backbone was conditioned on.

    Two reuse modes:

    ``embed_fn`` given (frozen)
        A closure ``embed_fn(observation) -> (batch, emb_dim)`` holding the
        pretrained weights outside the trainable parameter tree.  Nothing can
        drift -- note this matters with ``optax.adamw``, whose weight decay
        would otherwise shrink "frozen" weights even at zero gradient.

    ``embed_module_kwargs`` given (fine-tuned)
        An :class:`XEmbeddingNet` is built inside this module, so its weights
        land in the trainable tree and can be warm-started from the score model.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        embed_fn=None,
        embed_module_kwargs=None,
        hidden_dim: int = 512,
        num_layers: int = 6,
        time_embedding_dim: int = 16,
        act=silu,
        name: str = "log_ratio_mlp_embedded",
    ):
        super().__init__(name=name)
        if (embed_fn is None) == (embed_module_kwargs is None):
            raise ValueError(
                "Provide exactly one of embed_fn (frozen reuse) or "
                "embed_module_kwargs (fine-tuned reuse)."
            )
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.embed_fn = embed_fn
        self.embed_module_kwargs = embed_module_kwargs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.act = act

    def __call__(self, t, x, condition_mask, condition_value):
        batch_size = x.shape[0]
        condition_mask = condition_mask.reshape(batch_size, -1)

        latent = x[:, : self.theta_dim]
        observation = condition_value[:, self.theta_dim :]

        if self.embed_fn is not None:
            h_obs = self.embed_fn(observation)
        else:
            tokens = XEmbeddingNet(**self.embed_module_kwargs)(observation)
            h_obs = tokens.reshape(batch_size, -1)

        time_ctx = GaussianFourierEmbedding(self.time_embedding_dim)(t[..., None])
        features = jnp.concatenate([time_ctx, latent, h_obs], axis=-1)
        hidden_sizes = [self.hidden_dim] * self.num_layers + [1]
        out = hk.nets.MLP(hidden_sizes, activation=self.act)(features)
        return out.squeeze(-1)


def make_task_log_ratio_target(task, condition_mask):
    """Build a non-jitted oracle target function from the task exact log-ratio.

    This helper is intended for *training data preparation* of a log-ratio model,
    not for direct use inside the JAX sampler. The current `task.log_density_ratio`
    implementation may rely on Torch / Pyro, so it is not guaranteed to be JIT-safe.

    Args:
        task: Prior-shift task that provides `get_exact_log_ratio()` or
            `log_density_ratio()` over non-conditioned variables.
        condition_mask: Boolean or 0/1 mask over the full state `[latent, observed]`.

    Returns:
        A Python callable `target_fn(x_full)` that extracts the non-conditioned part
        and returns the exact log density ratio from the task.
    """
    condition_mask = jnp.asarray(condition_mask).astype(bool)
    latent_mask = ~condition_mask

    if hasattr(task, "get_exact_log_ratio"):
        oracle = task.get_exact_log_ratio
    else:
        oracle = task.log_density_ratio

    def target_fn(x_full):
        theta = jnp.asarray(x_full)[latent_mask]
        return oracle(theta)

    return target_fn


def generate_noised_samples_with_ratio(
    task,
    model,
    key,
    condition_mask,
    t,
    batch_size,
    log_density_ratio_fn=None,
    z_score_params=None,
):
    """Sample from task (unshifted), noise non-conditioned dims, return ratio and x_o.

    The full joint vector is [theta | x_obs]. condition_mask marks which dims are
    conditioned (theta is NEVER conditioned). Non-conditioned dims (all of theta +
    unconditioned x dims) are noised at time t; conditioned dims stay clean.

    Args:
        task: Prior-shift task with `sample_unshifted` and `log_density_ratio`.
        model: `AllConditionalScoreModel` with a `.sde` attribute.
        key: JAX PRNG key.
        condition_mask: 0/1 array of shape `(D,)`. 1 marks conditioned (observed)
            dims in x; theta positions are always 0.
        t: Scalar continuous diffusion time in `[T_min, T_max]`.
        batch_size: Number of samples to generate.
        log_density_ratio_fn: Optional callable `fn(theta) -> log_ratio` to use
            instead of `task.log_density_ratio`. Useful for plugging in a learned
            or alternative ratio estimator. Defaults to `task.log_density_ratio`.
        z_score_params: Optional dict with keys `z_score_fn` and `un_z_score_fn`.
            If provided, overrides `model.z_score_params` for normalising the
            training data. Use `compute_guidance_z_score_params` (in methods.py)
            to build this dict from task samples.

    Returns:
        x_t: shape `(batch_size, D)`. Full joint vector with non-conditioned dims
            noised at time t and conditioned dims left clean.
        x_obs_full: shape `(batch_size, D)`. Clean observed values embedded into the
            full D-dim space (zeros at latent positions, clean x_obs at conditioned
            positions). Suitable for passing directly as a conditioning input alongside
            x_t, e.g. `model_ratio(x_t, x_obs_full, t)`.
        log_ratios: shape `(batch_size,)`. Exact oracle log density ratio
            log p_target(theta) / log p_ref(theta) computed on clean theta.

    Note:
        NOT JIT-safe: `task.log_density_ratio` goes through Torch/Pyro internally.
        Intended for training data preparation only.
    """
    condition_mask = jnp.asarray(condition_mask)
    condition_mask_bool = condition_mask.astype(bool)
    latent_mask = 1 - condition_mask  # shape (D,), 1 where noise is applied

    # 1. Sample (theta, x_obs) from task.
    thetas, x_obs = task.sample_unshifted(batch_size, key=key)
    thetas = jnp.asarray(thetas)   # (batch_size, theta_dim)
    x_obs = jnp.asarray(x_obs)    # (batch_size, x_dim)

    # 2. Build full joint state [theta | x_obs], shape (batch_size, D).
    x_full = jnp.concatenate([thetas, x_obs], axis=-1)
    x_obs_full = jnp.zeros_like(x_full)
    x_obs_full = x_obs_full.at[:, condition_mask_bool].set(x_full[:, condition_mask_bool])

    # 2b. Z-score if the diffusion model was trained on standardised data so that
    #     the log-ratio model operates in the same space as the score model.
    #     An explicit z_score_params argument takes priority over the model's own.
    _z_score_params = z_score_params if z_score_params is not None else model.z_score_params
    if _z_score_params is not None:
        z_score_fn = _z_score_params["z_score_fn"]
        x_full = z_score_fn(x_full, model.node_id)
        x_obs_full = jnp.zeros_like(x_full)
        x_obs_full = x_obs_full.at[:, condition_mask_bool].set(
            x_full[:, condition_mask_bool]
        )

    # 3. Add diffusion noise to non-conditioned dims only in the model's SDE.
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=x_full.shape)
    mean_t = model.sde.marginal_mean(jnp.atleast_1d(t), x_full)
    std_t = model.sde.marginal_stddev(jnp.atleast_1d(t), x_full)
    x_t = x_full.copy()
    x_t = x_t.at[:, latent_mask.astype(bool)].set(
        mean_t[:, latent_mask.astype(bool)] + std_t[:, latent_mask.astype(bool)] * eps[:, latent_mask.astype(bool)]
    )

    # 4. Compute log density ratio on clean theta.
    _ratio_fn = log_density_ratio_fn if log_density_ratio_fn is not None else task.log_density_ratio
    log_ratios = jnp.asarray(_ratio_fn(thetas))  # (batch_size,)

    return x_t, x_obs_full, log_ratios


def prepare_log_ratio_training_batch(
    task,
    condition_mask,
    reference_samples,
    target_samples,
    times=None,
):
    """Placeholder helper for constructing supervised log-ratio training batches.

    Suggested usage:
        - `reference_samples`: samples from `task.sample_unshifted(...)`
        - `target_samples`: samples from `task.sample_shifted(...)`
        - `times`: diffusion times if you train a time-dependent ratio model
        - `task.get_exact_log_ratio(...)` provides the oracle target on the latent part

    Returns:
        A dictionary-like structure suitable for your trainer.
    """
    del task, condition_mask, reference_samples, target_samples, times
    raise NotImplementedError(
        "Implement the data preparation for your log-ratio model here."
    )


def train_log_ratio_model(
    task,
    diffusion_model,
    condition_mask,
    model_spec,
    rng,
    batch_size=256,
    epochs=5000,
    lr=1e-3,
    patience=1000,
    log_density_ratio_fn=None,
    z_score_params=None,
):
    """Train the time-dependent log-ratio model using a logistic DRE objective.

    Follows the same training loop as the reference `train_pt_div_qt`:
      - At each step, sample a random diffusion time t.
      - Draw two batches from the reference distribution via
        `generate_noised_samples_with_ratio`. Both batches use the same task
        sampler; one is importance-weighted by the oracle ratio w = exp(log_ratio).
      - Train the model to classify weighted-positive vs unweighted-negative
        samples with the logistic DRE loss:

            L = -E_ref[w * log σ(r(t, x_t))] - E_ref[log σ(-r(t, x_t))]

        The weighted positive samples approximate the target distribution, so the
        classifier converges to r_psi(t, x_t) ≈ log p_target_t(x_t) / p_ref_t(x_t).

    Args:
        task: Prior-shift task with `sample_unshifted` and `log_density_ratio`.
        diffusion_model: `AllConditionalScoreModel` (provides SDE noise schedule).
        condition_mask: 0/1 array, shape `(D,)`.
        model_spec: `LogRatioModelSpec` with `log_ratio_apply` and `params`.
        rng: JAX PRNG key.
        batch_size: Samples per gradient step.
        epochs: Maximum training epochs.
        lr: AdamW learning rate.
        patience: Early-stopping patience in epochs.
        log_density_ratio_fn: Optional callable `fn(theta) -> log_ratio`. If None,
            falls back to `task.log_density_ratio`. Pass a custom function to use
            a learned or alternative ratio estimator.
        z_score_params: Optional normalisation dict (keys: `z_score_fn`,
            `un_z_score_fn`, `mean_per_node_id`, `std_per_node_id`).
            If provided, overrides `diffusion_model.z_score_params` when generating
            training samples.  Build with `compute_guidance_z_score_params`.

    Returns:
        updated_model_spec: `LogRatioModelSpec` with trained params.
        loss_history: List of per-epoch scalar losses.
    """
    import optax

    condition_mask = jnp.asarray(condition_mask)
    params = model_spec.params
    log_ratio_apply = model_spec.log_ratio_apply

    schedule = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=0.4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, t):
        def loss_fn(params):
            def apply_single(x_t, x_obs):
                return log_ratio_apply(
                    params, jnp.atleast_1d(t), x_t, condition_mask, x_obs
                )

            r_p = jax.vmap(apply_single)(x_t_p, x_obs_p)  # (batch_size,)
            r_q = jax.vmap(apply_single)(x_t_q, x_obs_q)  # (batch_size,)

            loss_p = -wp * jax.nn.log_sigmoid(r_p)
            # log(1 - sigmoid(x)) = log_sigmoid(-x)
            loss_q = -jax.nn.log_sigmoid(-r_q)
            return jnp.mean(loss_p) + jnp.mean(loss_q)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    loss_history = []
    early_stopping_enabled = patience is not None and patience > 0
    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        rng, key_t, key_p, key_q = jax.random.split(rng, 4)

        t = jax.random.uniform(
            key_t, minval=diffusion_model.T_min, maxval=diffusion_model.T_max
        )

        # Positive batch: reference samples reweighted by oracle importance weights.
        x_t_p, x_obs_p, log_ratios_p = generate_noised_samples_with_ratio(
            task, diffusion_model, key_p, condition_mask, t, batch_size,
            log_density_ratio_fn=log_density_ratio_fn,
            z_score_params=z_score_params,
        )
        wp = jnp.exp(log_ratios_p)  # shape (batch_size,)

        # Negative batch: reference samples, uniform weight (w=1).
        x_t_q, x_obs_q, _ = generate_noised_samples_with_ratio(
            task, diffusion_model, key_q, condition_mask, t, batch_size,
            log_density_ratio_fn=log_density_ratio_fn,
            z_score_params=z_score_params,
        )

        params, opt_state, loss = update_step(params, opt_state, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, t)

        l = float(loss)
        loss_history.append(l)

        if early_stopping_enabled:
            if l < best_loss:
                best_loss = l
                counter = 0
            else:
                counter += 1

        if epoch % 30 == 0:
            print(f"Epoch {epoch}: loss={l:.4f}", flush=True)

        if early_stopping_enabled and counter >= patience:
            print(f"Stopping at epoch {epoch} (early stopping)", flush=True)
            break

    if loss_history:
        print(f"Stopping at epoch {len(loss_history) - 1}", flush=True)

    return LogRatioModelSpec(log_ratio_apply=log_ratio_apply, params=params), loss_history


def _apply_log_ratio_model(
    log_ratio_apply,
    log_ratio_params,
    t,
    x,
    condition_mask,
    condition_value,
):
    value = log_ratio_apply(
        log_ratio_params,
        jnp.atleast_1d(t),
        x,
        condition_mask,
        condition_value,
    )
    return jnp.squeeze(value)


def prior_guide_with_log_ratio_model(
    model,
    key,
    condition_mask,
    x_o,
    x_T,
    log_ratio_apply,
    log_ratio_params,
    num_steps=500,
    rho=2.0,
    clipping_range=None,
    verbose=False,
    guidance_scale=1.0,
):
    """Guided reverse diffusion using a trained log-ratio model.

    This mirrors the structure of `prior_guide_theta_prior_only()` but replaces the
    closed-form GMM guidance with a learned time-dependent log-ratio model
    `r_psi(t, x_t)`.

    The intended semantics are:
                - `log_ratio_apply(log_ratio_params, t, x_full, condition_mask, condition_value)`
                    returns a scalar log-ratio estimate for the *full noisy state* at time `t`
                - the autodiff of this model w.r.t. `x_t` on non-conditioned variables is
                    used directly as the additive guidance term
                - both the base score and the log-ratio model are explicitly conditioned
                    on the observation context

        Concretely:
                `score_new(x_t, t) = score_old(x_t, t) + guidance_scale * ∇_{x_t} r_psi(t, x_t)`

    Important:
        `log_ratio_apply` must be implemented in JAX if you want to JIT / VMAP this
        sampler efficiently.
    """

    condition_mask_bool = jnp.asarray(condition_mask, dtype=bool)
    condition_mask = condition_mask_bool.astype(x_T.dtype)
    latent_mask = 1 - condition_mask
    cond_size = x_o.shape[0]
    cond_idx = jnp.nonzero(condition_mask_bool, size=cond_size)[0]

    # Z-score x_o so the score model and log-ratio model both receive
    # standardised inputs (they were both trained in z-scored space).
    if model.z_score_params is not None:
        z_score_fn = model.z_score_params["z_score_fn"]
        cond_node_id = model.node_id[cond_idx]
        x_o = z_score_fn(x_o, cond_node_id)

    ts = jnp.linspace(model.T_min, model.T_max, num_steps) ** rho
    x_o_full = jnp.zeros_like(x_T)
    x_o_full = x_o_full.at[cond_idx].set(x_o)

    score_fn = model._init_score(
        node_id=model.node_id,
        condition_mask=condition_mask,
        edge_mask=None,
        meta_data=None,
    )

    def compute_guided_score(x, t):
        x_conditioned = condition_mask * x_o_full + latent_mask * x
        base_score = score_fn(t, x_conditioned)

        def log_ratio_on_xt(x_t_in):
            x_t_conditioned = condition_mask * x_o_full + latent_mask * x_t_in
            return _apply_log_ratio_model(
                log_ratio_apply,
                log_ratio_params,
                t,
                x_t_conditioned,
                condition_mask,
                x_o_full,
            )

        guidance_grad = grad(log_ratio_on_xt)(x_conditioned)
        guidance_grad = latent_mask * guidance_grad
        guided_score = base_score + guidance_scale * guidance_grad

        if clipping_range is not None:
            guided_score = jnp.clip(guided_score, clipping_range[0], clipping_range[1])

        if verbose and t == ts[-1]:
            jax.debug.print("Verbose mode --------------------------------------")
            jax.debug.print("t {x}", x=t)
            jax.debug.print("base_score {x}", x=base_score)
            jax.debug.print("guidance_grad {x}", x=guidance_grad)
            jax.debug.print("guided_score {x}", x=guided_score)

        return guided_score

    def scan_fn(carry, t0):
        key, t1, x1 = carry
        dt = t0 - t1

        # Langevin correction is intentionally disabled for this guidance method.
        # If you want to re-enable it later, restore the interleaved correction
        # steps here before the reverse diffusion update.
        x1_after_langevin = x1

        key, subkey = jax.random.split(key)
        guided_score = compute_guided_score(x1_after_langevin, t1)

        drift_backward = latent_mask * (
            model.sde.drift(t1, x1_after_langevin)
            - model.sde.diffusion(t1, x1_after_langevin) ** 2 * guided_score
        )
        diffusion_backward = latent_mask * model.sde.diffusion(t1, x1_after_langevin)

        xt0_uncond = (
            x1_after_langevin
            + drift_backward * dt
            + diffusion_backward
            * jnp.sqrt(jnp.abs(dt))
            * jax.random.normal(subkey, shape=x1_after_langevin.shape)
        )

        xt0 = condition_mask * x_o_full + latent_mask * xt0_uncond
        return (key, t0, xt0), xt0

    carry_init = (key, ts[-1], x_T)
    _, traj = jax.lax.scan(scan_fn, carry_init, ts[::-1][1:])
    # _, traj = debug_scan(scan_fn, carry_init, ts[::-1][1:])

    result = traj[-1]

    # Un-z-score the latent (theta) part of the output so the caller receives
    # samples in the original (un-standardised) space.
    if model.z_score_params is not None:
        un_z_score_fn = model.z_score_params["un_z_score_fn"]
        latent_size = result.shape[0] - cond_size
        latent_idx = jnp.nonzero(~condition_mask_bool, size=latent_size)[0]
        latent_node_id = model.node_id[latent_idx]
        latent_vals = result[latent_idx]
        latent_vals = un_z_score_fn(latent_vals, latent_node_id)
        result = result.at[latent_idx].set(latent_vals)

    return result
