"""Unified time-dependent density-ratio estimation (DRE) training.

This module merges the logistic-DRE and KLIEP training paths into one
implementation. The two methods now share the same data pipeline, optimizer
scaffolding, and training loop; only the loss differs.

This module also owns LogRatioModelSpec and all log-ratio model init helpers
(formerly in guidance_ours.py).
"""

import logging
import jax
import jax.numpy as jnp
from jax import grad
import jax.lax
from jax.nn import gelu, silu

import haiku as hk

_logger = logging.getLogger(__name__)

from priorg.sim.nn.helpers import GaussianFourierEmbedding
from priorg.sim.nn.tokenizer import ScalarTokenizer
from priorg.sim.nn.transformers import Transformer
# NOTE: LogRatioTransformer / LogRatioMLP / LogRatioMLPLarge are imported
# lazily (inside each init function) to avoid a circular import with
# guidance_ours, which re-exports the init functions defined here.


class LogRatioModelSpec:
    """Container for a time-dependent log-ratio model.

    Expected callable signature:
        log_ratio_apply(params, t, x, condition_mask, condition_value) -> scalar

    where:
        - `params` is an arbitrary model state / parameter pytree
        - `t` has shape `(1,)` or is scalar-like
        - `x` has shape `(D,)`
        - `condition_mask` has shape `(D,)`
        - `condition_value` has shape `(D,)`, with observed entries filled in
    """

    def __init__(self, log_ratio_apply, params=None, state=None):
        self.log_ratio_apply = log_ratio_apply
        self.params = params
        self.state = state


def init_log_ratio_model(*args, **kwargs):
    """Initialise a log-ratio model (MLP or Transformer).

    Keyword Args:
        model_type: ``'transformer'`` (default), ``'mlp'``, ``'mlp_large'``, or
            ``'mlp_embedded'`` (reuses the score model's observation embedding
            network; requires a model trained with ``embed_x=True``).
    """
    model_type = kwargs.pop("model_type", "transformer")
    if model_type == "transformer":
        kwargs.pop("mlp_hidden_dim", None)
        kwargs.pop("mlp_num_layers", None)
        kwargs.pop("mlp_time_embedding_dim", None)
        kwargs.pop("mlp_activation", None)
        kwargs.pop("mlp_cond_emb_dim", None)
        kwargs.pop("condition_mask", None)
        kwargs.pop("freeze_embedding", None)
        return init_log_ratio_transformer_from_score_model(*args, **kwargs)
    if model_type == "mlp":
        kwargs.pop("reinit_output_head", None)
        kwargs.pop("mlp_cond_emb_dim", None)
        kwargs.pop("freeze_embedding", None)
        return init_log_ratio_mlp(*args, **kwargs)
    if model_type == "mlp_large":
        kwargs.pop("reinit_output_head", None)
        kwargs.pop("freeze_embedding", None)
        return init_log_ratio_mlp_large(*args, **kwargs)
    if model_type == "mlp_embedded":

        kwargs.pop("reinit_output_head", None)
        kwargs.pop("mlp_cond_emb_dim", None)
        return init_log_ratio_mlp_embedded(*args, **kwargs)
    raise ValueError(f"Unsupported model_type: {model_type}")


def init_log_ratio_transformer_from_score_model(
    score_model,
    rng,
    reinit_output_head=True,
):
    """Build a transformer log-ratio model sharing the score-model backbone."""
    from priorg.sim.methods.guidance_ours import LogRatioTransformer  # lazy
    p = score_model.model_init_params
    num_nodes = p["num_nodes"]
    node_id = score_model.node_id

    def model_fn(t, data, data_id, condition_mask):
        return LogRatioTransformer(num_nodes=num_nodes, **{
            k: v for k, v in p.items() if k != "num_nodes"
        })(t, data, data_id, condition_mask)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(model_fn))

    rng, subkey = jax.random.split(rng)
    new_params = init_fn(
        subkey,
        jnp.ones((1,)),
        jnp.ones((1, num_nodes, 1)),
        node_id,
        jnp.zeros((1, num_nodes)),
    )


    new_params, n_copied = copy_score_backbone_params(
        new_params,
        score_model.params,
        module_name="log_ratio_transformer",
        reinit_output_head=reinit_output_head,
    )
    _logger.info(
        "Warm-started log-ratio transformer from score model: "
        "%d parameter arrays copied.",
        n_copied,
    )

    def log_ratio_apply(params, t, x, condition_mask, condition_value):
        result = apply_fn(
            params,
            jnp.atleast_1d(t).reshape(1,),
            x.reshape(1, num_nodes, 1),
            node_id,
            condition_mask.reshape(1, num_nodes),
        )
        return result.squeeze()

    return LogRatioModelSpec(log_ratio_apply=log_ratio_apply, params=new_params)


def init_log_ratio_mlp(
    score_model,
    rng,
    mlp_hidden_dim=256,
    mlp_num_layers=4,
    mlp_time_embedding_dim=4,
    mlp_activation="silu",
    condition_mask=None,
    **kwargs,
):
    """Build an MLP log-ratio model over time, noisy latent state, and observation."""
    from priorg.sim.methods.guidance_ours import LogRatioMLP  # lazy
    del kwargs

    num_nodes = score_model.model_init_params["num_nodes"]
    if condition_mask is None:
        condition_mask = score_model.condition_mask.astype(bool)
    else:
        condition_mask = jnp.asarray(condition_mask).astype(bool)
    latent_dim = int((~condition_mask).sum())
    observation_dim = int(condition_mask.sum())

    if mlp_activation == "gelu":
        mlp_act = gelu
    elif mlp_activation == "silu":
        mlp_act = silu
    else:
        raise ValueError(f"Unsupported mlp_activation: {mlp_activation}")

    def model_fn(t, x, condition_mask, condition_value):
        return LogRatioMLP(
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            time_embedding_dim=mlp_time_embedding_dim,
            act=mlp_act,
        )(t, x, condition_mask, condition_value)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(model_fn))

    rng, subkey = jax.random.split(rng)
    dummy_x = jnp.ones((1, num_nodes))
    dummy_condition_mask = condition_mask.reshape(1, num_nodes)
    dummy_condition_value = jnp.zeros((1, num_nodes)).at[:, condition_mask].set(1.0)
    params = init_fn(
        subkey,
        jnp.ones((1,)),
        dummy_x,
        dummy_condition_mask,
        dummy_condition_value,
    )

    def log_ratio_apply(params, t, x, condition_mask, condition_value):
        result = apply_fn(
            params,
            jnp.atleast_1d(t).reshape(1,),
            x.reshape(1, num_nodes),
            condition_mask.reshape(1, num_nodes),
            condition_value.reshape(1, num_nodes),
        )
        return result.squeeze()

    return LogRatioModelSpec(log_ratio_apply=log_ratio_apply, params=params)


def init_log_ratio_mlp_large(
    score_model,
    rng,
    mlp_hidden_dim=512,
    mlp_num_layers=6,
    mlp_time_embedding_dim=16,
    mlp_activation="silu",
    mlp_cond_emb_dim=64,
    condition_mask=None,
    **kwargs,
):
    """Build a large MLP log-ratio model with condition encoder for high-dim obs."""
    from priorg.sim.methods.guidance_ours import LogRatioMLPLarge  # lazy
    del kwargs

    num_nodes = score_model.model_init_params["num_nodes"]
    if condition_mask is None:
        condition_mask = score_model.condition_mask.astype(bool)
    else:
        condition_mask = jnp.asarray(condition_mask).astype(bool)
    latent_dim = int((~condition_mask).sum())
    observation_dim = int(condition_mask.sum())

    if mlp_activation == "gelu":
        mlp_act = gelu
    elif mlp_activation == "silu":
        mlp_act = silu
    else:
        raise ValueError(f"Unsupported mlp_activation: {mlp_activation}")

    def model_fn(t, x, condition_mask, condition_value):
        return LogRatioMLPLarge(
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            cond_emb_dim=mlp_cond_emb_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            time_embedding_dim=mlp_time_embedding_dim,
            act=mlp_act,
        )(t, x, condition_mask, condition_value)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(model_fn))

    rng, subkey = jax.random.split(rng)
    dummy_x = jnp.ones((1, num_nodes))
    dummy_condition_mask = condition_mask.reshape(1, num_nodes)
    dummy_condition_value = jnp.zeros((1, num_nodes)).at[:, condition_mask].set(1.0)
    params = init_fn(
        subkey,
        jnp.ones((1,)),
        dummy_x,
        dummy_condition_mask,
        dummy_condition_value,
    )

    def log_ratio_apply(params, t, x, condition_mask, condition_value):
        result = apply_fn(
            params,
            jnp.atleast_1d(t).reshape(1,),
            x.reshape(1, num_nodes),
            condition_mask.reshape(1, num_nodes),
            condition_value.reshape(1, num_nodes),
        )
        return result.squeeze()

    return LogRatioModelSpec(log_ratio_apply=log_ratio_apply, params=params)


def init_log_ratio_model_from_score_model(*args, **kwargs):
    """Backward-compatible alias for transformer initialization."""
    return init_log_ratio_transformer_from_score_model(*args, **kwargs)


def generate_noised_samples_with_ratio(
    task,
    model,
    key,
    condition_mask,
    t,
    batch_size,
    log_density_ratio_fn=None,
    z_score_params=None,
    dataset=None,
):
    """Sample reference points, noise latent dimensions, and return log-ratios.

    If ``dataset`` is provided, draw ``theta``/``x`` minibatches from the fixed
    dataset instead of calling ``task.sample_unshifted``.
    """
    condition_mask = jnp.asarray(condition_mask)
    condition_mask_bool = condition_mask.astype(bool)
    latent_mask = 1 - condition_mask

    if dataset is None:
        thetas, x_obs = task.sample_unshifted(batch_size, key=key)
        thetas = jnp.asarray(thetas)
        x_obs = jnp.asarray(x_obs)
    else:
        theta_all = jnp.asarray(dataset["theta"])
        x_all = jnp.asarray(dataset["x"])
        if theta_all.shape[0] != x_all.shape[0]:
            raise ValueError("dataset['theta'] and dataset['x'] must have same length")
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, theta_all.shape[0])
        thetas = theta_all[idx]
        x_obs = x_all[idx]

    x_full = jnp.concatenate([thetas, x_obs], axis=-1)
    x_obs_full = jnp.zeros_like(x_full)
    x_obs_full = x_obs_full.at[:, condition_mask_bool].set(x_full[:, condition_mask_bool])

    _z_score_params = z_score_params if z_score_params is not None else model.z_score_params
    if _z_score_params is not None:
        z_score_fn = _z_score_params["z_score_fn"]
        x_full = z_score_fn(x_full, model.node_id)
        x_obs_full = jnp.zeros_like(x_full)
        x_obs_full = x_obs_full.at[:, condition_mask_bool].set(x_full[:, condition_mask_bool])

    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=x_full.shape)
    mean_t = model.sde.marginal_mean(jnp.atleast_1d(t), x_full)
    std_t = model.sde.marginal_stddev(jnp.atleast_1d(t), x_full)

    x_t = x_full.copy()
    latent_mask_bool = latent_mask.astype(bool)
    x_t = x_t.at[:, latent_mask_bool].set(
        mean_t[:, latent_mask_bool] + std_t[:, latent_mask_bool] * eps[:, latent_mask_bool]
    )

    ratio_fn = log_density_ratio_fn if log_density_ratio_fn is not None else task.log_density_ratio
    log_ratios = jnp.asarray(ratio_fn(thetas))
    return x_t, x_obs_full, log_ratios


def train_log_ratio_model_dre(
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
    dataset=None,
    objective="logistic",
    softmax_t=False,
    n_time_samples=8,
    beta_start=0.0,
    beta_end=8.0,
    beta_warmup_frac=0.0,
    beta_schedule="linear",
):
    """Train a time-dependent log-ratio model with configurable DRE objective.

    Supported objectives:
      - ``logistic``: weighted logistic DRE (classifier-style objective)
      - ``kliep``: KLIEP objective (Sugiyama et al., 2008)

    Time aggregation
    ----------------
    By default (``softmax_t=False``) the per-step objective is the usual
    single-``t`` Monte-Carlo estimate of ``E_t[f(theta, t)]``: one ``t`` is
    drawn per step and, across steps, SGD minimises the expectation over ``t``.

    When ``softmax_t=True`` the per-step objective is replaced by a soft-max
    (log-sum-exp) relaxation of ``max_t f(theta, t)``::

        L_beta(theta) = (1/beta) * log( (1/K) * sum_k exp(beta * f(theta, t_k)) )

    where ``K = n_time_samples`` distinct ``t_k ~ Uniform[T_min, T_max]`` are
    drawn *per step*. ``beta -> 0`` recovers the plain mean over the ``K``
    points (i.e. the original expectation objective); ``beta -> inf`` approaches
    ``max_k f(theta, t_k)`` (the minimax / worst-``t`` objective). ``beta`` is
    annealed from ``beta_start`` to ``beta_end`` over training so early epochs
    are stable (mean-like) and later epochs push toward the worst-case ``t``.

    Args:
      softmax_t: enable the soft-max-in-``t`` relaxation described above.
      n_time_samples: number of distinct ``t`` evaluated per step (``K``). Only
        used when ``softmax_t=True``. Larger ``K`` lowers both the log-sum-exp
        estimation bias (``O(1/K)``) and the variance of the annealed target.
      beta_start, beta_end: inverse-temperature endpoints for the anneal.
      beta_warmup_frac: fraction of ``epochs`` held at ``beta_start`` before the
        anneal begins (a pure warmup phase). ``0.0`` disables the hold.
      beta_schedule: ``"linear"``, ``"cosine"``, or ``"exp"`` interpolation of
        ``beta`` from ``beta_start`` to ``beta_end`` after the warmup.

    Note:
      ``beta`` here is a schedule computed on the host per epoch and passed into
      the jitted step as a traced scalar, so no recompilation happens as it
      changes.
    """
    import optax

    if objective not in ("logistic", "kliep"):
        raise ValueError(f"Unsupported objective: {objective}")

    if softmax_t:
        if n_time_samples < 1:
            raise ValueError("n_time_samples must be >= 1")
        if beta_schedule not in ("linear", "cosine", "exp"):
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        if not (0.0 <= beta_warmup_frac < 1.0):
            raise ValueError("beta_warmup_frac must be in [0, 1)")

    def _beta_at(epoch):
        """Annealed inverse-temperature for the given epoch (host-side float)."""
        if epochs <= 1:
            return float(beta_end)
        warmup = int(beta_warmup_frac * epochs)
        if epoch < warmup:
            return float(beta_start)
        denom = max(epochs - 1 - warmup, 1)
        frac = (epoch - warmup) / denom
        frac = min(max(frac, 0.0), 1.0)
        if beta_schedule == "linear":
            w = frac
        elif beta_schedule == "cosine":
            # smooth ease-in/out from 0 -> 1
            w = 0.5 * (1.0 - jnp.cos(jnp.pi * frac))
            w = float(w)
        else:  # exp: geometric interpolation (guards against nonpositive endpoints)
            lo = max(beta_start, 1e-6)
            hi = max(beta_end, 1e-6)
            return float(lo * (hi / lo) ** frac)
        return float(beta_start + (beta_end - beta_start) * w)

    condition_mask = jnp.asarray(condition_mask)
    params = model_spec.params
    log_ratio_apply = model_spec.log_ratio_apply

    if objective == "logistic":
        schedule = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=0.4)
        adam = optax.adamw(schedule)
    else:
        adam = optax.adamw(lr)  # kliep uses a constant lr

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), adam)
    opt_state = optimizer.init(params)

    def per_t_loss(params, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, t):
        """Scalar DRE loss f(theta, t) at a single time ``t``.

        This is exactly the original per-step objective, factored out so it can
        be reused both for the plain expectation path (one ``t`` per step) and
        for the soft-max-in-``t`` path (``K`` times per step combined via
        annealed log-sum-exp).
        """
        def apply_single(x_t, x_obs):
            return log_ratio_apply(params, jnp.atleast_1d(t), x_t, condition_mask, x_obs)

        r_p = jax.vmap(apply_single)(x_t_p, x_obs_p)
        r_q = jax.vmap(apply_single)(x_t_q, x_obs_q)

        if objective == "logistic":
            loss_p = -wp * jax.nn.log_sigmoid(r_p)
            loss_q = -jax.nn.log_sigmoid(-r_q)
            return jnp.mean(loss_p) + jnp.mean(loss_q)

        # kliep: -E_ref[r̃·f] + log E_ref[exp f]
        wp_normalized = wp / (jnp.mean(wp) + 1e-8)
        weighted_term = -jnp.mean(wp_normalized * r_p)
        log_partition = jax.nn.logsumexp(r_q) - jnp.log(r_q.shape[0])
        return weighted_term + log_partition

    @jax.jit
    def update_step(params, opt_state, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, t):
        """Single-``t`` expectation step (SGD over t across epochs)."""
        def loss_fn(params):
            return per_t_loss(params, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, t)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def update_step_softmax(
        params, opt_state, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, ts, beta
    ):
        """Soft-max-in-``t`` step.

        ``x_t_p``/``x_t_q`` have a leading axis of size ``K = n_time_samples``
        (one p/q batch per time point); ``ts`` has shape ``(K,)``; ``wp`` has
        shape ``(K, batch)`` (its own importance weights per time slice).
        The objective is (1/beta) logmeanexp_k [ beta * f(theta, t_k) ], which
        interpolates from the mean over the K points (beta->0) to max_k
        (beta->inf).
        """
        def loss_fn(params):
            def one_slice(x_t_p_k, x_obs_p_k, x_t_q_k, x_obs_q_k, wp_k, t_k):
                return per_t_loss(
                    params, x_t_p_k, x_obs_p_k, x_t_q_k, x_obs_q_k, wp_k, t_k
                )

            f_vals = jax.vmap(one_slice)(
                x_t_p, x_obs_p, x_t_q, x_obs_q, wp, ts
            )  # shape (K,)

            K = f_vals.shape[0]
            log_K = jnp.log(jnp.asarray(K, f_vals.dtype))

            # Numerically stable, and smoothly reduces to the plain mean as
            # beta -> 0 (where logmeanexp(beta f)/beta -> mean(f)).
            def softmax_agg(_):
                return (jax.nn.logsumexp(beta * f_vals) - log_K) / beta

            def mean_agg(_):
                return jnp.mean(f_vals)

            # Guard the 1/beta as beta -> 0 (and allow beta==0 to mean "pure
            # expectation over the K points").
            return jax.lax.cond(beta > 1e-8, softmax_agg, mean_agg, operand=None)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    loss_history = []
    early_stopping_enabled = patience is not None and patience > 0
    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        if not softmax_t:
            # -------- plain expectation over t (original behaviour) --------
            rng, key_t, key_p, key_q = jax.random.split(rng, 4)

            t = jax.random.uniform(
                key_t,
                minval=diffusion_model.T_min,
                maxval=diffusion_model.T_max,
            )

            x_t_p, x_obs_p, log_ratios_p = generate_noised_samples_with_ratio(
                task,
                diffusion_model,
                key_p,
                condition_mask,
                t,
                batch_size,
                log_density_ratio_fn=log_density_ratio_fn,
                z_score_params=z_score_params,
                dataset=dataset,
            )
            wp = jnp.exp(log_ratios_p)

            x_t_q, x_obs_q, _ = generate_noised_samples_with_ratio(
                task,
                diffusion_model,
                key_q,
                condition_mask,
                t,
                batch_size,
                log_density_ratio_fn=log_density_ratio_fn,
                z_score_params=z_score_params,
                dataset=dataset,
            )

            params, opt_state, loss = update_step(
                params, opt_state, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, t,
            )
        else:
            # -------- soft-max relaxation of max_t f (annealed beta) --------
            rng, key_t = jax.random.split(rng)
            ts = jax.random.uniform(
                key_t,
                shape=(n_time_samples,),
                minval=diffusion_model.T_min,
                maxval=diffusion_model.T_max,
            )

            xtp_list, xop_list, wp_list = [], [], []
            xtq_list, xoq_list = [], []
            for k in range(n_time_samples):
                rng, key_p, key_q = jax.random.split(rng, 3)
                x_t_p_k, x_obs_p_k, log_ratios_p_k = generate_noised_samples_with_ratio(
                    task,
                    diffusion_model,
                    key_p,
                    condition_mask,
                    ts[k],
                    batch_size,
                    log_density_ratio_fn=log_density_ratio_fn,
                    z_score_params=z_score_params,
                    dataset=dataset,
                )
                x_t_q_k, x_obs_q_k, _ = generate_noised_samples_with_ratio(
                    task,
                    diffusion_model,
                    key_q,
                    condition_mask,
                    ts[k],
                    batch_size,
                    log_density_ratio_fn=log_density_ratio_fn,
                    z_score_params=z_score_params,
                    dataset=dataset,
                )
                xtp_list.append(x_t_p_k)
                xop_list.append(x_obs_p_k)
                wp_list.append(jnp.exp(log_ratios_p_k))
                xtq_list.append(x_t_q_k)
                xoq_list.append(x_obs_q_k)

            # Stack along a leading axis of size K = n_time_samples.
            x_t_p = jnp.stack(xtp_list)
            x_obs_p = jnp.stack(xop_list)
            wp = jnp.stack(wp_list)
            x_t_q = jnp.stack(xtq_list)
            x_obs_q = jnp.stack(xoq_list)

            beta = jnp.asarray(_beta_at(epoch), dtype=x_t_p.dtype)

            params, opt_state, loss = update_step_softmax(
                params, opt_state, x_t_p, x_obs_p, x_t_q, x_obs_q, wp, ts, beta,
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
            if softmax_t:
                _logger.info(
                    f"[train_dre/{objective}] epoch {epoch}/{epochs}  "
                    f"loss={l:.4f}  beta={_beta_at(epoch):.3f}  K={n_time_samples}"
                )
            else:
                _logger.info(f"[train_dre/{objective}] epoch {epoch}/{epochs}  loss={l:.4f}")

        if early_stopping_enabled and counter >= patience:
            _logger.info(f"[train_dre/{objective}] early stop at epoch {epoch}")
            break

    if loss_history:
        _logger.info(f"[train_dre/{objective}] finished at epoch {len(loss_history) - 1}  final_loss={loss_history[-1]:.4f}")

    return LogRatioModelSpec(log_ratio_apply=log_ratio_apply, params=params), loss_history


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
    dataset=None,
    softmax_t=False,
    n_time_samples=8,
    beta_start=0.0,
    beta_end=8.0,
    beta_warmup_frac=0.0,
    beta_schedule="linear",
):
    """Compatibility wrapper for logistic DRE training."""
    return train_log_ratio_model_dre(
        task=task,
        diffusion_model=diffusion_model,
        condition_mask=condition_mask,
        model_spec=model_spec,
        rng=rng,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        log_density_ratio_fn=log_density_ratio_fn,
        z_score_params=z_score_params,
        dataset=dataset,
        softmax_t=softmax_t,
        n_time_samples=n_time_samples,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_warmup_frac=beta_warmup_frac,
        beta_schedule=beta_schedule,
        objective="logistic",
    )


def train_log_ratio_model_kliep(
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
    dataset=None,
    softmax_t=False,
    n_time_samples=8,
    beta_start=0.0,
    beta_end=8.0,
    beta_warmup_frac=0.0,
    beta_schedule="linear",
):
    """Compatibility wrapper for KLIEP training."""
    return train_log_ratio_model_dre(
        task=task,
        diffusion_model=diffusion_model,
        condition_mask=condition_mask,
        model_spec=model_spec,
        rng=rng,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        patience=patience,
        log_density_ratio_fn=log_density_ratio_fn,
        z_score_params=z_score_params,
        dataset=dataset,
        softmax_t=softmax_t,
        n_time_samples=n_time_samples,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_warmup_frac=beta_warmup_frac,
        beta_schedule=beta_schedule,
        objective="kliep",
    )


def _apply_log_ratio_model(
    log_ratio_apply,
    log_ratio_params,
    t,
    x,
    condition_mask,
    condition_value,
):
    """Apply log-ratio model and squeeze output to scalar."""
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


def _zero_log_ratio_apply(params, t, x, condition_mask, condition_value):
    return jnp.zeros(())


def sample_unguided(
    model,
    key,
    condition_mask,
    x_o,
    x_T,
    num_steps=25,
    rho=2.0,
):
    """Unguided reverse diffusion with the same sampler as the guided methods.

    Runs exactly the reverse VP-SDE discretisation and power-law time schedule of
    `prior_guide_with_log_ratio_model`, with the guidance term set to zero, so the
    base model (Simformer) is evaluated under the same sampler settings as the
    guided methods instead of `model.sample`'s default uniform grid.

    Takes x_o in the original (un-z-scored) space and returns the full state with
    the latent part un-z-scored, like `prior_guide_with_log_ratio_model`.
    """
    return prior_guide_with_log_ratio_model(
        model=model,
        key=key,
        condition_mask=condition_mask,
        x_o=x_o,
        x_T=x_T,
        log_ratio_apply=_zero_log_ratio_apply,
        log_ratio_params=None,
        num_steps=num_steps,
        rho=rho,
        guidance_scale=0.0,
    )


__all__ = [
    "LogRatioModelSpec",
    "sample_unguided",
    "copy_score_backbone_params",
    "generate_noised_samples_with_ratio",
    "init_log_ratio_model",
    "init_log_ratio_mlp",
    "init_log_ratio_mlp_embedded",
    "init_log_ratio_mlp_large",
    "init_log_ratio_model_from_score_model",
    "init_log_ratio_transformer_from_score_model",
    "prior_guide_with_log_ratio_model",
    "train_log_ratio_model",
    "train_log_ratio_model_dre",
    "train_log_ratio_model_kliep",
]


# Top-level Haiku module paths of the score model that make up the shared
# backbone. Haiku parameter dicts are flat and keyed by "/"-separated module
# paths (e.g. "transformer/multi_head_attention/key"), so these are matched
# against the first path component.
_SCORE_BACKBONE_PREFIXES = (
    "scalar_tokenizer",
    "gaussian_fourier_embedding",
    "transformer",
    "x_embedding_net",
    "x_score_decoder",
)


def copy_score_backbone_params(
    new_params,
    score_params,
    module_name="log_ratio_transformer",
    reinit_output_head=True,
):
    """Copy a score model's backbone into a freshly-initialised log-ratio model.

    The log-ratio model wraps the same architecture inside an ``hk.Module``, so
    every backbone path gains a ``"<module_name>/"`` prefix, and parameters the
    score model creates at the top level of its transform (key ``"~"``:
    ``condition_token`` and, with ``embed_x``, ``x_token_id_embedding``) move
    into the module's own scope (key ``"<module_name>"``).

    Only arrays whose shapes match are copied, so a mismatched configuration
    degrades to partial warm-starting rather than raising.

    Args:
        new_params: Freshly initialised log-ratio parameter dict.
        score_params: ``model.params`` of the trained score model.
        module_name: Haiku name of the log-ratio module.
        reinit_output_head: If False, the score model's output head is copied
            into ``log_ratio_head`` as well.

    Returns:
        ``(params, num_arrays_copied)``.
    """
    out = {k: dict(v) for k, v in new_params.items()}
    n_copied = 0

    for key, value in score_params.items():
        if key == "~":
            target = module_name
        elif key == "linear":
            if reinit_output_head:
                continue
            target = f"{module_name}/log_ratio_head"
        elif key.split("/")[0] in _SCORE_BACKBONE_PREFIXES:
            target = f"{module_name}/{key}"
        else:
            continue

        if target not in out:
            continue
        for param_name, array in value.items():
            existing = out[target].get(param_name)
            if existing is not None and existing.shape == array.shape:
                out[target][param_name] = array
                n_copied += 1

    return out, n_copied


def _embedding_config_from_score_model(score_model):
    """Read the observation-embedding configuration off a trained score model."""
    p = score_model.model_init_params
    if not p.get("embed_x", False):
        raise ValueError(
            "This score model was not trained with embed_x=True, so it has no "
            "observation embedding network to reuse. Train one with "
            "--embed_x, or use model_type='mlp_large' instead."
        )
    token_dim = p.get("token_dim", 40)
    if p.get("condition_mode") in ("add", "none"):
        # scalar_transformer_model widens the token in these modes.
        token_dim = token_dim + p.get("condition_token_dim", 10)
    return dict(
        token_dim=token_dim,
        x_dim=p["x_dim"],
        num_x_tokens=p.get("num_x_tokens", 1),
        hidden_sizes=tuple(p.get("x_embed_hidden", (256, 128))),
        chunked=p.get("x_embed_chunked", False),
        layer_norm=p.get("x_embed_layer_norm", True),
    )


def init_log_ratio_mlp_embedded(
    score_model,
    rng,
    mlp_hidden_dim=512,
    mlp_num_layers=6,
    mlp_time_embedding_dim=16,
    mlp_activation="silu",
    freeze_embedding=True,
    condition_mask=None,
    **kwargs,
):
    """Build an MLP log-ratio model reusing the score model's observation encoder.

    The encoder is the embedding network trained end-to-end with the score model
    (``embed_x=True``), following Gloeckler et al. (2024) App. A3.2.  It is a
    pure function of the clean observation, which is what makes this reuse
    well-defined: in the targeted-inference setting the observation tokens are
    always conditioned, hence never noised.

    Args:
        freeze_embedding: Keep the pretrained encoder fixed and outside the
            trainable parameter tree (default).  Set False to fine-tune it,
            in which case its weights are warm-started from the score model.

    Raises:
        ValueError: If the score model was not trained with ``embed_x=True``,
            or if the condition mask does not condition exactly the observation
            block (this model assumes ``x`` is fully observed and ``theta``
            fully latent).
    """
    from priorg.sim.methods.guidance_ours import LogRatioMLPEmbedded  # lazy
    from priorg.sim.nn.embedding_net import (
        XEmbeddingNet,
        extract_x_embedding_net,
        split_embedding_params,
    )

    del kwargs

    # Check embed_x first: models pickled before embed_x existed have neither
    # "theta_dim" nor "x_dim" in model_init_params, and a bare KeyError there
    # is far less useful than the explicit message this raises.
    embed_cfg = _embedding_config_from_score_model(score_model)
    p = score_model.model_init_params
    num_nodes = p["num_nodes"]
    theta_dim, x_dim = p["theta_dim"], p["x_dim"]

    if condition_mask is None:
        condition_mask = score_model.condition_mask.astype(bool)
    else:
        condition_mask = jnp.asarray(condition_mask).astype(bool)

    expected = jnp.concatenate(
        [jnp.zeros(theta_dim, dtype=bool), jnp.ones(x_dim, dtype=bool)]
    )
    if not bool(jnp.all(condition_mask == expected)):
        raise ValueError(
            "init_log_ratio_mlp_embedded requires the posterior condition mask "
            "(theta latent, x fully conditioned), because the pretrained "
            "encoder is only defined on a clean, complete observation."
        )

    if mlp_activation == "gelu":
        mlp_act = gelu
    elif mlp_activation == "silu":
        mlp_act = silu
    else:
        raise ValueError(f"Unsupported mlp_activation: {mlp_activation}")

    embed_fn = None
    embed_module_kwargs = None
    if freeze_embedding:
        encode, embed_params, _ = extract_x_embedding_net(
            score_model.params, flatten=True, **embed_cfg
        )
        embed_fn = lambda obs: jax.lax.stop_gradient(encode(embed_params, obs))
    else:
        embed_module_kwargs = dict(embed_cfg)

    def model_fn(t, x, condition_mask, condition_value):
        return LogRatioMLPEmbedded(
            theta_dim=theta_dim,
            x_dim=x_dim,
            embed_fn=embed_fn,
            embed_module_kwargs=embed_module_kwargs,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            time_embedding_dim=mlp_time_embedding_dim,
            act=mlp_act,
        )(t, x, condition_mask, condition_value)

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(model_fn))

    rng, subkey = jax.random.split(rng)
    dummy_condition_mask = condition_mask.reshape(1, num_nodes)
    dummy_condition_value = jnp.zeros((1, num_nodes)).at[:, condition_mask].set(1.0)
    params = init_fn(
        subkey,
        jnp.ones((1,)),
        jnp.ones((1, num_nodes)),
        dummy_condition_mask,
        dummy_condition_value,
    )

    if not freeze_embedding:
        # Warm-start the in-tree copy of the encoder from the score model.
        pretrained, _ = split_embedding_params(score_model.params)
        params = {k: dict(v) for k, v in params.items()}
        n_copied = 0
        for key, value in pretrained.items():
            target = f"log_ratio_mlp_embedded/{key}"
            if target not in params:
                continue
            for name, array in value.items():
                existing = params[target].get(name)
                if existing is not None and existing.shape == array.shape:
                    params[target][name] = array
                    n_copied += 1
        _logger.info(
            "Warm-started observation encoder for fine-tuning: %d arrays copied.",
            n_copied,
        )

    def log_ratio_apply(params, t, x, condition_mask, condition_value):
        result = apply_fn(
            params,
            jnp.atleast_1d(t).reshape(1,),
            x.reshape(1, num_nodes),
            condition_mask.reshape(1, num_nodes),
            condition_value.reshape(1, num_nodes),
        )
        return result.squeeze()

    return LogRatioModelSpec(log_ratio_apply=log_ratio_apply, params=params)
