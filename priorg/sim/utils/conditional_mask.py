import jax
import jax.numpy as jnp
import jax.random as jrandom

from functools import partial

try:  # methods.py imports this package as `sim.*`, guidance_* as `priorg.sim.*`
    from priorg.sim.nn.embedding_net import chunk_boundaries
except ImportError:  # pragma: no cover - depends on sys.path setup
    from priorg.sim.nn.embedding_net import chunk_boundaries


def sample_random_conditional_mask(
    key, num_samples, theta_dim, x_dim, alpha=1.0, beta=4.0
):
    # More likely to condition on a few nodes
    key1, key2 = jax.random.split(key, 2)
    condition_mask = jax.random.bernoulli(
        key1,
        jax.random.beta(key2, alpha, beta, shape=(num_samples, 1)),
        shape=(num_samples, theta_dim + x_dim),
    ).astype(jnp.bool_)
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


def joint_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * (theta_dim + x_dim)]*num_samples)


def posterior_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * theta_dim + [True] * x_dim]*num_samples)


def likelihood_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[True] * theta_dim + [False] * x_dim]*num_samples)


def partial_data_conditional_mask(key, num_samples, theta_dim, x_dim):
    n_unknown = int(jnp.ceil(x_dim / 2))
    n_known = x_dim - n_unknown
    mask = jnp.array([[False] * theta_dim + [True] * x_dim] * num_samples)
    # Split the key into num_samples keys
    keys = jax.random.split(key, num_samples)
    for i in range(num_samples):
        x_indices = jax.random.choice(keys[i], x_dim, shape=(n_unknown,), replace=False)
        mask = mask.at[i, theta_dim + x_indices].set(False)
    return mask


def partial_half_data_conditional_mask(key, num_samples, theta_dim, x_dim):

    mask = jnp.array([[False] * theta_dim + [True] * x_dim] * num_samples)
    n_unknown = int(jnp.ceil(x_dim / 2))

    first_half = jax.random.bernoulli(key)  # returns True or False with 50% chance

    if first_half:
        mask = mask.at[:, theta_dim:theta_dim+n_unknown].set(False)
    else:
        mask = mask.at[:, theta_dim+n_unknown-1:theta_dim+x_dim].set(False)

    return mask

def all_false_conditional_mask(key, num_samples, theta_dim, x_dim):
    return jnp.array([[False] * (theta_dim + x_dim)] * num_samples)


def theta_random_conditional_mask(key, num_samples, theta_dim, x_dim, p_theta=0.3):
    """Random conditioning within theta, with the observation always conditioned.

    Intended for models trained with an observation embedding network
    (``embed_x=True``): the observation must stay fully conditioned so its
    tokens are always clean, but conditioning across parameters can still be
    learned, preserving all-conditional behaviour in the theta direction.
    """
    theta_mask = jax.random.bernoulli(key, p_theta, shape=(num_samples, theta_dim))
    # Never condition on every parameter: there would be nothing left to learn.
    all_ones_mask = jnp.all(theta_mask, axis=-1, keepdims=True)
    theta_mask = jnp.where(all_ones_mask, False, theta_mask)
    x_mask = jnp.ones((num_samples, x_dim), dtype=jnp.bool_)
    return jnp.concatenate([theta_mask, x_mask], axis=-1)


def block_posterior_conditional_mask(
    key, num_samples, theta_dim, x_dim, num_blocks=2, p_full=0.5
):
    """Posterior masks at observation-block granularity (partial posteriors).

    Mirrors the gravitational-wave setup of Gloeckler et al. (2024) App. A3.2,
    where the targeted conditionals are ``p(theta | x1, x2)``, ``p(theta | x1)``
    and ``p(theta | x2)``: theta is always latent, and each observation *block*
    is either fully observed or fully missing.  Requires the score model to be
    built with ``x_embed_chunked=True``, ``num_x_tokens=num_blocks`` and
    ``predict_x_score=True`` (missing blocks are latent, so they need scores).

    Args:
        num_blocks: Number of observation blocks; must match ``num_x_tokens``.
        p_full: Probability of drawing the full posterior mask (all blocks
            observed) instead of a random non-empty subset of blocks.
    """
    bounds = chunk_boundaries(x_dim, num_blocks)
    key1, key2 = jax.random.split(key, 2)

    full = jax.random.bernoulli(key1, p_full, shape=(num_samples, 1))
    subset = jax.random.bernoulli(key2, 0.5, shape=(num_samples, num_blocks))
    # Guard against the empty subset (which would be the prior, not a posterior).
    none_observed = jnp.all(~subset, axis=-1, keepdims=True)
    subset = jnp.where(none_observed, True, subset)
    block_mask = jnp.where(full, True, subset)

    x_mask = jnp.concatenate(
        [
            jnp.repeat(block_mask[:, i : i + 1], hi - lo, axis=-1)
            for i, (lo, hi) in enumerate(bounds)
        ],
        axis=-1,
    )
    theta_mask = jnp.zeros((num_samples, theta_dim), dtype=jnp.bool_)
    return jnp.concatenate([theta_mask, x_mask], axis=-1)


def sample_strutured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    condition_mask = jax.random.choice(
        key1,
        jnp.array(
            [[False] * (theta_dim + x_dim)]
            + [[False] * theta_dim + [True] * x_dim]
            + [
                [True] * theta_dim + [False] * x_dim,
                jax.random.bernoulli(
                    key2, rnd1_prob, shape=(theta_dim + x_dim,)
                ).astype(jnp.bool_),
                jax.random.bernoulli(
                    key3, rnd2_prob, shape=(theta_dim + x_dim,)
                ).astype(jnp.bool_),
            ]
        ),
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
        axis=0,
    )
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask


def get_condition_mask_fn(name, **kwargs):
    # Simple functions that don't need any kwargs
    if name.lower() == "joint":
        return joint_conditional_mask
    elif name.lower() == "posterior":
        return posterior_conditional_mask
    elif name.lower() == "likelihood":
        return likelihood_conditional_mask
    elif name.lower() == "all_false":
        return all_false_conditional_mask
    # Functions that might need kwargs
    elif name.lower() == "theta_random":
        return partial(theta_random_conditional_mask, **kwargs)
    elif name.lower() == "block_posterior":
        return partial(block_posterior_conditional_mask, **kwargs)
    elif name.lower() == "structured_random":
        return partial(sample_strutured_conditional_mask, **kwargs)
    elif name.lower() == "random":
        return partial(sample_random_conditional_mask, **kwargs)
    elif name.lower() == "partial_data":
        return partial(partial_half_data_conditional_mask, **kwargs)
    elif name.lower() == "partial_data_scrambled":
        return partial(partial_data_conditional_mask, **kwargs)
    else:
        raise NotImplementedError()