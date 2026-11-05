"""Utilities for augmenting the guidance model training dataset.

Generates additional (theta, x) pairs by:
  1. Sampling n_unique_x distinct x values from the base training dataset.
  2. For each x, conditioning the base diffusion model on the observed part
     and sampling k = ceil(n_augment / n_unique_x) thetas in one batched call.
  3. Reconstructing the full (theta, x) pairs.

Using batched model.sample calls (one call per unique x, many thetas each)
avoids the memory and compilation overhead of calling model.sample once per
sample (which would require ~n_augment separate calls).

The returned augmented dataset has the same format as the base dataset
{'theta': (N, theta_dim), 'x': (N, x_dim)} and can be passed directly as
the `dataset` argument of `train_log_ratio_model` / `train_score_field_model`.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp


def build_augmented_dataset(
    base_dataset,
    model,
    condition_mask,
    theta_dim,
    n_augment,
    key,
    k_per_x=1000,
    num_steps=25,
    rho=2.0,
):
    """Build an augmented training dataset by sampling from the base model.

    Selects n_unique_x = ceil(n_augment / k_per_x) distinct x values from
    base_dataset and calls model.sample(k_per_x, x_o=x_i, ...) once per x.
    The final dataset is trimmed to exactly n_augment rows.

    Args:
        base_dataset:   dict with keys 'theta' (N, theta_dim) and 'x' (N, x_dim).
        model:          Trained AllConditionalScoreModel.
        condition_mask: bool or 0/1 array of shape (D = theta_dim + x_dim,).
                        True/1 marks the observed (conditioned) dimensions.
                        Theta dimensions must always be 0/False (latent).
        theta_dim:      Number of theta (parameter) dimensions.
        n_augment:      Total number of additional (theta, x) pairs to generate.
        key:            JAX PRNG key.
        k_per_x:        Number of thetas to sample per unique x in one
                        model.sample call (default: 1000).
        num_steps:      Reverse diffusion steps for model sampling (default: 25).
        rho:            Unused; kept for API compatibility.

    Returns:
        augmented_dataset: dict with 'theta' and 'x', each of shape
            (N + n_augment, ...), base data followed by the new samples.
    """
    condition_mask_bool = np.asarray(condition_mask, dtype=bool)
    n_base = np.asarray(base_dataset["theta"]).shape[0]

    # Which x dims are observed vs latent.
    x_cond_mask = condition_mask_bool[theta_dim:]
    x_latent_mask = ~x_cond_mask

    n_unique_x = math.ceil(n_augment / k_per_x)

    # Draw n_unique_x x indices (with replacement).
    key, key_idx = jax.random.split(key)
    indices = np.array(
        jax.random.choice(key_idx, n_base, shape=(n_unique_x,), replace=True)
    )
    x_unique = np.asarray(base_dataset["x"])[indices]  # (n_unique_x, x_dim)

    new_thetas_list = []
    new_xs_list = []

    for i in range(n_unique_x):
        x_i = x_unique[i]                                  # (x_dim,)
        x_cond_i = x_i[x_cond_mask]                       # (cond_obs_dim,)
        key, key_s = jax.random.split(key)

        # One batched call: k thetas for this single x.
        latents = np.array(
            model.sample(
                k_per_x,
                x_o=x_cond_i,
                condition_mask=condition_mask_bool,
                rng=key_s,
                num_steps=num_steps,
            )
        )  # (k_per_x, latent_dim)

        thetas_i = latents[:, :theta_dim]                      # (k_per_x, theta_dim)

        xs_i = np.tile(x_i, (k_per_x, 1)).astype(np.float32)  # (k_per_x, x_dim)
        # For pp tasks, overwrite the latent x dims with model-sampled values.
        if x_latent_mask.any():
            xs_i[:, x_latent_mask] = latents[:, theta_dim:]

        new_thetas_list.append(thetas_i)
        new_xs_list.append(xs_i)

        if (i + 1) % 10 == 0:
            print(
                f"  [augment] {i + 1}/{n_unique_x} x values processed "
                f"({(i + 1) * k_per_x} samples so far)",
                flush=True,
            )

    new_thetas = np.concatenate(new_thetas_list, axis=0)[:n_augment]  # (n_augment, theta_dim)
    new_xs = np.concatenate(new_xs_list, axis=0)[:n_augment]          # (n_augment, x_dim)

    augmented_dataset = {
        "theta": np.concatenate(
            [np.asarray(base_dataset["theta"]), new_thetas], axis=0
        ),
        "x": np.concatenate(
            [np.asarray(base_dataset["x"]), new_xs], axis=0
        ),
    }
    return augmented_dataset
