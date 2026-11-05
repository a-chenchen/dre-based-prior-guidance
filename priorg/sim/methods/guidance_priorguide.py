import jax
import jax.numpy as jnp


def _sde_scalar_mean_std(model, t):
    unit = jnp.ones((1,))
    alpha_t = jnp.squeeze(model.sde.marginal_mean(jnp.atleast_1d(t), unit))
    std_t = jnp.squeeze(model.sde.marginal_stddev(jnp.atleast_1d(t), unit))
    return alpha_t, std_t


def predict_x0(score_fn, model, t, x, condition_mask=None):
    """Predicts x0 using the score function and model parameters."""
    del condition_mask
    score = score_fn(t, x)
    alpha_t, std_t = _sde_scalar_mean_std(model, t)
    return (x + (std_t**2) * score) / alpha_t


def posterior_x0_std(model, t, x_dim):
    alpha_t, std_t = _sde_scalar_mean_std(model, t)
    prior_var = jnp.ones((x_dim,))
    posterior_var = 1.0 / (alpha_t**2 / (std_t**2) + 1.0 / prior_var)
    return jnp.sqrt(posterior_var)


# Replace jax.lax.scan with this during debugging
def debug_scan(f, init, xs):
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.array(ys)


def prior_guide_theta_prior_only(
    model,
    key,
    condition_mask,
    x_o,
    x_T,
    theta_prior_log_weights,  # shape (k,)
    theta_prior_mean,  # shape (k, l_p,)
    theta_prior_cov,  # shape (k, l_p,l_p)
    num_steps=500,
    rho=2.0,
    prior_dim=2,
    theta_original_prior_cov=None,  # shape (k, l_p,)
    clipping_range=None,  # shape (2, l_p,)
    verbose=False,
    langevin_steps=4,  # Number of Langevin steps to perform between diffusion steps
    langevin_ratio=0.5,  # Ratio of Langevin noise to diffusion noise
):
    """
    A variant that:
      - Assumes the first prior_dim coordinates of x_T are 'theta',
      - Applies a Gaussian prior only to those dimensions,
      - Leaves the other coords unaffected by the prior directly,
        but they can still be indirectly influenced.
      - Interleaves Langevin dynamics steps between diffusion reverse steps.
    """

    condition_mask_bool = jnp.asarray(condition_mask, dtype=bool)
    condition_mask = condition_mask_bool.astype(x_T.dtype)
    latent_mask = 1 - condition_mask
    cond_size = x_o.shape[0]
    cond_idx = jnp.nonzero(condition_mask_bool, size=cond_size)[0]

    # Time discretization
    ts = jnp.linspace(model.T_min, model.T_max, num_steps) ** rho

    # Score function initialization (no guidance mask here)
    score_fn = model._init_score(
        node_id=model.node_id,
        condition_mask=jnp.zeros_like(condition_mask),
        edge_mask=None,
        meta_data=None,
    )

    # Helper function to compute the guided score at a given point and time
    def compute_guided_score(x, t):
        # Predict the x0-mean
        x_0_mean = predict_x0(score_fn, model, t, x)
        alpha_t, sigma = _sde_scalar_mean_std(model, t)

        if theta_original_prior_cov is not None:
            # Sigma_post based on original prior covariance
            Sigma_post = jnp.linalg.inv(
                jnp.linalg.inv(theta_original_prior_cov)
                + (alpha_t**2 / sigma**2) * jnp.eye(prior_dim)
            )
        else:
            sigma_post = posterior_x0_std(model, t, prior_dim)
            Sigma_post = jnp.diag(sigma_post**2)

        Sigmas_combined = (
            theta_prior_cov + Sigma_post[None, :, :]
        )  # (k,4,4) as (k,4,4) @ (1,4,4)  - k is the number of mixtures
        cur_theta = x_0_mean[:prior_dim]

        if verbose and t == ts[-1]:  # Only print for first step to avoid flooding
            jax.debug.print("Verbose mode --------------------------------------")
            jax.debug.print("t {x}", x=t)
            jax.debug.print("x_0_mean {x}", x=x_0_mean)
            jax.debug.print("sigma {x}", x=sigma)
            jax.debug.print("Sigma_post {x}", x=Sigma_post)
            jax.debug.print("Sigmas_combined {x}", x=Sigmas_combined)
            jax.debug.print("cur_theta {x}", x=cur_theta)

        # Computations to be used in weights
        residual = (
            theta_prior_mean - cur_theta[None, :]
        )  # shape (k, l_p) - k is the number of mixtu
        Sigma_combined_inv = jnp.linalg.inv(Sigmas_combined)  # shape (k, l_p, l_p)
        residual_times_inverse = jnp.matmul(
            residual[:, None, :], Sigma_combined_inv
        )  # shape (k,1,l_p)

        # Build the weight matrix term by term for k mixtures
        first_term = -0.5 * jnp.matmul(
            residual_times_inverse, residual[..., None]
        )  # (k,1,1)
        second_term = -0.5 * jnp.linalg.slogdet(Sigmas_combined)[1]  # (k,)
        weights = jax.nn.softmax(
            first_term.squeeze() + second_term + theta_prior_log_weights
        )  # (k,)

        grad_x0_mean = jax.jacrev(
            lambda x_in: predict_x0(score_fn, model, t, x_in)[:prior_dim]
        )(x)  # shape (l_p, D) — prior_dim VJPs instead of full_dim JVPs

        # Building eqn 18
        # Einsum magic its the sums of the residuals times the inverse of the covariance for each mixture
        # Compute scaled residuals: (2, d) -> (2, d) after applying (d, d) covariance inverse
        scaled_residuals = jnp.einsum("bi, bij -> bj", residual, Sigma_combined_inv)
        weighted_residual = jnp.sum(
            weights[:, None] * scaled_residuals, axis=0
        )  # Final shape: (l_p) rhs of eqn 18 without gradient

        p_y_xt_grad = jnp.matmul(
            weighted_residual[None, :], grad_x0_mean
        ).squeeze()  # shape (D,) full dimension

        # eqn 18 rhs is adjusted by SD and some correction
        x_0_mean_new = x_0_mean + p_y_xt_grad * model.sde.marginal_stddev(
            t, jnp.array([1.0])
        ) ** 2 / model.sde.marginal_mean(t, jnp.array([1.0]))

        if clipping_range is not None:
            x_0_mean_new = jnp.clip(x_0_mean_new, clipping_range[0], clipping_range[1])
        else:
            x_0_mean_new = jnp.clip(
                x_0_mean_new, -50, 50
            )  # change this to something sensible

        if verbose and t == ts[-1]:
            jax.debug.print("x_0_mean_new {x}", x=x_0_mean_new)

        # Tweedies formula conversion for guided score
        guided_score = (
            x_0_mean_new * alpha_t - x
        ) / sigma**2

        return guided_score

    # We'll define the scanning function that goes from t1 -> t0
    def scan_fn(carry, t0):
        key, t1, x1 = carry
        dt = t0 - t1

        # --- First run Langevin dynamics steps (at time t1) ---
        # Expand x_o to full dimensions for conditioning
        x_o_full = jnp.zeros_like(x1)
        x_o_full = x_o_full.at[cond_idx].set(x_o)

        # Calculate adaptive step size based on diffusion characteristics
        # We'll use the diffusion coefficient at t1 to set the step size
        diffusion_coeff = model.sde.diffusion(t1, x1)
        diffusion_variance = diffusion_coeff**2 * jnp.abs(dt)
        langevin_step_size = (langevin_ratio * diffusion_variance) / 2.0

        # Run multiple Langevin steps at time t1
        x_langevin = x1  # Start from the current point

        def langevin_step(state_carry, _):
            x_state, step_key = state_carry
            step_key, noise_key = jax.random.split(step_key)
            # Use the shared guided score function
            guided_score = compute_guided_score(x_state, t1)

            # Langevin update (only on unconditioned dimensions)
            noise = jax.random.normal(noise_key, shape=x_state.shape)
            grad_step = latent_mask * langevin_step_size * guided_score
            eps = 1e-8
            noise_term = (
                latent_mask * jnp.sqrt(2 * langevin_step_size + eps) * noise
            )

            x_updated = x_state + grad_step + noise_term

            # Ensure conditions are still respected
            x_updated = condition_mask * x_o_full + latent_mask * x_updated
            return (x_updated, step_key), x_updated

        # Perform multiple Langevin steps
        key, langevin_key = jax.random.split(key)
        init_carry = (x_langevin, langevin_key)
        (final_langevin_x, final_langevin_key), _ = jax.lax.scan(
            langevin_step, init_carry, jnp.arange(langevin_steps)
        )
        # (final_langevin_x, final_langevin_key), _ = debug_scan(
        #     langevin_step, init_carry, jnp.arange(langevin_steps)
        # )

        # Use the result from Langevin steps for the diffusion update
        x1_after_langevin = final_langevin_x
        key = final_langevin_key

        # --- Now perform regular diffusion reverse step ---
        key, subkey = jax.random.split(key)

        # Compute guided score using the same function
        guided_score = compute_guided_score(x1_after_langevin, t1)

        drift_backward = latent_mask * (
            model.sde.drift(t1, x1_after_langevin)
            - model.sde.diffusion(t1, x1_after_langevin) ** 2 * guided_score
        )
        diffusion_backward = latent_mask * (
            model.sde.diffusion(t1, x1_after_langevin)
        )
        xt0_uncond = (
            x1_after_langevin
            + drift_backward * dt
            + diffusion_backward
            * jnp.sqrt(jnp.abs(dt))
            * jax.random.normal(subkey, shape=x1_after_langevin.shape)
        )

        # Apply conditioning
        xt0 = condition_mask * x_o_full + latent_mask * xt0_uncond

        return (key, t0, xt0), xt0

    # Run the chain backwards from T to 0
    carry_init = (key, ts[-1], x_T)
    _, traj = jax.lax.scan(scan_fn, carry_init, ts[::-1][1:])
    # _, traj = debug_scan(scan_fn, carry_init, ts[::-1][1:])
    return traj[-1]


def prior_guide_mc_ratio(
    model,
    key,
    condition_mask,
    x_o,
    x_T,
    log_ratio_fn,        # callable: (x_0: shape (prior_dim,)) -> scalar log q/p
    num_steps=500,
    rho=2.0,
    prior_dim=2,
    num_mc_samples=100,
    guidance_scale=1.0,
    clipping_range=None,  # (min, max) to clip x_0 mean before MC sampling, or None
    verbose=False,
    langevin_steps=8,
    langevin_ratio=0.5,
):
    """
    PriorGuide reverse diffusion using Monte Carlo estimation of
    nabla_{x_t} log E[r_0(x_0) | x_t].

    Mirrors `get_guided_noise_mc` from DiffusionGuidedBaseline, ported to JAX
    following the same structural conventions as `prior_guide_theta_prior_only`.

    At each reverse step the guided score is:

        score_guided(x_t, t) = score(x_t, t)
                              + guidance_scale * grad_{x_t} log E[r_0(x_0)]

    where x_0 ~ N(mu_0t, sigma_post^2 * I) is the Tweedie posterior and
    r_0(x_0) = q(x_0)/p(x_0) is the density ratio. The expectation is
    approximated via logsumexp over `num_mc_samples` draws.

    Args:
        model: AllConditionalScoreModel with .sde attribute.
        key: JAX PRNG key.
        condition_mask: 0/1 array of shape (D,). 1 marks conditioned dims.
        x_o: Observed values at conditioned dims, shape (obs_dim,).
        x_T: Initial noisy state, shape (D,).
        log_ratio_fn: Callable (theta: shape (prior_dim,)) -> scalar.
            Returns log q(theta)/p(theta) for a single latent sample.
            Vmapped over MC samples internally.
        num_steps: Number of reverse diffusion steps.
        rho: Time exponent for time step spacing.
        prior_dim: Number of latent (theta) dimensions (first prior_dim coords of x).
        num_mc_samples: Number of Monte Carlo samples per diffusion step.
        guidance_scale: Multiplicative scale for the guidance gradient.
        clipping_range: Optional (min, max) tuple. When provided, clips the predicted
            x_0 mean (latent dims) before MC sampling to keep samples in bounds.
            E.g. (-5.0, 5.0). None disables clipping.
        verbose: If True, print debug info at the first reverse step.
        langevin_steps: Number of Langevin refinement steps between diffusion steps.
        langevin_ratio: Step-size ratio for Langevin dynamics.

    Returns:
        Final sample, shape (D,).
    """
    condition_mask_bool = jnp.asarray(condition_mask, dtype=bool)
    condition_mask = condition_mask_bool.astype(x_T.dtype)
    latent_mask = 1 - condition_mask
    cond_size = x_o.shape[0]
    cond_idx = jnp.nonzero(condition_mask_bool, size=cond_size)[0]

    ts = jnp.linspace(model.T_min, model.T_max, num_steps) ** rho

    score_fn = model._init_score(
        node_id=model.node_id,
        condition_mask=jnp.zeros_like(condition_mask),
        edge_mask=None,
        meta_data=None,
    )

    def compute_guided_score(x, t, mc_key):
        sigma_post = posterior_x0_std(model, t, prior_dim)
        x0_var = sigma_post**2

        x_0_mean = predict_x0(score_fn, model, t, x)
        x_0_mean_theta = x_0_mean[:prior_dim]
        if clipping_range is not None:
            x_0_mean_theta = jnp.clip(x_0_mean_theta, clipping_range[0], clipping_range[1])

        noise = jax.random.normal(mc_key, shape=(num_mc_samples, prior_dim))
        samples = x_0_mean_theta + noise * sigma_post[None, :]

        # Call log_ratio_fn directly — no gradient needed through it
        log_ratios = jax.vmap(log_ratio_fn)(samples)  # (num_mc_samples,)
        weights = jax.nn.softmax(log_ratios)           # (num_mc_samples,)

        # Importance-weighted mean in theta space
        weighted_mean = jnp.sum(weights[:, None] * samples, axis=0)  # (prior_dim,)
        residual = weighted_mean - x_0_mean_theta                     # (prior_dim,)

        grad_x0_mean = jax.jacrev(
            lambda x_in: predict_x0(score_fn, model, t, x_in)[:prior_dim]
        )(x)  # shape (prior_dim, D) — prior_dim VJPs instead of full_dim JVPs

        scaled_residual = residual / x0_var
        guidance_grad = jnp.matmul(scaled_residual[None, :], grad_x0_mean).squeeze()

        base_score = score_fn(t, x)
        guided_score = base_score + guidance_scale * guidance_grad

        if verbose and t == ts[-1]:
            jax.debug.print("Verbose mode --------------------------------------")
            jax.debug.print("t {x}", x=t)
            jax.debug.print("sigma_post {x}", x=sigma_post)
            jax.debug.print("guidance_grad {x}", x=guidance_grad)
            jax.debug.print("guided_score {x}", x=guided_score)

        return guided_score

    def scan_fn(carry, t0):
        key, t1, x1 = carry
        dt = t0 - t1

        x_o_full = jnp.zeros_like(x1)
        x_o_full = x_o_full.at[cond_idx].set(x_o)

        diffusion_coeff = model.sde.diffusion(t1, x1)
        diffusion_variance = diffusion_coeff ** 2 * jnp.abs(dt)
        langevin_step_size = (langevin_ratio * diffusion_variance) / 2.0

        def langevin_step(state_carry, _):
            x_state, step_key = state_carry
            step_key, noise_key, mc_key = jax.random.split(step_key, 3)
            guided_score = compute_guided_score(x_state, t1, mc_key)
            noise = jax.random.normal(noise_key, shape=x_state.shape)
            grad_step = latent_mask * langevin_step_size * guided_score
            eps = 1e-8
            noise_term = (
                latent_mask * jnp.sqrt(2 * langevin_step_size + eps) * noise
            )
            x_updated = x_state + grad_step + noise_term
            x_updated = condition_mask * x_o_full + latent_mask * x_updated
            return (x_updated, step_key), x_updated

        key, langevin_key = jax.random.split(key)
        init_carry = (x1, langevin_key)
        (final_langevin_x, final_langevin_key), _ = jax.lax.scan(
            langevin_step, init_carry, jnp.arange(langevin_steps)
        )
        # (final_langevin_x, final_langevin_key), _ = debug_scan(
        #     langevin_step, init_carry, jnp.arange(langevin_steps)
        # )

        x1_after_langevin = final_langevin_x
        key = final_langevin_key

        key, subkey, mc_key = jax.random.split(key, 3)
        guided_score = compute_guided_score(x1_after_langevin, t1, mc_key)

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
    return traj[-1]
