import json
import math
import random
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import Categorical, Distribution as PyroDistribution
from pyro.distributions import MixtureSameFamily, MultivariateNormal

from priorg.sim.tasks.sbibm_task import GaussianLinear as SBIBMGaussianLinear
from priorg.sim.tasks.sbibm_task import TwoMoons as SBIBMTwoMoons
from priorg.sim.utils.conditional_mask import get_condition_mask_fn


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def base_batch_sampler(key, batch_size, data, node_id, meta_data=None, num_devices=1):
    assert data.ndim == 3, "Data must be 3D, (num_samples, num_nodes, dim)"
    assert node_id.ndim in {1, 2}, "Node id must be 1D or 2D"

    index = jax.random.randint(
        key,
        shape=(num_devices, batch_size),
        minval=0,
        maxval=data.shape[0],
    )
    data_batch = data[index, ...]
    node_id_batch = jnp.repeat(node_id[None, ...], num_devices, axis=0).astype(
        jnp.int32
    )

    if meta_data is None:
        meta_data_batch = None
    elif meta_data.ndim == 3:
        meta_data_batch = meta_data[index, ...]
    else:
        meta_data_batch = jnp.repeat(meta_data[None, ...], num_devices, axis=0)

    return data_batch, node_id_batch, meta_data_batch


def _torch_from_any(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _backend_from_torch(x: torch.Tensor, backend: str):
    if backend == "torch":
        return x
    if backend == "numpy":
        return x.detach().cpu().numpy()
    if backend == "jax":
        return jnp.array(x.detach().cpu().numpy())
    raise ValueError(f"Unknown backend: {backend}")


def _seed_from_key(key) -> Optional[int]:
    if key is None:
        return None
    return int(np.asarray(key).reshape(-1)[-1])


def _maybe_seed_torch(key):
    seed = _seed_from_key(key)
    if seed is not None:
        set_seed(seed)


def _diag_normal(loc, scale):
    return pdist.Independent(
        pdist.Normal(_torch_from_any(loc), _torch_from_any(scale), validate_args=False),
        1,
    )


def _mvn(loc, cov):
    return MultivariateNormal(
        loc=_torch_from_any(loc),
        covariance_matrix=_torch_from_any(cov),
        validate_args=False,
    )


def _uniform(low, high):
    return pdist.Independent(
        pdist.Uniform(_torch_from_any(low), _torch_from_any(high), validate_args=False),
        1,
    )


class Task:
    def __init__(
        self,
        name: str,
        theta_dim: int,
        x_dim: int,
        reference_prior: PyroDistribution,
        target_prior: PyroDistribution,
        simulator_fn: Callable[[torch.Tensor], torch.Tensor],
        backend: str = "jax",
    ) -> None:
        self.name = name
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.dim_para = theta_dim
        self.dim_obs = x_dim
        self.reference_prior = reference_prior
        self.target_prior = target_prior
        self.prior = reference_prior
        self.backend = backend
        self._simulator_fn = simulator_fn
        self.has_exact_log_ratio = True
        self.var_names = [f"theta{i}" for i in range(theta_dim)] + [
            f"x{i}" for i in range(x_dim)
        ]
        # Set by _make_task for tasks that support reference posterior sampling.
        self._reference_posterior_sampler = None

    def get_theta_dim(self):
        return self.theta_dim

    def get_x_dim(self):
        return self.x_dim

    def get_node_id(self):
        return jnp.arange(self.theta_dim + self.x_dim)

    def get_prior(self):
        return self.reference_prior

    def get_prior_dist(self, role: str = "reference"):
        if role == "reference":
            return self.reference_prior
        if role == "target":
            return self.target_prior
        raise ValueError(f"Unknown prior role: {role}")

    def get_reference_prior(self):
        return self.reference_prior

    def get_target_prior(self):
        return self.target_prior

    def get_simulator(self):
        return self._simulator_fn

    def _sample_theta(self, num_samples: int, role: str, key=None):
        _maybe_seed_torch(key)
        return self.get_prior_dist(role).sample((num_samples,))

    def sample_reference_theta(self, num_samples: int, key=None):
        theta = self._sample_theta(num_samples, role="reference", key=key)
        return _backend_from_torch(theta, self.backend)

    def sample_target_theta(self, num_samples: int, key=None):
        theta = self._sample_theta(num_samples, role="target", key=key)
        return _backend_from_torch(theta, self.backend)

    def _simulate(self, key, theta):
        _maybe_seed_torch(key)
        theta_torch = torch.atleast_2d(_torch_from_any(theta))
        x = self._simulator_fn(theta_torch)
        return _backend_from_torch(x, self.backend)

    def get_data(self, num_samples: int, key=None, prior_role: str = "reference"):
        theta = self._sample_theta(num_samples, role=prior_role, key=key)
        x = self._simulator_fn(theta)
        return {
            "theta": _backend_from_torch(theta, self.backend),
            "x": _backend_from_torch(x, self.backend),
        }

    def sample_reference_data(self, num_samples: int, key=None):
        return self.get_data(num_samples=num_samples, key=key, prior_role="reference")

    def sample_target_data(self, num_samples: int, key=None):
        return self.get_data(num_samples=num_samples, key=key, prior_role="target")

    def sample_unshifted(self, batch_size: int, key=None):
        data = self.sample_reference_data(num_samples=batch_size, key=key)
        return data["theta"], data["x"]

    def sample_shifted(self, batch_size: int, key=None):
        data = self.sample_target_data(num_samples=batch_size, key=key)
        return data["theta"], data["x"]

    def get_reference_data(self, num_samples: int, key=None):
        return self.sample_reference_data(num_samples=num_samples, key=key)

    def get_target_data(self, num_samples: int, key=None):
        return self.sample_target_data(num_samples=num_samples, key=key)

    def _log_prob(self, theta, role: str):
        theta_torch = _torch_from_any(theta)
        value = self.get_prior_dist(role).log_prob(theta_torch)
        out_backend = "torch" if isinstance(theta, torch.Tensor) else self.backend
        return _backend_from_torch(value, out_backend)

    def log_reference_prior(self, theta):
        return self._log_prob(theta, role="reference")

    def log_target_prior(self, theta):
        return self._log_prob(theta, role="target")

    def log_density_ratio(self, theta):
        theta_torch = _torch_from_any(theta)
        value = self.target_prior.log_prob(theta_torch) - self.reference_prior.log_prob(theta_torch)
        out_backend = "torch" if isinstance(theta, torch.Tensor) else self.backend
        return _backend_from_torch(value, out_backend)

    def make_jax_log_density_ratio_fn(self):
        """Return a JAX-differentiable scalar callable: theta -> log q(theta)/p(theta).

        Handles MultivariateNormal, Independent(Normal), and MixtureSameFamily Pyro distributions.
        Unlike log_density_ratio(), the returned function is pure JAX and supports
        jax.grad / jax.jit.
        """
        def _jax_log_prob(dist):
            # Try MixtureSameFamily first (most complex)
            mixing_dist = getattr(dist, "mixture_distribution", None)
            component_dist = getattr(dist, "component_distribution", None)
            if mixing_dist is not None and component_dist is not None:
                # MixtureSameFamily: log p(x) = logsumexp(log mix_probs + component_log_probs)
                probs = getattr(mixing_dist, "probs", None)
                if probs is None:
                    probs = torch.softmax(mixing_dist.logits, dim=-1)
                log_mix_probs = jnp.log(jnp.array(probs.detach().cpu().numpy()))

                # Build individual component log_prob functions
                comp_fns = []

                comp_base = getattr(component_dist, "base_dist", None)
                if comp_base is not None and hasattr(comp_base, "loc") and hasattr(comp_base, "scale"):
                    # Component is Independent(Normal)
                    means = jnp.array(comp_base.loc.detach().cpu().numpy())
                    scales = jnp.array(comp_base.scale.detach().cpu().numpy())
                    for k in range(means.shape[0]):
                        mean_k = means[k]
                        scale_k = scales[k]
                        def make_fn(m, s):
                            def fn(theta):
                                return jnp.sum(
                                    -0.5 * jnp.log(2 * jnp.pi) - jnp.log(s)
                                    - 0.5 * ((theta - m) / s) ** 2
                                )
                            return fn
                        comp_fns.append(make_fn(mean_k, scale_k))
                elif hasattr(component_dist, "loc") and hasattr(component_dist, "covariance_matrix"):
                    # Component is MultivariateNormal (batch of them)
                    locs = jnp.array(component_dist.loc.detach().cpu().numpy())
                    covs = jnp.array(component_dist.covariance_matrix.detach().cpu().numpy())
                    for k in range(locs.shape[0]):
                        loc_k = locs[k]
                        cov_k = covs[k]
                        L_k = jnp.linalg.cholesky(cov_k)
                        D = loc_k.shape[0]
                        log_det_k = 2.0 * jnp.sum(jnp.log(jnp.diag(L_k)))
                        def make_fn(l, L, ld):
                            def fn(theta):
                                diff = theta - l
                                v = jnp.linalg.solve(L, diff)
                                return -0.5 * (D * jnp.log(2 * jnp.pi) + ld + jnp.dot(v, v))
                            return fn
                        comp_fns.append(make_fn(loc_k, L_k, log_det_k))

                def fn(theta):
                    comp_log_probs = jnp.array([f(theta) for f in comp_fns])
                    return jax.scipy.special.logsumexp(log_mix_probs + comp_log_probs)
                return fn

            # Try Independent(Uniform) - box prior
            base = getattr(dist, "base_dist", None)
            if base is not None and hasattr(base, "low") and hasattr(base, "high"):
                low_arr = jnp.array(base.low.detach().cpu().numpy())
                high_arr = jnp.array(base.high.detach().cpu().numpy())
                def fn(theta):
                    in_support = jnp.all((theta >= low_arr) & (theta <= high_arr))
                    log_vol = jnp.sum(jnp.log(high_arr - low_arr))
                    return jnp.where(in_support, -log_vol, -jnp.inf)
                return fn

            # Try Independent(Normal) - diagonal Gaussian
            base = getattr(dist, "base_dist", None)
            if base is not None and hasattr(base, "loc") and hasattr(base, "scale"):
                loc = jnp.array(base.loc.detach().cpu().numpy())
                scale = jnp.array(base.scale.detach().cpu().numpy())
                def fn(theta):
                    return jnp.sum(
                        -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scale)
                        - 0.5 * ((theta - loc) / scale) ** 2
                    )
                return fn

            # Try MultivariateNormal
            if hasattr(dist, "loc") and hasattr(dist, "covariance_matrix"):
                loc = jnp.array(dist.loc.detach().cpu().numpy())
                cov = jnp.array(dist.covariance_matrix.detach().cpu().numpy())
                L = jnp.linalg.cholesky(cov)
                D = loc.shape[0]
                log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                def fn(theta):
                    diff = theta - loc
                    v = jnp.linalg.solve(L, diff)
                    return -0.5 * (D * jnp.log(2 * jnp.pi) + log_det + jnp.dot(v, v))
                return fn

            raise NotImplementedError(
                f"Cannot convert distribution {type(dist)} to JAX log_prob. "
                "Supported: MultivariateNormal, Independent(Normal), Independent(Uniform), MixtureSameFamily."
            )

        log_q = _jax_log_prob(self.target_prior)
        log_p = _jax_log_prob(self.reference_prior)

        def log_ratio_fn(theta):
            return log_q(theta) - log_p(theta)

        return log_ratio_fn

    def get_posterior_observations(self, key, num_observations):
        key_mask, key_data = jax.random.split(key, 2)
        condition_mask_fn = get_condition_mask_fn("posterior")
        condition_masks = condition_mask_fn(
            key_mask, num_observations, self.theta_dim, self.x_dim
        )
        data = self.get_data(num_observations, key=key_data)
        return condition_masks, data["x"], data["theta"]

    def get_observation_generator(self, condition_mask_fn="structured_random"):
        condition_mask_fn = get_condition_mask_fn(condition_mask_fn)

        def observation_generator(key):
            while True:
                key, key_sample, key_condition_mask = jax.random.split(key, 3)
                condition_mask = condition_mask_fn(
                    key_condition_mask, 1, self.theta_dim, self.x_dim
                )[0]
                sample = self.get_data(num_samples=1, key=key_sample)
                flat_sample = {
                    **{f"theta{i}": sample["theta"][..., i] for i in range(self.theta_dim)},
                    **{f"x{i}": sample["x"][..., i] for i in range(self.x_dim)},
                }
                conditioned_names = [
                    self.var_names[i]
                    for i in range(len(self.var_names))
                    if condition_mask[i]
                ]
                try:
                    x_o = jnp.concatenate(
                        [flat_sample[var] for var in conditioned_names], axis=-1
                    )
                except Exception:
                    x_o = jnp.array([])
                theta_o = jnp.concatenate(
                    [
                        flat_sample[var]
                        for var in self.var_names
                        if var not in conditioned_names
                    ],
                    axis=-1,
                )
                yield condition_mask, x_o, theta_o

        return observation_generator

    def get_base_mask_fn(self):
        theta_mask = jnp.eye(self.theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((self.theta_dim, self.x_dim), dtype=jnp.bool_))
        base_mask = jnp.block(
            [
                [theta_mask, jnp.zeros((self.theta_dim, self.x_dim))],
                [jnp.ones((self.x_dim, self.theta_dim)), x_mask],
            ]
        ).astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn

    def get_batch_sampler(self):
        return base_batch_sampler

    # ------------------------------------------------------------------
    # Reference posterior sampling
    # ------------------------------------------------------------------

    def _sample_reference_posterior(self, num_samples: int, x_o) -> torch.Tensor:
        """Sample from target posterior p_target(theta | x_o).

        Requires that a `_reference_posterior_sampler` was registered when the
        task was constructed (see `get_task`).
        """
        if self._reference_posterior_sampler is None:
            raise NotImplementedError(
                f"No reference posterior sampler registered for task '{self.name}'. "
                "Turin is not supported; for other tasks check get_task()."
            )
        return self._reference_posterior_sampler(num_samples, _torch_from_any(x_o))

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def generate_and_save_observation(self, seed: int, save_dir) -> dict:
        """Sample one (theta, x) from the target prior and save to JSON.

        Args:
            seed: Integer seed used to seed the RNG before sampling, so the
                observation is reproducible. Also used as part of the filename.
            save_dir: Directory in which to write ``obs_{seed}.json``.

        Returns:
            The observation dict ``{"theta": [...], "x": [...]}``.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _maybe_seed_torch(jnp.array([seed]))
        theta = self._sample_theta(1, role="target").squeeze(0)
        theta_t = _torch_from_any(theta)
        x = self._simulator_fn(theta_t.unsqueeze(0)).squeeze(0)
        obs = {"theta": theta_t.tolist(), "x": x.tolist()}
        with open(save_dir / f"obs_{seed}.json", "w") as f:
            json.dump(obs, f)
        return obs

    def get_saved_observation(self, seed: int, save_dir) -> dict:
        """Load a previously saved observation JSON.

        Returns:
            Dict with keys ``"theta"`` and ``"x"`` (Python lists).
        """
        with open(Path(save_dir) / f"obs_{seed}.json") as f:
            return json.load(f)

    def generate_and_save_reference_posterior(
        self, seed: int, save_dir, num_samples: int = 10000
    ) -> jnp.ndarray:
        """Load saved observation, sample the target posterior, save to NPZ.

        The observation ``obs_{seed}.json`` must exist in ``save_dir`` (call
        ``generate_and_save_observation`` first).  Samples are saved to
        ``ref_{seed}.npz`` under key ``"samples"``.

        Returns:
            JAX array of shape ``(num_samples, theta_dim)``.
        """
        obs = self.get_saved_observation(seed, save_dir)
        x_o = _torch_from_any(jnp.array(obs["x"]))
        samples = self._sample_reference_posterior(num_samples, x_o)
        samples_np = np.array(samples.detach().cpu())
        np.savez(Path(save_dir) / f"ref_{seed}.npz", samples=samples_np)
        return jnp.array(samples_np)

    def get_saved_reference_posterior(self, seed: int, save_dir) -> jnp.ndarray:
        """Load previously saved reference posterior samples.

        Returns:
            JAX array of shape ``(N, theta_dim)``.
        """
        data = np.load(Path(save_dir) / f"ref_{seed}.npz")
        return jnp.array(data["samples"])


def _build_two_moons_simulator(reference_prior: PyroDistribution):
    return SBIBMTwoMoons(p_dist=reference_prior).get_simulator()


def _build_gaussian_linear_simulator(
    reference_prior: PyroDistribution, dim: int, simulator_scale: float
):
    return SBIBMGaussianLinear(
        p_dist=reference_prior,
        dim=dim,
        simulator_scale=simulator_scale,
    ).get_simulator()


def _slcp_simulator(theta: torch.Tensor) -> torch.Tensor:
    theta = torch.atleast_2d(theta)
    t1, t2, t3, t4, t5 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]
    mu = torch.stack([t1, t2], dim=-1)
    s1 = t3**2
    s2 = t4**2
    rho = torch.tanh(t5)
    cov12 = rho * t3 * t4
    eps = 1e-6
    row1 = torch.stack([s1 + eps, cov12], dim=-1)
    row2 = torch.stack([cov12, s2 + eps], dim=-1)
    sigma = torch.stack([row1, row2], dim=-2)
    likelihood = pdist.MultivariateNormal(mu, covariance_matrix=sigma, validate_args=False)
    samples = likelihood.sample((4,))
    return samples.permute(1, 0, 2).reshape(theta.shape[0], -1)


def _oup_simulator(theta: torch.Tensor, x_dim: int = 25, y0: float = 10.0, dt: float = 0.2):
    theta = torch.atleast_2d(theta)
    batch_size = theta.shape[0]
    device = theta.device
    theta1 = theta[:, 0]
    theta2 = theta[:, 1]
    y = torch.zeros(batch_size, x_dim, device=device)
    y[:, 0] = y0
    noise_scale = 0.5 * torch.sqrt(torch.tensor(dt, device=device))

    for t in range(x_dim - 1):
        noise = torch.randn(batch_size, device=device) * noise_scale
        drift = theta1 * (torch.exp(theta2) - y[:, t]) * dt
        y[:, t + 1] = y[:, t] + drift + noise

    return y


def _turin_simulator(theta: torch.Tensor, x_dim: int = 101, bandwidth: float = 0.5e9):
    theta = torch.atleast_2d(theta)
    device = theta.device
    batch_size = theta.shape[0]
    g0, t, lambda0, sigma2 = theta.T
    delta_f = bandwidth / (x_dim - 1)
    k = torch.arange(x_dim, device=device)
    h = torch.zeros(batch_size, x_dim, dtype=torch.cfloat, device=device)
    t_max = 1e-7

    for i in range(batch_size):
        num_paths = torch.poisson(lambda0[i] * torch.tensor(t_max, device=device)).int().item() + 1
        tau = torch.rand(num_paths, device=device) * t_max
        var = g0[i] * torch.exp(-tau / t[i]) / lambda0[i]
        alpha_real = torch.randn(num_paths, device=device) * torch.sqrt(var / 2)
        alpha_imag = torch.randn(num_paths, device=device) * torch.sqrt(var / 2)
        alpha = alpha_real + 1j * alpha_imag
        exp_term = torch.exp(-1j * 2 * torch.pi * delta_f * k[:, None] * tau[None, :])
        h[i] = exp_term @ alpha

    noise_real = torch.randn_like(h.real) * torch.sqrt(sigma2[:, None] / 2)
    noise_imag = torch.randn_like(h.real) * torch.sqrt(sigma2[:, None] / 2)
    y = h + noise_real + 1j * noise_imag
    y_complex = torch.fft.ifft(y, dim=1)
    return 10 * torch.log10(torch.abs(y_complex) ** 2 + 1e-12)


def _make_task(
    name: str,
    reference_prior: PyroDistribution,
    target_prior: PyroDistribution,
    simulator_fn,
    theta_dim: int,
    x_dim: int,
):
    return Task(
        name=name,
        theta_dim=theta_dim,
        x_dim=x_dim,
        reference_prior=reference_prior,
        target_prior=target_prior,
        simulator_fn=simulator_fn,
        backend="jax",
    )


# ------------------------------------------------------------------
# Reference posterior sampler builders
# ------------------------------------------------------------------

def _build_two_moons_posterior_sampler(target_prior: PyroDistribution):
    """Rejection sampler for Two Moons target posterior (reuses SBIBMTwoMoons)."""
    sbibm_tm = SBIBMTwoMoons(p_dist=target_prior)

    def sampler(num_samples: int, x_o: torch.Tensor) -> torch.Tensor:
        return sbibm_tm._sample_reference_posterior(
            num_samples, observation=x_o.unsqueeze(0)
        )

    return sampler


def _build_gaussian_linear_posterior_sampler(
    target_prior: PyroDistribution, dim: int, simulator_scale: float
):
    """Analytical Gaussian posterior for Gaussian Linear (reuses SBIBMGaussianLinear)."""
    sbibm_gl = SBIBMGaussianLinear(
        p_dist=target_prior, dim=dim, simulator_scale=simulator_scale
    )

    def sampler(num_samples: int, x_o: torch.Tensor) -> torch.Tensor:
        return sbibm_gl._sample_reference_posterior(num_samples, observation=x_o)

    return sampler


def _build_slcp_posterior_sampler(
    target_prior: PyroDistribution, num_warmup: int = 500, thinning: int = 5
):
    """NUTS sampler for SLCP target posterior with sign-flip augmentation.

    Likelihood: 4 independent bivariate-Normal draws given theta = (t1..t5).

    The covariance depends on (t3, t4, t5) only through t3**2, t4**2 and the
    off-diagonal tanh(t5) * t3 * t4. Flipping the sign of t3 or t4 alone flips
    the off-diagonal, so it must be paired with a flip of t5 (tanh is odd).
    The likelihood is therefore invariant under
        (t3, t4, t5) -> (-t3, t4, -t5), (t3, -t4, -t5), (-t3, -t4, t5),
    and so is the posterior, provided the target prior is symmetric in each of
    t3, t4, t5 about zero (true for both SLCP target priors). NUTS will only find
    one mode, so we run MCMC for num_samples//4 and generate the other three
    modes with these sign flips.
    """

    def sampler(num_samples: int, x_o: torch.Tensor) -> torch.Tensor:
        from pyro.infer import MCMC, NUTS

        n_mcmc = max(num_samples // 4, 1)
        x_obs = x_o.float().view(4, 2)

        def model():
            theta = pyro.sample("theta", target_prior)
            t1, t2, t3, t4, t5 = theta.unbind()
            mu = torch.stack([t1, t2])
            rho = torch.tanh(t5)
            Sigma = torch.stack(
                [
                    torch.stack([t3 ** 2 + 1e-6, rho * t3 * t4]),
                    torch.stack([rho * t3 * t4, t4 ** 2 + 1e-6]),
                ]
            )
            with pyro.plate("obs", 4):
                pyro.sample(
                    "x", pdist.MultivariateNormal(mu, covariance_matrix=Sigma), obs=x_obs
                )

        mcmc = MCMC(
            NUTS(model, target_accept_prob=0.9),
            num_samples=n_mcmc * thinning,
            warmup_steps=num_warmup,
            num_chains=1,
        )
        mcmc.run()
        base = mcmc.get_samples()["theta"][::thinning][:n_mcmc]  # (n_mcmc, 5)

        # Generate the other 3 symmetric modes. Dims: t3 = 2, t4 = 3, t5 = 4.
        # A single flip of t3 or t4 must be paired with a flip of t5 so that
        # tanh(t5) * t3 * t4 (the covariance off-diagonal) is unchanged.
        flip_t3   = base.clone(); flip_t3[:, [2, 4]] *= -1
        flip_t4   = base.clone(); flip_t4[:, [3, 4]] *= -1
        flip_both = base.clone(); flip_both[:, 2:4]  *= -1

        samples = torch.cat([base, flip_t3, flip_t4, flip_both], dim=0)
        idx = torch.randperm(samples.shape[0])
        return samples[idx][:num_samples]

    return sampler


def _train_posterior_score_model(
    key,
    target_prior: PyroDistribution,
    simulator_fn,
    theta_dim: int,
    x_dim: int,
    num_train_samples: int = 500_000,
    num_steps: int = 100_000,
    batch_size: int = 512,
    print_every: int = 20_000,
):
    """Train an AllConditionalScoreModel for posterior sampling p_target(theta | x).

    Uses the same architecture and diffusion SDE as the base score model but trains only
    on the posterior condition mask (x observed, theta latent) with more data and steps.
    Imports are deferred to avoid circular dependency with methods.py.
    """
    import optax
    from functools import partial as fn_partial
    from priorg.sim.methods.methods import (
        AllConditionalScoreModel,
        scalar_transformer_model,
        init_sde_related,
        mean_std_per_node_id,
        get_z_score_fn,
    )
    from priorg.sim.nn.loss_fn import denoising_score_matching_loss

    # 1. Simulate training data from the target prior.
    print(f"Simulating {num_train_samples} samples from target prior...", flush=True)
    theta_t = target_prior.sample((num_train_samples,))
    x_t = simulator_fn(theta_t)
    theta_np = np.array(theta_t.detach().cpu())
    x_np = np.array(x_t.detach().cpu())
    data = jnp.concatenate([jnp.array(theta_np), jnp.array(x_np)], axis=-1)[..., jnp.newaxis]
    node_id = jnp.arange(theta_dim + x_dim)

    # 2. Z-score.
    mean_per_node_id, std_per_node_id = mean_std_per_node_id(data, node_id)
    z_score_fn, un_z_score_fn = get_z_score_fn(mean_per_node_id, std_per_node_id)
    data = z_score_fn(data, node_id)

    # 3. Diffusion SDE (same defaults as the base score model).
    sde_name = "vpsde"
    sde, T_min, T_max, _weight_fn, output_scale_fn = init_sde_related(data, name=sde_name)
    weight_fn = lambda t: _weight_fn(t).reshape(-1, 1, 1)

    # 4. Transformer score model (same default architecture).
    init_fn, model_fn = scalar_transformer_model(
        num_nodes=theta_dim + x_dim, output_scale_fn=output_scale_fn
    )
    key, key_init = jax.random.split(key)
    params = init_fn(
        key_init,
        jnp.ones((10,)),
        data[:10],
        node_id,
        jnp.zeros_like(data[:10]),
    )

    # 5. Optimizer with linear decay.
    schedule = optax.linear_schedule(5e-4, 5e-6, num_steps // 2, num_steps // 2)
    optimizer = optax.chain(optax.adaptive_grad_clip(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    # 6. Fixed posterior condition mask: x observed, theta latent (never changes).
    posterior_mask = jnp.array([False] * theta_dim + [True] * x_dim)

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        batch_size_local = batch.shape[0]
        cm = jnp.broadcast_to(posterior_mask[None], (batch_size_local, theta_dim + x_dim))
        key_t, key_loss = jax.random.split(key)
        times = jax.random.uniform(key_t, (batch_size_local,), minval=T_min, maxval=T_max)

        def loss_fn(p):
            return denoising_score_matching_loss(
                p, key_loss, times, batch,
                loss_mask=cm,
                model_fn=model_fn,
                mean_fn=sde.marginal_mean,
                std_fn=sde.marginal_stddev,
                weight_fn=weight_fn,
                rebalance_loss=False,
                data_id=node_id,
                condition_mask=cm,
                meta_data=None,
                edge_mask=None,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state, loss

    # 7. Training loop.
    n = data.shape[0]
    print(f"Training posterior score model ({num_steps} steps)...", flush=True)
    for i in range(num_steps):
        key, k_idx, k_step = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (batch_size,), 0, n)
        params, opt_state, loss = step_fn(params, opt_state, k_step, data[idx])
        if i % print_every == 0:
            print(f"  step {i:7d}  loss={float(loss):.5f}", flush=True)
    print("Training complete.", flush=True)

    # 8. Wrap in AllConditionalScoreModel (same interface as base model).
    z_score_params = {
        "mean_per_node_id": mean_per_node_id,
        "std_per_node_id": std_per_node_id,
        "z_score_fn": z_score_fn,
        "un_z_score_fn": un_z_score_fn,
    }
    score_model = AllConditionalScoreModel(
        params, model_fn, sde,
        sde_init_params={"data": data, "name": sde_name, "T_min": T_min, "T_max": T_max},
        model_init_params={"num_nodes": theta_dim + x_dim},
        edge_mask_fn_params={"task": "posterior_sampler"},
        z_score_params=z_score_params,
    )
    score_model.set_default_condition_mask(posterior_mask)
    score_model.set_default_node_id(node_id)
    score_model.set_default_edge_mask_fn(lambda *a, **kw: None)
    score_model.set_default_sampling_kwargs(num_steps=500, sampling_method="sde")
    return score_model


def _save_posterior_score_model(model, save_dir: str):
    """Save a trained AllConditionalScoreModel to disk for later loading.

    Saves model parameters and configuration needed to reconstruct the model.
    The model functions (model_fn, sde, z_score functions) will be recreated
    from their init parameters on load.

    Args:
        model: AllConditionalScoreModel instance to save
        save_dir: Directory path to save model files
    """
    import pickle

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model parameters via pickle (params are nested dicts of JAX arrays)
    params_np = jax.tree_util.tree_map(np.asarray, model.params)
    with open(save_path / "model_params.pkl", "wb") as f:
        pickle.dump(params_np, f)

    # Save configuration and metadata as JSON
    config = {
        "sde_init_params": {
            "data": None,  # Don't save data, it's not needed for inference
            "name": model.sde_init_params.get("name", "vpsde"),
            "T_min": float(model.sde_init_params.get("T_min", 1e-4)),
            "T_max": float(model.sde_init_params.get("T_max", 1.0)),
        },
        "model_init_params": model.model_init_params,
        "z_score_params": {
            "mean_per_node_id": np.asarray(model.z_score_params["mean_per_node_id"]).tolist()
            if model.z_score_params is not None else None,
            "std_per_node_id": np.asarray(model.z_score_params["std_per_node_id"]).tolist()
            if model.z_score_params is not None else None,
        },
        "posterior_mask": np.asarray(model.condition_mask).tolist()
        if model.condition_mask is not None else None,
        "node_id": np.asarray(model.node_id).tolist()
        if model.node_id is not None else None,
        "T_min": float(model.T_min),
        "T_max": float(model.T_max),
    }

    with open(save_path / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {save_path}", flush=True)


def _load_posterior_score_model(load_dir: str):
    """Load a previously saved AllConditionalScoreModel from disk.

    Reconstructs the model by:
    1. Loading saved parameters
    2. Recreating model functions from their init parameters
    3. Restoring the model object

    Args:
        load_dir: Directory path containing saved model files

    Returns:
        Reconstructed AllConditionalScoreModel instance
    """
    from priorg.sim.methods.methods import (
        AllConditionalScoreModel,
        scalar_transformer_model,
        init_sde_related,
        get_z_score_fn,
    )

    import pickle

    load_path = Path(load_dir)

    # Load model parameters
    with open(load_path / "model_params.pkl", "rb") as f:
        params_np = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)

    # Load configuration
    with open(load_path / "model_config.json", "r") as f:
        config = json.load(f)

    # Reconstruct model functions
    sde_init_params = config["sde_init_params"]
    model_init_params = config["model_init_params"]
    z_score_data = config["z_score_params"]

    # Recreate SDE
    sde_name = sde_init_params["name"]
    # Create dummy data for sde initialization (only needed for initialization)
    theta_dim_x_dim = model_init_params["num_nodes"]
    dummy_data = jnp.ones((10, theta_dim_x_dim, 1))
    sde, T_min, T_max, _weight_fn, output_scale_fn = init_sde_related(
        dummy_data, name=sde_name
    )

    # Recreate model function (output_scale_fn comes from the recreated SDE)
    init_fn, model_fn = scalar_transformer_model(**model_init_params, output_scale_fn=output_scale_fn)

    # Recreate z_score functions
    if z_score_data["mean_per_node_id"] is not None:
        mean_per_node_id = jnp.array(z_score_data["mean_per_node_id"])
        std_per_node_id = jnp.array(z_score_data["std_per_node_id"])
        z_score_fn, un_z_score_fn = get_z_score_fn(mean_per_node_id, std_per_node_id)
        z_score_params = {
            "mean_per_node_id": mean_per_node_id,
            "std_per_node_id": std_per_node_id,
            "z_score_fn": z_score_fn,
            "un_z_score_fn": un_z_score_fn,
        }
    else:
        z_score_params = None

    # Recreate model
    sde_init_params_full = {
        "data": dummy_data,
        "name": sde_name,
        "T_min": T_min,
        "T_max": T_max,
    }

    score_model = AllConditionalScoreModel(
        params, model_fn, sde,
        sde_init_params=sde_init_params_full,
        model_init_params=model_init_params,
        edge_mask_fn_params={"task": "posterior_sampler"},
        z_score_params=z_score_params,
    )

    # Restore saved state
    if config["posterior_mask"] is not None:
        score_model.set_default_condition_mask(jnp.array(config["posterior_mask"]))
    if config["node_id"] is not None:
        score_model.set_default_node_id(jnp.array(config["node_id"]))

    score_model.set_default_edge_mask_fn(lambda *a, **kw: None)
    score_model.set_default_sampling_kwargs(num_steps=500, sampling_method="sde")

    print(f"Model loaded from {load_path}", flush=True)
    return score_model


def _build_score_posterior_sampler(
    target_prior: PyroDistribution,
    simulator_fn,
    theta_dim: int,
    x_dim: int,
    num_train_samples: int = 1000_000,
    num_steps: int = 60_000,
    model_dir: Optional[str] = None,
):
    """Return a lazy posterior sampler backed by a trained AllConditionalScoreModel.

    The model is trained on first call and cached for all subsequent calls.
    If model_dir is given, checks for a saved model first; if found, loads it
    instead of training. After training, saves the model to model_dir.
    """
    _cache: dict = {}

    def _ensure_trained():
        if "model" not in _cache:
            if model_dir is not None and (Path(model_dir) / "model_params.pkl").exists():
                _cache["model"] = _load_posterior_score_model(model_dir)
                return

            key = jax.random.PRNGKey(0)
            _cache["model"] = _train_posterior_score_model(
                key, target_prior, simulator_fn, theta_dim, x_dim,
                num_train_samples=num_train_samples,
                num_steps=num_steps,
            )

            if model_dir is not None:
                _save_posterior_score_model(_cache["model"], model_dir)

    def sampler(num_samples: int, x_o: torch.Tensor) -> torch.Tensor:
        _ensure_trained()
        model = _cache["model"]
        key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
        x_o_jax = jnp.array(np.array(x_o.detach().cpu()))
        samples = model.sample(num_samples, x_o_jax, rng=key)
        return torch.tensor(np.array(samples), dtype=torch.float32)

    return sampler


def _build_gl_truncated_posterior_sampler(target_prior_uniform, simulator_scale=0.1, dim=6):
    low = target_prior_uniform.base_dist.low.detach().cpu()
    high = target_prior_uniform.base_dist.high.detach().cpu()

    def sampler(num_samples: int, x_o: torch.Tensor) -> torch.Tensor:
        x_o = x_o.float()
        posterior = MultivariateNormal(x_o, torch.eye(dim) * simulator_scale ** 2, validate_args=False)
        collected, needed = [], num_samples
        while needed > 0:
            batch = posterior.sample((needed * 20,))
            mask = ((batch >= low) & (batch <= high)).all(-1)
            kept = batch[mask]
            collected.append(kept)
            needed -= kept.shape[0]
        return torch.cat(collected)[:num_samples]

    return sampler


def _build_oup_truncated_posterior_sampler(
    target_prior_uniform, x_dim: int = 25, dt: float = 0.2, num_warmup: int = 500, thinning: int = 5
):
    noise_scale = 0.5 * (dt ** 0.5)

    def sampler(num_samples: int, x_o: torch.Tensor) -> torch.Tensor:
        from pyro.infer import MCMC, NUTS
        y_obs = x_o.float()
        n_mcmc = num_samples * thinning

        def model():
            theta = pyro.sample("theta", target_prior_uniform)
            theta1, theta2 = theta[0], theta[1]
            means = y_obs[:-1] + theta1 * (torch.exp(theta2) - y_obs[:-1]) * dt
            pyro.sample(
                "y",
                pdist.Independent(pdist.Normal(means, noise_scale), 1),
                obs=y_obs[1:],
            )

        mcmc = MCMC(
            NUTS(model, target_accept_prob=0.9),
            num_samples=n_mcmc,
            warmup_steps=num_warmup,
            num_chains=1,
        )
        mcmc.run()
        return mcmc.get_samples()["theta"][::thinning][:num_samples]

    return sampler


def get_task(name: str, p_dist: Optional[PyroDistribution] = None):
    if p_dist is not None and not isinstance(p_dist, PyroDistribution):
        raise TypeError("Task prior override must be a Pyro distribution")

    if name == "two_moons_id":
        ref = p_dist or _mvn([0.0, 0.0], [[1.5, 0.6], [0.6, 1.5]])
        tgt = _diag_normal([0.3, 0.5], [0.4, 0.3])
        task = _make_task(name, ref, tgt, _build_two_moons_simulator(ref), 2, 2)
        task._reference_posterior_sampler = _build_two_moons_posterior_sampler(tgt)
        return task

    if name == "two_moons_ood":
        ref = p_dist or _mvn([-0.3, 0.2], [[1.4, 0.6], [0.6, 1.4]])
        tgt = _mvn([0.0, 0.0], [[1.2, -0.2], [-0.2, 1.2]])
        task = _make_task(name, ref, tgt, _build_two_moons_simulator(ref), 2, 2)
        task._reference_posterior_sampler = _build_two_moons_posterior_sampler(tgt)
        return task

    if name == "gaussian_linear_6d_id":
        dim = 6
        ref = p_dist or _mvn(
            torch.zeros(dim),
            1.0 * (0.2 * torch.ones(dim, dim) + 0.8 * torch.eye(dim)),
        )
        mix = Categorical(torch.tensor([0.5, 0.5]))
        means = torch.stack([torch.ones(dim) * 0.6, torch.ones(dim) * -0.6])
        cov = 0.2 * (-0.1 * torch.ones(dim, dim) + 1.1 * torch.eye(dim))
        tgt = MixtureSameFamily(
            mix,
            MultivariateNormal(
                means,
                covariance_matrix=torch.stack([cov, cov]),
                validate_args=False,
            ),
            validate_args=False,
        )
        simulator = _build_gaussian_linear_simulator(ref, dim=dim, simulator_scale=0.1)
        task = _make_task(name, ref, tgt, simulator, dim, dim)
        task.simulator_scale = 0.1
        task._reference_posterior_sampler = _build_gaussian_linear_posterior_sampler(
            tgt, dim=dim, simulator_scale=0.1
        )
        return task

    if name == "gaussian_linear_6d_ood":
        dim = 6
        ref = p_dist or _mvn(
            torch.zeros(dim),
            1.0 * (-0.1 * torch.ones(dim, dim) + 1.1 * torch.eye(dim)),
        )
        tgt = _mvn(
            torch.ones(dim) * 1.1,
            1.0 * (0.2 * torch.ones(dim, dim) + 0.8 * torch.eye(dim)),
        )
        simulator = _build_gaussian_linear_simulator(ref, dim=dim, simulator_scale=0.05)
        task = _make_task(name, ref, tgt, simulator, dim, dim)
        task.simulator_scale = 0.05
        task._reference_posterior_sampler = _build_gaussian_linear_posterior_sampler(
            tgt, dim=dim, simulator_scale=0.05
        )
        return task

    if name == "slcp_id":
        ref = p_dist or _diag_normal([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 1.0, 1.0, 1.0])
        tgt = _diag_normal([1.0, 0.5, 0.0, 0.0, 0.0], [0.5, 1.0, 1.0, 0.5, 1.0])
        task = _make_task(name, ref, tgt, _slcp_simulator, 5, 8)
        task._reference_posterior_sampler = _build_slcp_posterior_sampler(tgt)
        return task

    if name == "slcp_ood":
        ref = p_dist or _diag_normal([0.0, 2.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.6, 1.0, 0.8])
        tgt = _diag_normal([1.0, 0.5, 0.0, 0.0, 0.0], [0.5, 1.0, 1.0, 0.5, 1.0])
        task = _make_task(name, ref, tgt, _slcp_simulator, 5, 8)
        task._reference_posterior_sampler = _build_slcp_posterior_sampler(tgt)
        return task

    if name == "oup_id":
        ref = p_dist or _mvn([2.0, 0.0], torch.eye(2)*0.5)
        tgt = _mvn([1.8, 0.0], torch.eye(2) * 0.3)
        sim = lambda theta: _oup_simulator(theta, x_dim=25)
        task = _make_task(name, ref, tgt, sim, 2, 25)
        task._reference_posterior_sampler = _build_score_posterior_sampler(
            tgt, sim, 2, 25, model_dir="checkpoints/oup_id"
        )
        return task

    if name == "oup_ood":
        ref = p_dist or _mvn(torch.ones(2) * 1.5, torch.eye(2) * 0.5)
        tgt = _mvn([1.8, 0.0], torch.eye(2) * 0.5)
        sim = lambda theta: _oup_simulator(theta, x_dim=25)
        task = _make_task(name, ref, tgt, sim, 2, 25)
        task._reference_posterior_sampler = _build_score_posterior_sampler(
            tgt, sim, 2, 25, model_dir="checkpoints/oup_ood"
        )
        return task

    if name == "turin_id":
        ref = p_dist or _diag_normal([0.64, 0.55, 0.5, 0.6], [0.05, 0.08, 0.07, 0.1])
        tgt = _diag_normal([0.6, 0.5, 0.5, 0.5], [0.02, 0.05, 0.05, 0.06])
        sim = lambda theta: _turin_simulator(theta, x_dim=101)
        task = _make_task(name, ref, tgt, sim, 4, 101)
        task._reference_posterior_sampler = _build_score_posterior_sampler(
            tgt, sim, 4, 101, num_steps=60_000, model_dir="checkpoints/turin_id"
        )
        return task

    if name == "turin_ood":
        ref = p_dist or _diag_normal([0.7, 0.6, 0.65, 0.35], [0.04, 0.07, 0.06, 0.06])
        tgt = _diag_normal([0.6, 0.5, 0.5, 0.5], [0.02, 0.05, 0.05, 0.06])
        sim = lambda theta: _turin_simulator(theta, x_dim=101)
        task = _make_task(name, ref, tgt, sim, 4, 101)
        task._reference_posterior_sampler = _build_score_posterior_sampler(
            tgt, sim, 4, 101, num_steps=50_000, model_dir="checkpoints/turin_ood"
        )
        return task

    if name == "oup_truncated":
        ref = p_dist or _mvn([2.0, 0.0], torch.eye(2) * 0.5)
        tgt = _uniform([1.3, -0.7], [3.3, 1.3])
        sim = lambda theta: _oup_simulator(theta, x_dim=25)
        task = _make_task(name, ref, tgt, sim, 2, 25)
        task._reference_posterior_sampler = _build_oup_truncated_posterior_sampler(tgt)
        return task

    if name == "gaussian_linear_6d_truncated":
        dim = 6
        ref = p_dist or _mvn(
            torch.zeros(dim),
            1.0 * (0.2 * torch.ones(dim, dim) + 0.8 * torch.eye(dim)),
        )
        tgt = _uniform(torch.ones(dim) * -0.9, torch.ones(dim) * 1.5)
        simulator = _build_gaussian_linear_simulator(ref, dim=dim, simulator_scale=0.1)
        task = _make_task(name, ref, tgt, simulator, dim, dim)
        task._reference_posterior_sampler = _build_gl_truncated_posterior_sampler(
            tgt, simulator_scale=0.1, dim=dim
        )
        return task

    if name == "turin_truncated":
        ref = p_dist or _diag_normal([0.64, 0.55, 0.5, 0.6], [0.05, 0.08, 0.07, 0.1])
        tgt = _uniform([0.54, 0.39, 0.36, 0.40], [0.74, 0.71, 0.64, 0.80])
        sim = lambda theta: _turin_simulator(theta, x_dim=101)
        task = _make_task(name, ref, tgt, sim, 4, 101)
        task._reference_posterior_sampler = _build_score_posterior_sampler(
            tgt, sim, 4, 101, num_steps=60_000, model_dir="checkpoints/turin_truncated"
        )
        return task

    raise NotImplementedError(f"Task {name} not implemented")
