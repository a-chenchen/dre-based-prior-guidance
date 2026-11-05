"""TESS light-curve task: simulator, priors, data loading and the samplers.

theta = [k, a, inc, t0, P, u1, u2, sigma]
    k      planet-to-star radius ratio
    a      scaled semi-major axis a / R_star
    inc    orbital inclination (degrees)
    t0     transit midpoint offset from the window centre (days)
    P      orbital period (days)
    u1,u2  quadratic limb-darkening coefficients
    sigma  per-cadence Gaussian noise std (relative flux)
x = normalised flux on a 128-point grid over +-0.5 days around a transit.
"""

import logging
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from priorg.sim.methods.data_augment import build_augmented_dataset
from priorg.sim.methods.guidance_dre import sample_unguided
from priorg.sim.methods.guidance_ours import (
    init_log_ratio_model,
    prior_guide_with_log_ratio_model,
    train_log_ratio_model,
)
from priorg.sim.methods.guidance_priorguide import prior_guide_mc_ratio
from priorg.sim.tasks.task import Task, _diag_normal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (Appendix J of the paper)
# ---------------------------------------------------------------------------

THETA_DIM = 8
PARAM_NAMES = ["k", "a", "inc", "t0", "P", "u1", "u2", "sigma"]
T_WINDOW = 128
X_DIM = T_WINDOW
WINDOW_HALF_WIDTH = 0.5
T_SIM = np.linspace(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH, T_WINDOW, dtype=np.float32)
POSTERIOR_MASK = jnp.array([False] * THETA_DIM + [True] * X_DIM)

# t0 is not updated sequentially: every window is re-centred on its own dip.
FROZEN_DIMS = [3]

REFERENCE_PRIOR = _diag_normal(
    loc=[0.10, 10.0, 87.5, 0.0, 3.6, 0.35, 0.18, 0.001],
    scale=[0.04, 3.0, 2.0, 0.05, 0.8, 0.08, 0.06, 3e-4],
)

# Base score model
NUM_SIMULATIONS = 300_000
TRAIN_STEPS = 8_000
TRAIN_BATCH = 2_048
TRAIN_LR = 1e-3

# Augmentation of the log-ratio training set with model-sampled (theta, x)
NUM_AUGMENT = 200_000

# Prior-ratio DRE r(theta) ~ log q(theta)/p(theta), from samples
STATIC_NRE_STEPS = 16_000
STATIC_NRE_BATCH = 4_096
STATIC_NRE_HIDDEN = 128
STATIC_NRE_LAYERS = 3
STATIC_NRE_LR = 1e-3

# Time-dependent log-ratio model (ours)
NRE_STEPS = 26_000
NRE_BATCH = 4_096
NRE_LR = 1e-4

# PriorGuide-MC
PG_MC_SAMPLES = 1000
PG_LANGEVIN_STEPS = 8
PG_LANGEVIN_RATIO = 0.5  # step size = ratio * beta(t)|dt| / 2 = beta(t)|dt| / 4

# Sampling
NUM_SAMPLES = 100_000
NUM_STEPS = 25
RHO = 2.0
BATCH_SIZE = {"ours": 2_000, "priorguide": 1_000}

# Targets: TESS query, number of windows used, and the BLS download path.
# WASP-126 b has 42 sectors; stitching all of them makes BLS run out of memory,
# so it uses a single download.
TARGETS = {
    "wasp126b": {"name": "WASP-126 b", "query": "TIC 25155310", "num_windows": 7, "single_download": True},
    "wasp39b": {"name": "WASP-39 b", "query": "WASP-39 b", "num_windows": 5, "single_download": False},
    "gj436b": {"name": "GJ 436 b", "query": "GJ 436", "num_windows": 7, "single_download": False},
}

MODEL_DIR = ROOT / "experiments" / "models" / "lightcurve"
RESULTS_DIR = ROOT / "results" / "lightcurve"

# ---------------------------------------------------------------------------
# Simulator (PyTransit, quadratic limb darkening)
# ---------------------------------------------------------------------------

_TRANSIT_MODEL = None


def _transit_model():
    global _TRANSIT_MODEL
    if _TRANSIT_MODEL is None:
        try:
            import sqlite3  # noqa: F401
        except ImportError:
            # PyTransit imports sqlite3 indirectly but never uses it; some Python
            # builds lack the C extension.
            import types
            sys.modules["sqlite3"] = types.ModuleType("sqlite3")
        from pytransit import QuadraticModel

        _TRANSIT_MODEL = QuadraticModel()
        _TRANSIT_MODEL.set_data(T_SIM)
    return _TRANSIT_MODEL


def _simulate_flux(k, a, inc, t0, P, u1, u2):
    """Noiseless transit on T_SIM. Parameters are clipped to the physical range."""
    k = float(np.clip(k, 0.005, 0.35))
    a = float(np.clip(a, 1.5, 35.0))
    inc_rad = float(np.clip(inc, 60.0, 90.0)) * np.pi / 180.0
    P = float(np.clip(P, 0.5, 20.0))
    u1 = float(np.clip(u1, 0.01, 0.89))
    u2 = float(np.clip(u2, 0.01, 0.99 - u1))
    try:
        flux = _transit_model().evaluate(k=k, ldc=[u1, u2], t0=float(t0), p=P, a=a, i=inc_rad)
        return np.asarray(flux, dtype=np.float32)
    except Exception:
        return np.ones(T_WINDOW, dtype=np.float32)


def simulator(theta: torch.Tensor) -> torch.Tensor:
    """(N, 8) -> (N, 128): transit flux plus Gaussian noise of std sigma."""
    theta = torch.atleast_2d(theta).detach().cpu().numpy()
    rng = np.random.default_rng()
    out = np.empty((theta.shape[0], T_WINDOW), dtype=np.float32)
    for i, (k, a, inc, t0, P, u1, u2, sigma) in enumerate(theta):
        sigma = float(np.clip(sigma, 1e-5, 0.1))
        out[i] = _simulate_flux(k, a, inc, t0, P, u1, u2) + rng.normal(0.0, sigma, T_WINDOW)
    return torch.from_numpy(out)


class LightcurveTask(Task):
    """Reference-prior task. If `theta`/`x` are given, training batches for the
    log-ratio model are drawn from them instead of calling the simulator."""

    def __init__(self, theta=None, x=None):
        super().__init__(
            name="lightcurve",
            theta_dim=THETA_DIM,
            x_dim=X_DIM,
            reference_prior=REFERENCE_PRIOR,
            target_prior=REFERENCE_PRIOR,
            simulator_fn=simulator,
            backend="jax",
        )
        self._theta = None if theta is None else jnp.asarray(theta)
        self._x = None if x is None else jnp.asarray(x)

    def sample_unshifted(self, batch_size, key=None):
        if self._theta is None:
            return super().sample_unshifted(batch_size, key=key)
        idx = jax.random.randint(key, (batch_size,), 0, self._theta.shape[0])
        return self._theta[idx], self._x[idx]


def score_model_config():
    """Simformer configuration; same architecture as the benchmark tasks."""

    class Cfg(dict):
        __getattr__ = dict.__getitem__

    return Cfg(
        device="cpu",
        sde=Cfg(name="vpsde", beta_max=10.0, beta_min=0.1, T_max=1.0, T_min=1e-5, scale_min=0.001),
        model=Cfg(
            token_dim=40, condition_token_dim=10, condition_token_init_scale=0.1,
            condition_token_init_mean=0.0, condition_mode="concat", time_embedding_dim=128,
            num_heads=4, num_layers=6, attn_size=10, widening_factor=3, num_hidden_layers=1,
            skip_connection_attn=True, skip_connection_mlp=True, layer_norm=True,
        ),
        train=Cfg(
            condition_mask_fn=Cfg(name="structured_random"), edge_mask_fn=Cfg(name="none"),
            max_number_steps=TRAIN_STEPS, min_number_steps=5_000, total_number_steps_scaling=3,
            training_batch_size=TRAIN_BATCH, learning_rate=TRAIN_LR, min_learning_rate=1e-6,
            clip_max_norm=10.0, validation_fraction=0.05, val_repeat=5, val_every=50,
            stop_early_count=5, rebalance_loss=False, z_score_data=True,
        ),
        posterior=Cfg(sampling_method="sde", num_steps=NUM_STEPS, method="euler_maruyama"),
    )


# ---------------------------------------------------------------------------
# TESS data: download, BLS period search, dip-aligned transit windows
# ---------------------------------------------------------------------------

def load_lightcurve(query, single_download=False):
    """Download, sigma-clip, normalise and detrend a TESS light curve."""
    import lightkurve as lk

    if single_download:
        lc = lk.search_lightcurve(query, mission="TESS").download()
        lc = lc.remove_nans().normalize().flatten()
    else:
        search = lk.search_lightcurve(query, mission="TESS", author="SPOC", exptime=120)
        if len(search) == 0:
            search = lk.search_lightcurve(query, mission="TESS", author="SPOC")
        if len(search) == 0:
            search = lk.search_lightcurve(query, mission="TESS", author="TESS-SPOC")
        if len(search) == 0:
            raise ValueError(f"No TESS data found for {query}")
        sectors = []
        for result in search:
            try:
                lc = result.download()
            except Exception as exc:
                logger.warning(f"Skipping sector: {exc}")
                continue
            if lc is None:
                continue
            lc = lc.remove_nans().remove_outliers(sigma=5).normalize().flatten(window_length=401).remove_nans()
            if lc.time.value.max() < 1e6:  # skip sectors with a broken time axis
                sectors.append(lc)
        if not sectors:
            raise RuntimeError(f"No usable sectors for {query}")
        lc = lk.LightCurveCollection(sectors).stitch().remove_nans()
    return lc.time.value.astype(np.float64), lc.flux.value.astype(np.float32)


def bls_period(t, y, min_period=0.5, max_period=10.0):
    """Return (period, epoch) in days from a BLS periodogram."""
    import astropy.units as u
    import lightkurve as lk

    pg = lk.LightCurve(time=t, flux=y).to_periodogram(
        method="bls", minimum_period=min_period, maximum_period=max_period, frequency_factor=10
    )
    period = float(pg.period_at_max_power.to(u.day).value)
    epoch = float(pg.transit_time_at_max_power.value)  # astropy Time, same scale as t
    return period, epoch


def _dip_center(t, y, smooth_window=7):
    """Sub-cadence dip time: Savitzky-Golay smoothing, then a parabola fit at the minimum."""
    from scipy.signal import savgol_filter

    n = len(y)
    sw = max(3, min(smooth_window, n - 1 if (n - 1) % 2 else n - 2))
    try:
        y = savgol_filter(y, sw, polyorder=2)
    except Exception:
        pass
    i = int(np.argmin(y))
    lo, hi = max(0, i - 3), min(n, i + 4)
    if hi - lo >= 3:
        c = np.polyfit(t[lo:hi], y[lo:hi], 2)
        if c[0] > 0 and t[0] <= -c[1] / (2 * c[0]) <= t[-1]:
            return float(-c[1] / (2 * c[0]))
    return float(t[i])


def extract_windows(t, y, period, epoch, min_points=20, search_factor=1.5):
    """Transit windows on T_SIM, each re-centred on its detected dip.

    Windows failing the quality checks (mean flux off by >5%, min < 0.5,
    max > 1.05, or linear trend > 2e-4 per cadence) are dropped.
    """
    from scipy.interpolate import interp1d

    hw = WINDOW_HALF_WIDTH
    windows = []
    for n in range(int(np.ceil((t.min() - epoch) / period)), int(np.floor((t.max() - epoch) / period)) + 1):
        t_bls = epoch + n * period
        wide = np.abs(t - t_bls) < hw * search_factor
        if wide.sum() < min_points:
            continue
        t_dip = t_bls + _dip_center(t[wide] - t_bls, y[wide])
        win = np.abs(t - t_dip) < hw
        if win.sum() < min_points:
            continue
        flux = interp1d(t[win] - t_dip, y[win], fill_value="extrapolate")(T_SIM).astype(np.float32)
        slope = np.polyfit(np.arange(T_WINDOW), flux, 1)[0]
        if abs(flux.mean() - 1) > 0.05 or flux.min() < 0.5 or flux.max() > 1.05 or abs(slope) > 2e-4:
            continue
        windows.append(flux)
    logger.info(f"Extracted {len(windows)} transit windows")
    return windows


def get_windows(target):
    cfg = TARGETS[target]
    t, y = load_lightcurve(cfg["query"], cfg["single_download"])
    windows = extract_windows(t, y, *bls_period(t, y))
    if cfg["num_windows"] is not None:
        windows = windows[: cfg["num_windows"]]
    if len(windows) < 2:
        raise RuntimeError(f"Need at least 2 transit windows for {target}, found {len(windows)}")
    return windows


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_path(model_id):
    return MODEL_DIR / f"model_{model_id}.pkl"


def data_path(model_id):
    return MODEL_DIR / f"data_{model_id}.npz"


def load_score_model(model_id):
    import pickle

    with open(model_path(model_id), "rb") as f:
        model = pickle.load(f)
    model.set_default_node_id(jnp.arange(THETA_DIM + X_DIM))
    model.set_default_condition_mask(POSTERIOR_MASK)
    model.set_default_edge_mask_fn(lambda *a, **kw: None)
    model.set_default_sampling_kwargs(num_steps=NUM_STEPS, sampling_method="sde")
    return model


def load_training_set(model, model_id, key):
    """Base simulations plus NUM_AUGMENT model-sampled pairs (cached on disk)."""
    d = np.load(data_path(model_id))
    theta, x = d["theta"], d["x"]
    aug = MODEL_DIR / f"augmented_data_{model_id}_{NUM_AUGMENT}.npz"
    if aug.exists():
        a = np.load(aug)
        new_theta, new_x = a["theta"], a["x"]
    else:
        logger.info(f"Augmenting the log-ratio training set with {NUM_AUGMENT} pairs")
        out = build_augmented_dataset(
            base_dataset={"theta": theta, "x": x}, model=model,
            condition_mask=np.array(POSTERIOR_MASK), theta_dim=THETA_DIM,
            n_augment=NUM_AUGMENT, key=key, k_per_x=1_000, num_steps=NUM_STEPS,
        )
        new_theta = out["theta"][len(theta):].astype(np.float32)
        new_x = out["x"][len(theta):].astype(np.float32)
        np.savez(aug, theta=new_theta, x=new_x)
    return np.concatenate([theta, new_theta]), np.concatenate([x, new_x])


def train_prior_ratio(posterior_samples, key):
    """Logistic DRE on theta only: returns r(theta) ~ log q(theta)/p(theta),
    with q = current posterior (samples) and p = reference prior."""
    import haiku as hk
    import optax

    q = jnp.asarray(posterior_samples, dtype=jnp.float32)
    p = jnp.asarray(REFERENCE_PRIOR.sample((len(q),)).numpy(), dtype=jnp.float32)

    net = hk.without_apply_rng(hk.transform(
        lambda th: hk.nets.MLP([STATIC_NRE_HIDDEN] * STATIC_NRE_LAYERS + [1], activation=jax.nn.silu)(th).squeeze(-1)
    ))
    key, k_init = jax.random.split(key)
    params = net.init(k_init, q[:1])
    opt = optax.adamw(STATIC_NRE_LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, key):
        kq, kp = jax.random.split(key)
        qb = q[jax.random.randint(kq, (STATIC_NRE_BATCH,), 0, len(q))]
        pb = p[jax.random.randint(kp, (STATIC_NRE_BATCH,), 0, len(p))]
        loss_fn = lambda w: -jnp.mean(jax.nn.log_sigmoid(net.apply(w, qb))) - jnp.mean(jax.nn.log_sigmoid(-net.apply(w, pb)))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for i in range(STATIC_NRE_STEPS):
        key, sub = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, sub)
        if i % 2_000 == 0:
            logger.info(f"  prior-ratio DRE step {i}: loss {float(loss):.4f}")
    return lambda theta: net.apply(params, jnp.atleast_2d(theta))


def train_time_dependent_ratio(task, model, prior_ratio_fn, key):
    """Our guidance model: NRE on diffused states, weighted by exp(r(theta_0))."""
    k_init, k_train = jax.random.split(key)
    spec = init_log_ratio_model(
        model, k_init, model_type="mlp_large", mlp_hidden_dim=256, mlp_num_layers=4,
        mlp_time_embedding_dim=16, mlp_activation="silu", mlp_cond_emb_dim=32,
        condition_mask=POSTERIOR_MASK,
    )
    spec, _ = train_log_ratio_model(
        task=task, diffusion_model=model, condition_mask=POSTERIOR_MASK, model_spec=spec,
        rng=k_train, batch_size=NRE_BATCH, epochs=NRE_STEPS, lr=NRE_LR, patience=0,
        log_density_ratio_fn=prior_ratio_fn,
    )
    return spec


# ---------------------------------------------------------------------------
# Samplers. Each returns a function key -> full state (theta in original units).
# ---------------------------------------------------------------------------

def _init(model, key):
    """Split key into (initial state x_T, key for the reverse diffusion)."""
    k_T, k_run = jax.random.split(key)
    D = THETA_DIM + X_DIM
    T = jnp.array([model.T_max])
    x_T = (model.sde.marginal_mean(T, jnp.zeros(D)).reshape(-1)
           + model.sde.marginal_stddev(T, jnp.zeros(D)).reshape(-1) * jax.random.normal(k_T, (D,)))
    return x_T, k_run


def base_sampler(model, x_obs):
    """Unguided base model, same discretisation as the guided samplers."""
    x_obs = jnp.asarray(x_obs)

    def sample(key):
        x_T, key = _init(model, key)
        return sample_unguided(
            model=model, key=key, condition_mask=POSTERIOR_MASK, x_o=x_obs,
            x_T=x_T, num_steps=NUM_STEPS, rho=RHO,
        )

    return sample


def ours_sampler(model, spec, x_obs):
    x_obs = jnp.asarray(x_obs)

    def sample(key):
        x_T, key = _init(model, key)
        return prior_guide_with_log_ratio_model(
            model=model, key=key, condition_mask=POSTERIOR_MASK, x_o=x_obs, x_T=x_T,
            log_ratio_apply=spec.log_ratio_apply, log_ratio_params=spec.params,
            num_steps=NUM_STEPS, rho=RHO, guidance_scale=1.0,
        )

    return sample


def priorguide_sampler(model, prior_ratio_fn, x_obs):
    """PriorGuide with a Monte Carlo estimate of the guidance term (the target
    prior is only available as samples, so the GMM fit is replaced by r(theta))."""
    zp = model.z_score_params
    cond = np.asarray(POSTERIOR_MASK)
    theta_nodes, x_nodes = model.node_id[~cond], model.node_id[cond]
    x_obs_z = zp["z_score_fn"](jnp.asarray(x_obs), x_nodes)

    def log_ratio_z(theta_z):
        return prior_ratio_fn(zp["un_z_score_fn"](theta_z, theta_nodes)[None, :])[0]

    def sample(key):
        x_T, key = _init(model, key)
        out = prior_guide_mc_ratio(
            model=model, key=key, condition_mask=POSTERIOR_MASK, x_o=x_obs_z, x_T=x_T,
            log_ratio_fn=log_ratio_z, num_steps=NUM_STEPS, rho=RHO, prior_dim=THETA_DIM,
            num_mc_samples=PG_MC_SAMPLES, guidance_scale=1.0,
            langevin_steps=PG_LANGEVIN_STEPS, langevin_ratio=PG_LANGEVIN_RATIO,
        )
        return out.at[:THETA_DIM].set(zp["un_z_score_fn"](out[:THETA_DIM], theta_nodes))

    return sample


def draw(sampler, key, num_samples=NUM_SAMPLES, batch_size=2_000):
    """Run a single-sample sampler in vmapped batches; return theta samples."""
    fn = jax.jit(jax.vmap(sampler))
    keys = jax.random.split(key, num_samples)
    out = [np.asarray(fn(keys[i:i + batch_size])[:, :THETA_DIM]) for i in range(0, num_samples, batch_size)]
    return np.concatenate(out).astype(np.float32)
