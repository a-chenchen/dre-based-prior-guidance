"""Sequential posterior updates over the transit windows of one TESS target.

    step 0: base model                 p(theta | x1)
    step n: guided with prior q^(n-1)  p(theta | x1..x(n+1))

At each step the current posterior samples define the new prior. A
prior-ratio DRE r(theta) ~ log q(theta)/p(theta) is fitted to them, and then
  --method ours        trains a time-dependent log-ratio model weighted by
                       exp(r) and guides with its gradient (Algorithm 1);
  --method priorguide  guides with a Monte Carlo estimate of PriorGuide's
                       guidance term using r, plus Langevin correction.

Writes results/lightcurve/<method>/<target>/step_<n>/{x_obs,posterior}.npy.
"""

import argparse
import logging

import jax
import numpy as np

import lightcurve_task as lc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=list(lc.TARGETS))
    parser.add_argument("--method", required=True, choices=["ours", "priorguide"])
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = lc.RESULTS_DIR / args.method / args.target
    key = jax.random.PRNGKey(args.seed)
    model = lc.load_score_model(args.model_id)
    windows = lc.get_windows(args.target)
    batch_size = lc.BATCH_SIZE[args.method]

    if args.method == "ours":
        key, sub = jax.random.split(key)
        theta, x = lc.load_training_set(model, args.model_id, sub)
        task = lc.LightcurveTask(theta, x)

    samples = None
    for step, x_obs in enumerate(windows):
        logging.info(f"{args.target} | {args.method} | step {step + 1}/{len(windows)}")
        key, k_ratio, k_guide, k_sample = jax.random.split(key, 4)

        if step == 0:
            sampler = lc.base_sampler(model, x_obs)
        else:
            # The prior-ratio DRE sees prior samples in the frozen dims, so r is
            # flat there and those parameters are not updated.
            q = samples.copy()
            frozen = lc.REFERENCE_PRIOR.sample((len(q),)).numpy()
            q[:, lc.FROZEN_DIMS] = frozen[:, lc.FROZEN_DIMS]
            prior_ratio = lc.train_prior_ratio(q, k_ratio)

            if args.method == "ours":
                spec = lc.train_time_dependent_ratio(task, model, prior_ratio, k_guide)
                sampler = lc.ours_sampler(model, spec, x_obs)
            else:
                sampler = lc.priorguide_sampler(model, prior_ratio, x_obs)

        samples = lc.draw(sampler, k_sample, batch_size=batch_size)
        step_dir = out_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        np.save(step_dir / "x_obs.npy", x_obs)
        np.save(step_dir / "posterior.npy", samples)
        logging.info("  posterior mean " + " ".join(f"{n}={m:.4g}" for n, m in zip(lc.PARAM_NAMES, samples.mean(0))))


if __name__ == "__main__":
    main()
