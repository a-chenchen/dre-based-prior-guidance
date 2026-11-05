"""Train the base score model on simulations from the reference prior.

Writes experiments/models/lightcurve/{model_<id>.pkl, data_<id>.npz}.
"""

import argparse
import logging
import pickle

import jax
import numpy as np

import lightcurve_task as lc
from priorg.sim.methods.methods import run_score_transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--num_simulations", type=int, default=lc.NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    key_data, key_train = jax.random.split(jax.random.PRNGKey(args.seed))
    task = lc.LightcurveTask()
    logging.info(f"Simulating {args.num_simulations} (theta, x) pairs")
    data = task.get_data(args.num_simulations, key=key_data)

    lc.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(lc.data_path(args.model_id), theta=np.asarray(data["theta"]), x=np.asarray(data["x"]))

    model = run_score_transformer(task, data, lc.score_model_config(), rng=key_train)
    with open(lc.model_path(args.model_id), "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved {lc.model_path(args.model_id)}")


if __name__ == "__main__":
    main()
