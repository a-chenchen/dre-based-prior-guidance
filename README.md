# Guidance for Prior Change via Density Ratio Estimation

Official code for the NeurIPS 2026 paper
[Guidance for Prior Change via Density Ratio Estimation](https://arxiv.org/abs/2608.21729)
by Yichen Zang, Jiun-Yi Lin and Song Liu.

This repository covers the numerical benchmark experiments and the sequential posterior updates on TESS light curves.

## Methods

| key | description |
|---|---|
| `guided` | ours (NRE) |
| `kliep_guided` | ours (KLIEP) |
| `wsm_guided` | weighted score matching |
| `priorguide` | PriorGuide |
| `simformer` | the unguided base model |
| `simformer_importance` | SIR |

All six run by default. Use `--methods` to run a subset.

## Tasks

| `--task_name` | posterior | posterior predictive |
|---|---|---|
| `two_moons` | yes | no |
| `gaussian_linear_6d` | yes | no |
| `slcp` | yes | no |
| `oup` | yes | yes |
| `turin` | yes | yes |

`--task_type` sets how far the target prior `q` sits from the reference prior
`p`: `id` (in-distribution) or `ood` (out-of-distribution).

## Installation

Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example

[`Example/`](Example/) has a notebook that reproduces Figure 1 (OUP,
in-distribution) from a pretrained base model and cached data, without running
the pipeline below. See [`Example/README.md`](Example/README.md).

## Running the experiments

The three stages run in order. Outputs are keyed by
`(task_name, task_type, model_id)`.

### 1. Reference data

For each observation seed, this draws an observation from the target prior and
samples its reference posterior.

```bash
python experiments/data/gen_reference_samples.py \
    --task_name two_moons --task_type id
```

Files are written to `experiments/data/reference/<task>_<type>/`.

For `oup` and `turin` the reference posterior comes from a score model trained on simulations from the target prior. The first run trains this model, which takes much longer than the other tasks, and saves it to `checkpoints/` under the current working directory.

### 2. Base score model

```bash
python experiments/train.py \
    --task_name two_moons --task_type id --model_id 0 --save_data
```

Keep `--save_data`: the log-ratio models reuse the training set.

### 3. Guidance and metrics

```bash
python experiments/posterior/run_guidance.py \
    --task_name two_moons --task_type id --model_id 0 \
    --num_runs 1 --run_start_idx 0

python experiments/posterior/aggregate_metrics.py \
    --task_name two_moons --task_type id --model_id 0
```

Results are written under `results/`. The aggregation step writes
`summary_metrics_*.csv` (mean ± std over observations, each averaged over runs).

### Posterior predictive

This experiment conditions on the first third of the observation and scores RMSE on the rest. It reuses the base model and needs only the observations:

```bash
python experiments/data/gen_reference_samples.py \
    --task_name oup --task_type id --obs_only

python experiments/posterior_predictive/run_guidance_pp.py \
    --task_name oup --task_type id --model_id 0

python experiments/posterior_predictive/aggregate_metrics_pp.py \
    --task_name oup --task_type id --model_id 0
```

## TESS light curves

Sequential posterior updates for exoplanet transit parameters from TESS light curves. Each transit window updates the posterior, and the posterior so far becomes the prior for the next window. The targets are `wasp126b`, `wasp39b` and `gj436b` (WASP-126 b, WASP-39 b and GJ 436 b in the paper). This needs the light-curve packages in `requirements.txt` (`lightkurve`, `pytransit`) and internet access to download the TESS data.

Run from `experiments/lightcurve/`:

```bash
cd experiments/lightcurve

python train.py

python run_sequential.py --target wasp39b --method ours
python run_sequential.py --target wasp39b --method priorguide

python plot.py
```
