# OUP Guidance Example — Figure 1

## 1. Background

This example demonstrates test-time prior adaptation for the Ornstein–Uhlenbeck Process (OUP), corresponding to the posterior comparison in **Figure 1**. It uses the in-distribution (`oup_id`) setting and a fixed observation to compare inference under the original training prior with inference adapted to a target prior.

The notebook compares three methods:

- **Simformer:** the pretrained base diffusion model without prior adaptation.
- **KLIEP-based guidance (Ours):** the same base model with an additive guidance term derived from the gradient of a log-ratio model trained using KLIEP.
- **PriorGuide:** the same base model with guidance based on a Gaussian mixture approximation of the prior ratio and Langevin corrections.

The notebook loads an existing base-model checkpoint, trains the KLIEP guidance model, fits the PriorGuide Gaussian mixture model, generates posterior samples, and produces the comparison figure. 


## 2. Data and model files

```text
Example/
├── README.md
├── OUP_guidance_example.ipynb
├── Data/
│   ├── data_0.pkl
│   ├── augmented_data_0_420000.pkl
│   ├── obs_1000003.json
│   └── ref_1000003.npz
└── Model/
    └── model_0.pkl
```

Paths in the following table are relative to `Example/`.

| File | Description |
| --- | --- |
| `Data/data_0.pkl` | Original base-model training dataset of parameter–observation pairs. Used when the augmented training cache needs to be rebuilt. |
| `Data/augmented_data_0_420000.pkl` | Cached dataset for KLIEP training, combining original training data with model-generated parameter–observation pairs. The `420000` suffix denotes the requested number of additional pairs, not the total dataset size. |
| `Data/obs_1000003.json` | Selected observation `x` and its generating parameters `theta`, identified by observation seed `1000003`. |
| `Data/ref_1000003.npz` | Reference posterior samples for the selected observation. |
| `Model/model_0.pkl` | Pretrained Simformer checkpoint for the OUP in-distribution setting, reused by all three methods. |


## 3. Running the example

Keep `Example/` inside the full repository, alongside `priorg/` and `experiments/`. This example depends on those project modules and is not a standalone package. Refer to the [project README](../README.md) and [requirements file](../requirements.txt) for project setup and dependencies.

Open [OUP_guidance_example.ipynb](OUP_guidance_example.ipynb) in Jupyter, select a Python kernel with the project dependencies installed, and run the cells from top to bottom. The kernel working directory should be either the repository root or `Example/`.

The default configuration uses model `0`, observation seed `1000003`, and generates **5,000 posterior samples per method in five batches**. Experiment settings are collected in the notebook's configuration cell. After loading the data and checkpoint, the notebook prepares both guidance methods, samples the posteriors, and plots the two parameter dimensions.

Sample archives are saved using this path pattern:

```text
results/<method>/oup/id/model_0/samples_5000_obs_1000003_run_0.npz
```

Here, `<method>` is `simformer`, `kliep_guidance`, or `priorguide`. Each archive contains `samples`, `theta`, and `x`.

The final figure is displayed in the notebook and saved as:

```text
results/plots/paper/posterior_bias_comparison.pdf
results/plots/paper/posterior_bias_comparison.png
```

All output paths are relative to the **repository root**, not `Example/`. Output directories are created automatically. Rerunning the notebook overwrites sample archives and figures with the same filenames.
