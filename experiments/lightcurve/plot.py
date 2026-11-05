"""Figures for the light-curve experiment, from results/lightcurve/.

    observations_<target>.pdf        the transit windows used (Figure 6)
    posterior_evolution_<target>.pdf ours vs PriorGuide per step (Figures 4, 7, 8)
    rmse.pdf                         normalised posterior RMSE per step (Figure 9)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import lightcurve_task as lc

# Literature values for (k, a, inc, P): WASP-126 b [Maxted et al. 2016],
# WASP-39 b [Faedi et al. 2011], GJ 436 b [Lanotte et al. 2014].
REFERENCE_VALUES = {
    "wasp126b": [0.0777, 7.634, 87.9, 3.28880],
    "wasp39b": [0.1467, 11.40, 87.83, 4.0552],
    "gj436b": [0.0827, 14.54, 86.86, 2.6439],
}
PARAMS = [0, 1, 2, 4]  # k, a, inc, P
LABELS = ["k", "a", "inc", "P"]
METHODS = {"ours": ("Ours", "#D55E00", "-"), "priorguide": ("PriorGuide", "#009E73", "--")}


def load(method, target, name):
    d = lc.RESULTS_DIR / method / target
    steps = sorted(d.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return [np.load(s / f"{name}.npy") for s in steps]


def plot_observations(target, out_dir):
    xs = load("ours", target, "x_obs")
    fig, axes = plt.subplots(1, len(xs), figsize=(2.4 * len(xs), 2), sharey=True, squeeze=False)
    for i, (ax, x) in enumerate(zip(axes[0], xs)):
        ax.plot(lc.T_SIM, x, lw=0.8, color="k")
        ax.set_title(f"$x_{i + 1}$")
        ax.set_xticks([])
    axes[0, 0].set_ylabel("flux")
    fig.suptitle(f"{lc.TARGETS[target]['name']} light-curve observations")
    fig.tight_layout()
    fig.savefig(out_dir / f"observations_{target}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_evolution(target, out_dir):
    runs = {m: load(m, target, "posterior") for m in METHODS}
    n = min(len(r) for r in runs.values())
    prior = lc.REFERENCE_PRIOR.sample((200_000,)).numpy()
    grids = [np.linspace(prior[:, i].min(), prior[:, i].max(), 300) for i in PARAMS]

    fig, axes = plt.subplots(n, len(PARAMS), figsize=(3.2 * len(PARAMS), 1.4 * n), squeeze=False)
    for col, (i, g) in enumerate(zip(PARAMS, grids)):
        prior_kde = gaussian_kde(prior[:, i])(g)
        for row in range(n):
            ax = axes[row, col]
            ax.fill_between(g, prior_kde, color="0.55", alpha=0.25, lw=0)
            for m, (_, color, ls) in METHODS.items():
                ax.plot(g, gaussian_kde(runs[m][row][:, i])(g), color=color, ls=ls, lw=2.5)
            ax.axvline(REFERENCE_VALUES[target][col], color="red", ls="--", lw=1.5)
            ax.set_xlim(g[0], g[-1])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(LABELS[col], fontsize=16)
            if row < n - 1:
                ax.set_xticklabels([])
            if col == 0:
                lab = "$p(\\theta|x^{(1)})$" if row == 0 else f"$p(\\theta|x^{{(1:{row + 1})}})$"
                ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=13)

    handles = [plt.Line2D([], [], color=c, ls=ls, lw=3, label=l) for l, c, ls in METHODS.values()]
    handles += [plt.matplotlib.patches.Patch(color="0.55", alpha=0.4, label="Prior"),
                plt.Line2D([], [], color="red", ls="--", lw=2, label="Reference value")]
    fig.suptitle(f"{lc.TARGETS[target]['name']}: posterior evolution (k, a, inc, P)", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13)
    fig.savefig(out_dir / f"posterior_evolution_{target}.pdf", bbox_inches="tight")
    plt.close(fig)


def normalised_rmse(samples, ref):
    """Per-parameter RMSE to the reference value, divided by |reference|, averaged."""
    return np.mean([np.sqrt(np.mean((samples[:, i] - r) ** 2)) / abs(r) for i, r in zip(PARAMS, ref)])


def plot_rmse(targets, out_dir):
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4.5), squeeze=False)
    for ax, target in zip(axes[0], targets):
        for m, (label, color, ls) in METHODS.items():
            err = [normalised_rmse(s, REFERENCE_VALUES[target]) for s in load(m, target, "posterior")]
            ax.plot(np.arange(1, len(err) + 1), err, color=color, ls=ls, lw=2.5, marker="o", label=label)
        ax.set(title=lc.TARGETS[target]["name"], xlabel="Number of observations", ylabel="Normalized RMSE", yscale="log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Sequential posterior RMSE (k, a, inc, P)")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=list(lc.TARGETS), choices=list(lc.TARGETS))
    args = parser.parse_args()

    out_dir = lc.RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        plot_observations(target, out_dir)
        plot_evolution(target, out_dir)
    plot_rmse(args.targets, out_dir)
    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
