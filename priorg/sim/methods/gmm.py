import copy
import math
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def init_gmm_params_torch(num_components: int, dim: int) -> Dict[str, nn.Parameter]:
    means = nn.Parameter(torch.randn(num_components, dim))
    log_stds = nn.Parameter(torch.randn(num_components, dim))

    # Initialize log_weights to be uniform
    log_weights = nn.Parameter(torch.log(torch.ones(num_components) / num_components))

    return {"log_weights": log_weights, "means": means, "log_stds": log_stds}


def init_gmm_params_to_samples_torch(
    samples: torch.Tensor, num_components: int, sigma_min: float = 1e-3
) -> dict:
    """
    Initialize GMM parameters using a heuristic based on data samples.
    """
    num_samples, dim = samples.shape
    device = samples.device

    # --- 1. Means: Pick random samples ---
    # torch.randperm generates a random permutation of integers from 0 to n-1
    indices = torch.randperm(num_samples, device=device)[:num_components]
    means = samples[indices].clone()

    # --- 2. Stds: Compute std of closest neighbors ---
    # Compute pairwise Euclidean distance between Means (K, D) and Samples (N, D)
    # Resulting shape: (num_components, num_samples)
    distances = torch.cdist(means, samples)

    # Determine k (number of neighbors to consider)
    # k = max(1, num_samples // num_components)
    k = num_samples // 4

    # Find indices of the k closest samples for each mean
    # largest=False makes it return the smallest elements
    _, closest_indices = torch.topk(distances, k, dim=1, largest=False)  # Shape: (K, k)

    # Select the actual sample vectors using advanced indexing
    # closest_samples shape: (num_components, k, dim)
    closest_samples = samples[closest_indices]

    # Compute standard deviation along the neighbor dimension (dim=1)
    stds = torch.std(closest_samples, dim=1)

    # Safety clamp to prevent log(0) or extremely small stds
    stds = torch.clamp(stds, min=sigma_min)
    # log_stds = torch.log(stds)
    log_stds = torch.log(torch.exp(stds) - 1)

    # --- 3. Weights: Equal probability ---
    # log(1/K) = -log(K)
    log_weights = torch.full(
        (num_components,), -math.log(num_components), device=device
    )

    # Return with requires_grad=True so they are ready for the optimizer
    return {
        "log_weights": log_weights.requires_grad_(True),
        "means": means.requires_grad_(True),
        "log_stds": log_stds.requires_grad_(True),
    }


def gmm_log_prob_torch(
    params: Dict[str, torch.Tensor], x: torch.Tensor, sigma_min: float = 1e-3
) -> torch.Tensor:
    log_weights = params["log_weights"]
    means = params["means"]
    stds = F.softplus(params["log_stds"]) + sigma_min

    x_ = x.unsqueeze(1)
    means_ = means.unsqueeze(0)
    stds_ = stds.unsqueeze(0)

    dist = Normal(means_, stds_)

    log_prob_dims = dist.log_prob(x_)

    log_prob_components = log_prob_dims.sum(dim=-1)

    normalised_log_weights = F.log_softmax(log_weights, dim=0)

    weighted_log_prob = log_prob_components + normalised_log_weights
    return torch.logsumexp(weighted_log_prob, dim=-1)


def gmm_prob_torch(params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return torch.exp(gmm_log_prob_torch(params, x))


def rsample_gmm(params: dict, num_samples: int, sigma_min: float = 1e-3):
    # log_weights = params["log_weights"]
    means = params["means"]
    log_stds = params["log_stds"]
    stds = F.softplus(log_stds) + sigma_min

    means_expanded = means.unsqueeze(0).expand(num_samples, -1, -1)
    stds_expanded = stds.unsqueeze(0).expand(num_samples, -1, -1)

    noise = torch.randn_like(means_expanded)

    samples = noise * stds_expanded + means_expanded

    return samples


def fit_gmm(
    log_target: Callable,
    data_sampler: Callable,
    num_components: int,
    num_iters: int = 10000,
    learning_rate: float = 0.01,
    batch_size: int = 1000,
) -> Dict[str, nn.Parameter]:
    params = init_gmm_params_to_samples_torch(data_sampler(100), num_components)

    optimizer = optim.Adam(params.values(), lr=learning_rate)

    def loss_fn(p_dict, x_batch):
        num_samples, num_components, dim = x_batch.shape
        x_batch_flat = x_batch.reshape(num_samples * num_components, dim)
        gt = log_target(x_batch_flat)
        pred = gmm_log_prob_torch(p_dict, x_batch_flat)
        gt = gt.reshape(num_samples, num_components)
        pred = pred.reshape(num_samples, num_components)
        log_weights = p_dict["log_weights"]
        component_probs = F.softmax(log_weights, dim=-1)
        return torch.sum(component_probs * torch.mean((pred - gt) ** 2, dim=0))

    best_loss = float("inf")
    best_params = copy.deepcopy(params)
    all_losses = []

    for i in range(num_iters):
        x = rsample_gmm(params, batch_size)

        optimizer.zero_grad()

        loss = loss_fn(params, x)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)

        optimizer.step()

        all_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = copy.deepcopy(params)

        if i % 1000 == 0:
            print(
                f"fitting GMM, iteration {i}, loss: {loss.item()}, best loss: {best_loss}",
                flush=True,
            )

    # Return the dictionary of learned parameters
    return params, best_params


class GMMApproximator(torch.nn.Module):
    def __init__(self, num_components: int, dim: int):
        super().__init__()
        self.num_components = num_components
        self.dim = dim
        self.means = torch.nn.Parameter(torch.randn(num_components, dim))
        self.log_stds = torch.nn.Parameter(torch.randn(num_components, dim))
        self.log_weights = torch.nn.Parameter(
            torch.full((num_components,), math.log(1.0 / num_components))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stds = torch.exp(self.log_stds) + 1e-6
        component_log_probs = Normal(x.unsqueeze(1), stds).log_prob(
            self.means
        ).sum(dim=-1)
        return torch.logsumexp(component_log_probs + self.log_weights, dim=-1)


def fit_gmm_v2(
    log_target_fn: Callable,
    data_sampler: Callable,
    num_components: int,
    dim: int,
    num_iters: int = 10000,
    learning_rate: float = 0.01,
    batch_size: int = 1000,
    device: str = "cpu",
) -> "GMMApproximator":
    """Fit GMM to log_target_fn by sampling from data_sampler (e.g. q(theta)).

    Unlike fit_gmm, training samples come from data_sampler rather than the
    GMM itself, which avoids self-referential collapse.

    Returns a GMMApproximator module whose .forward(x) gives log GMM(x).
    """
    model = GMMApproximator(num_components, dim).to(device)

    # Initialise means from data samples
    with torch.no_grad():
        init_samples = data_sampler(num_components).to(device)
        model.means.copy_(init_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(num_iters):
        optimizer.zero_grad()
        x = data_sampler(batch_size).to(device)
        gt = log_target_fn(x).to(device)
        pred = model(x)
        loss = torch.mean((gt - pred) ** 2)
        if torch.isnan(loss):
            print(f"fit_gmm_v2: NaN loss at iteration {i}", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if i % 1000 == 0:
            print(f"fit_gmm_v2 iter {i}, loss: {loss.item():.4f}", flush=True)

    return model
