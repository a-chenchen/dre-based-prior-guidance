import math
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyro
import torch
from pyro import distributions as pdist
from pyro.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
    Uniform,
)
from sbibm.tasks.simulator import Simulator
from sbibm.utils.io import get_tensor_from_csv, save_tensor_to_csv
from sbibm.utils.pyro import (
    get_log_prob_fn,
    get_log_prob_grad_fn,
    make_log_prob_grad_fn,
)


class Task:
    def __init__(
        self,
        dim_data: int,
        dim_parameters: int,
        name: str,
        num_observations: int,
        num_posterior_samples: List[int],
        num_simulations: List[int],
        path: Path,
        name_display: Optional[str] = None,
        num_reference_posterior_samples: int = None,
        observation_seeds: Optional[List[int]] = None,
    ):
        """Base class for tasks.

        Args:
            dim_data: Dimensionality of data.
            dim_parameters: Dimensionality of parameters.
            name: Name of task. Should be the name of the folder in which
                the task is stored. Used with `sbibm.get_task(name)`.
            num_observations: Number of different observations for this task.
            num_posterior_samples: Number of posterior samples to generate.
            num_simulations: List containing number of different simulations to
                run this task for.
            path: Path to folder of task.
            name_display: Display name of task, with correct upper/lower-case
                spelling and spaces. Defaults to `name`.
            num_reference_posterior_samples: Number of reference posterior samples
                to generate for this task. Defaults to `num_posterior_samples`.
            observation_seeds: List of observation seeds to use. Defaults to
                a sequence of length `num_observations`. Override to use specific
                seeds.
        """
        self.dim_data = dim_data
        self.dim_parameters = dim_parameters
        self.name = name
        self.num_observations = num_observations
        self.num_posterior_samples = num_posterior_samples
        self.num_simulations = num_simulations
        self.path = path

        self.name_display = name_display if name_display is not None else name
        self.num_reference_posterior_samples = (
            num_reference_posterior_samples
            if num_reference_posterior_samples is not None
            else num_posterior_samples
        )
        self.observation_seeds = (
            observation_seeds
            if observation_seeds is not None
            else [i + 1000000 for i in range(self.num_observations)]
        )

    @abstractmethod
    def get_prior(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_prior_dist(self) -> torch.distributions.Distribution:
        """Get prior distribution"""
        return self.prior_dist

    def get_prior_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters of prior distribution"""
        return self.prior_params

    def get_labels_data(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"data_{i+1}" for i in range(self.dim_data)]

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [f"parameter_{i+1}" for i in range(self.dim_parameters)]

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        return get_tensor_from_csv(path)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Get reference posterior samples for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        return get_tensor_from_csv(path)

    @abstractmethod
    def get_simulator(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Get true parameters (parameters that generated the data) for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        return get_tensor_from_csv(path)

    def save_data(self, path: Union[str, Path], data: torch.Tensor):
        """Save data to a given path"""
        save_tensor_to_csv(path, data, self.get_labels_data())

    def save_parameters(self, path: Union[str, Path], parameters: torch.Tensor):
        """Save parameters to a given path"""
        save_tensor_to_csv(path, parameters, self.get_labels_parameters())

    def flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Flattens data

        Data returned by the simulator is always flattened into 2D Tensors
        """
        return data.reshape(-1, self.dim_data)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data

        Tasks that require more than 2 dimensions for output of the simulator (e.g.
        returning images) may override this method.
        """
        return data.reshape(-1, self.dim_data)

    def _get_log_prob_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        posterior: bool = True,
        implementation: str = "pyro",
        **kwargs: Any,
    ) -> Callable:
        """Gets function returning the unnormalized log probability of the posterior or
        likelihood

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will get likelihood instead of posterior
            implementation: Implementation to use, `pyro` or `experimental`
            kwargs: Additional keywords passed to `sbibm.utils.pyro.get_log_prob_fn`

        Returns:
            `log_prob_fn` that returns log probablities as `batch_size`
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)
        assert type(posterior) is bool

        conditioned_model = self._get_pyro_model(
            num_observation=num_observation,
            observation=observation,
            posterior=posterior,
        )

        log_prob_fn, _ = get_log_prob_fn(
            conditioned_model,
            implementation=implementation,
            **kwargs,
        )

        def log_prob_pyro(parameters):
            assert parameters.ndim == 2

            num_parameters = parameters.shape[0]
            if num_parameters == 1:
                return log_prob_fn({"parameters": parameters})
            else:
                log_probs = []
                for i in range(num_parameters):
                    log_probs.append(
                        log_prob_fn({"parameters": parameters[i, :].reshape(1, -1)})
                    )
                return torch.cat(log_probs)

        def log_prob_experimental(parameters):
            return log_prob_fn({"parameters": parameters})

        if implementation == "pyro":
            return log_prob_pyro
        elif implementation == "experimental":
            return log_prob_experimental
        else:
            raise NotImplementedError

    def _get_log_prob_grad_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        posterior: bool = True,
        implementation: str = "pyro",
        **kwargs: Any,
    ) -> Callable:
        """Gets function returning the unnormalized log probability of the posterior

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will get likelihood instead of posterior
            implementation: Implementation to use, `pyro` or `experimental`
            kwargs: Passed to `sbibm.utils.pyro.get_log_prob_grad_fn`

        Returns:
            `log_prob_grad_fn` that returns gradients as `batch_size` x
            `dim_parameter`
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)
        assert type(posterior) is bool
        assert implementation == "pyro"

        conditioned_model = self._get_pyro_model(
            num_observation=num_observation,
            observation=observation,
            posterior=posterior,
        )
        log_prob_grad_fn, _ = get_log_prob_grad_fn(
            conditioned_model,
            implementation=implementation,
            **kwargs,
        )

        def log_prob_grad_pyro(parameters):
            assert parameters.ndim == 2

            num_parameters = parameters.shape[0]
            if num_parameters == 1:
                grads, _ = log_prob_grad_fn({"parameters": parameters})
                return grads["parameters"].reshape(
                    parameters.shape[0], parameters.shape[1]
                )
            else:
                grads = []
                for i in range(num_parameters):
                    grad, _ = log_prob_grad_fn(
                        {"parameters": parameters[i, :].reshape(1, -1)}
                    )
                    grads.append(grad["parameters"].squeeze())
                return torch.stack(grads).reshape(
                    parameters.shape[0], parameters.shape[1]
                )

        if implementation == "pyro":
            return log_prob_grad_pyro
        else:
            raise NotImplementedError

    def _get_transforms(
        self,
        automatic_transforms_enabled: bool = True,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Gets transforms

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            automatic_transforms_enabled: If True, will automatically construct
                transforms to unconstrained space

        Returns:
            Dict containing transforms
        """
        conditioned_model = self._get_pyro_model(
            num_observation=num_observation, observation=observation
        )

        _, transforms = get_log_prob_fn(
            conditioned_model,
            automatic_transform_enabled=automatic_transforms_enabled,
        )

        return transforms

    def _get_observation_seed(self, num_observation: int) -> int:
        """Get observation seed for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "observation_seed.csv"
        )
        return int(pd.read_csv(path)["observation_seed"][0])

    def _get_pyro_model(
        self,
        posterior: bool = True,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> Callable:
        """Get model function for use with Pyro

        If `num_observation` or `observation` is passed, the model is conditioned.

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly
            posterior: If False, will mask prior which will result in model useful
                for calculating log likelihoods instead of log posterior probabilities
        """
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        prior = self.get_prior()
        simulator = self.get_simulator()

        def model_fn():
            prior_ = pyro.poutine.mask(prior, torch.tensor(posterior))
            return simulator(prior_())

        if observation is not None:
            observation = self.unflatten_data(observation)
            return pyro.condition(model_fn, {"data": observation})
        else:
            return model_fn

    @abstractmethod
    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_samples: Number of samples
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Samples from reference posterior
        """
        raise NotImplementedError

    def _save_observation_seed(self, num_observation: int, observation_seed: int):
        """Save observation seed for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "observation_seed.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[int(observation_seed), int(num_observation)]],
            columns=["observation_seed", "num_observation"],
        ).to_csv(path, index=False)

    def _save_observation(self, num_observation: int, observation: torch.Tensor):
        """Save observed data for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_data(path, observation)

    def _save_reference_posterior_samples(
        self, num_observation: int, reference_posterior_samples: torch.Tensor
    ):
        """Save reference posterior samples for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, reference_posterior_samples)

    def _save_true_parameters(
        self, num_observation: int, true_parameters: torch.Tensor
    ):
        """Save true parameters (parameters that generated the data) for a given observation number"""
        path = (
            self.path
            / self.name
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, true_parameters)

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed

        def run(num_observation, observation_seed, **kwargs):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            self._save_true_parameters(num_observation, true_parameters)

            simulator = self.get_simulator()
            observation = simulator(true_parameters)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,
                    **kwargs,
                )
                num_unique = torch.unique(reference_posterior_samples, dim=0).shape[0]
                assert num_unique == self.num_reference_posterior_samples
                self._save_reference_posterior_samples(
                    num_observation,
                    reference_posterior_samples,
                )

        # Parallel(n_jobs=n_jobs, verbose=50, backend="loky")(
        #     delayed(run)(num_observation, observation_seed, **kwargs)
        #     for num_observation, observation_seed in enumerate(
        #         self.observation_seeds, start=1
        #     )
        # )
        for num_observation, observation_seed in enumerate(
            self.observation_seeds, start=1
        ):
            run(num_observation, observation_seed, **kwargs)


class TwoMoons(Task):
    def __init__(self, p_dist: pdist.Distribution = None):
        """Two Moons"""

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000011,  # observation 1
            1000001,  # observation 2
            1000002,  # observation 3
            1000003,  # observation 4
            1000013,  # observation 5
            1000005,  # observation 6
            1000006,  # observation 7
            1000007,  # observation 8
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=2,
            name="two_moons",
            name_display="Two Moons",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        if p_dist is None:
            prior_bound = 1.0
            self.prior_params = {
                "low": -prior_bound * torch.ones((self.dim_parameters,)),
                "high": +prior_bound * torch.ones((self.dim_parameters,)),
            }
            self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
            self.prior_dist.set_default_validate_args(False)
        else:
            self.prior_dist = p_dist

        self.simulator_params = {
            "a_low": -math.pi / 2.0,
            "a_high": +math.pi / 2.0,
            "base_offset": 0.25,
            "r_loc": 0.1,
            "r_scale": 0.01,
        }

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            a_dist = (
                pdist.Uniform(
                    low=self.simulator_params["a_low"],
                    high=self.simulator_params["a_high"],
                )
                .expand_by((num_samples, 1))
                .to_event(1)
            )
            a = a_dist.sample()

            r_dist = (
                pdist.Normal(
                    self.simulator_params["r_loc"], self.simulator_params["r_scale"]
                )
                .expand_by((num_samples, 1))
                .to_event(1)
            )
            r = r_dist.sample()

            p = torch.cat(
                (
                    torch.cos(a) * r + self.simulator_params["base_offset"],
                    torch.sin(a) * r,
                ),
                dim=1,
            )

            return self._map_fun(parameters, p)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    @staticmethod
    def _map_fun(parameters: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return p + torch.cat((-torch.abs(z0), z1), dim=1)

    @staticmethod
    def _map_fun_inv(parameters: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return x - torch.cat((-torch.abs(z0), z1), dim=1)

    def _likelihood(
        self,
        parameters: torch.Tensor,
        data: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)

        assert parameters.shape[1] == self.dim_parameters
        assert data.shape[1] == self.dim_data

        p = self._map_fun_inv(parameters, data).squeeze(0)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        u = p[:, 0] - self.simulator_params["base_offset"]
        v = p[:, 1]

        r = torch.sqrt(u**2 + v**2)
        L = -0.5 * (
            (r - self.simulator_params["r_loc"]) / self.simulator_params["r_scale"]
        ) ** 2 - 0.5 * torch.log(
            2 * torch.tensor([math.pi]) * self.simulator_params["r_scale"] ** 2
        )

        if len(torch.where(u < 0.0)[0]) > 0:
            L[torch.where(u < 0.0)[0]] = -torch.tensor(math.inf)

        return L if log else torch.exp(L)

    def _get_transforms(
        self,
        *args,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "parameters": torch.distributions.transforms.IndependentTransform(
                torch.distributions.transforms.identity_transform, 1
            )
        }

    def _get_log_prob_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Callable:
        """Get potential function and initial parameters

        The potential function returns the unnormalized negative log
        posterior probability, and is useful to establish and verify
        the reference posterior.

        Args:
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Potential function and proposal for initial parameters, e.g., to start MCMC
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        prior_dist = self.get_prior_dist()

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        observation = self.unflatten_data(observation)

        def log_prob_fn(parameters):
            if type(parameters) == dict:
                parameters = parameters["parameters"]
            return self._likelihood(
                parameters=parameters, data=observation, log=True
            ) + prior_dist.log_prob(parameters)

        return log_prob_fn

    def _get_log_prob_grad_fn(
        self,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Callable:
        lpgf = make_log_prob_grad_fn(
            self._get_log_prob_fn(
                num_observation=num_observation, observation=observation, **kwargs
            )
        )

        def log_prob_grad_fn(parameters):
            num_params = parameters.shape[0]
            grads = []
            for i in range(num_params):
                _, grad = lpgf({"parameters": parameters[i]})
                grads.append(grad)
            if len(grads) > 1:
                return torch.cat(grads).reshape(1, -1)
            else:
                return grad

        return log_prob_grad_fn

    def _sample_reference_posterior(
        self,
        num_samples: int,
        observation: torch.Tensor,  # shape (1,2)
    ) -> torch.Tensor:
        """
        Rejection-sampler in theta-space using prior proposal and likelihood acceptance,
        yielding "partial-moon" shapes under truncated Gaussian or mixture priors.

        Steps:
          1. propose theta from the (truncated) prior
          2. compute likelihood p(x|theta)
          3. accept with probability p(x|theta)/M where M is the max likelihood
        """
        import math

        import torch

        # constants for likelihood
        sigma = 0.01
        mu_r = 0.1
        # maximum likelihood over valid region occurs at rho = mu_r, phi = 0
        M_val = (1 / math.pi) * (1 / (math.sqrt(2 * math.pi) * sigma)) * (1 / mu_r)

        samples = []
        while len(samples) < num_samples:
            # 1) propose theta from truncated prior
            # theta = self.prior_dist.sample(())  # shape (2,)
            if isinstance(self.prior_dist, pdist.MixtureSameFamily):
                comp_dist = self.prior_dist.component_distribution
                new_weights = torch.tensor([0.5, 0.5])
                proposal_dist = pdist.MixtureSameFamily(
                    mixture_distribution=pdist.Categorical(probs=new_weights),
                    component_distribution=comp_dist,
                )
                theta = proposal_dist.sample(())
            else:
                theta = self.prior_dist.sample(())

            # 2) compute forward map deterministic part M(theta)
            t1, t2 = theta[0].item(), theta[1].item()
            rot = 1 / math.sqrt(2)
            m_x = -abs(t1 + t2) * rot
            m_y = (-t1 + t2) * rot

            # 3) compute residual u = observation - [m_x+0.25, m_y]
            x_val = observation.squeeze(0)
            u = x_val - torch.tensor([m_x + 0.25, m_y])
            rho = torch.norm(u).item()
            if rho <= 0:
                continue

            # 4) check half-circle support
            phi = math.atan2(u[1].item(), u[0].item())
            if phi < -math.pi / 2 or phi > math.pi / 2:
                continue

            # 5) compute likelihood p(x|theta)
            p_x = (
                (1 / math.pi)
                * (1 / (math.sqrt(2 * math.pi) * sigma))
                * math.exp(-((rho - mu_r) ** 2) / (2 * sigma**2))
                * (1 / rho)
            )

            # 6) accept with probability p_x / M_val
            if torch.rand(()) < (p_x / M_val):
                samples.append(theta)

        return torch.stack(samples, dim=0)


class GaussianLinear(Task):
    def __init__(
        self,
        p_dist: pdist.Distribution = None,
        dim: int = 10,
        prior_scale: float = 0.1,
        simulator_scale: float = 0.1,
    ):
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.dim = dim
        # simulator noise precision
        self.sim_precision = torch.inverse(simulator_scale * torch.eye(dim))

        # set or build prior
        if p_dist is None:
            self.prior_dist = MultivariateNormal(
                loc=torch.zeros(dim),
                precision_matrix=torch.inverse(prior_scale * torch.eye(dim)),
            )
        else:
            self.prior_dist = p_dist

        # detect prior type
        if isinstance(self.prior_dist, MultivariateNormal):
            self.prior_type = "gaussian"
            self.prior_precision = self.prior_dist.precision_matrix  # [dim,dim]
            self.prior_mean = self.prior_dist.loc  # [dim]
        elif isinstance(self.prior_dist, MixtureSameFamily):
            self.prior_type = "mixture"
            mix = self.prior_dist.mixture_distribution  # Categorical
            comp = (
                self.prior_dist.component_distribution
            )  # MultivariateNormal batch_shape=[K]
            self.prior_weights = mix.probs  # [K]
            self.comp_means = comp.loc  # [K,dim]
            self.comp_covs = comp.covariance_matrix  # [K,dim,dim]
        elif isinstance(self.prior_dist, Uniform):
            self.prior_type = "uniform"
        else:
            raise ValueError(f"Unsupported prior type: {type(self.prior_dist)}")

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls=None):
        def simulator(parameters):
            return pyro.sample(
                "data",
                MultivariateNormal(
                    loc=parameters.float(),
                    precision_matrix=self.sim_precision.float(),
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_reference_posterior(
        self,
        num_observation=None,
        observation=None,
    ):
        # must pass exactly one of num_observation or observation
        assert (num_observation is None) ^ (observation is None)
        if num_observation is not None:
            observation = self.get_observation(num_observation)

        # single-datum case
        x = observation.reshape(-1)  # [dim]
        n = 1

        if self.prior_type == "gaussian":
            # standard Gaussian–Gaussian update
            precision_post = self.prior_precision + n * self.sim_precision
            cov_post = torch.inverse(precision_post)
            mean_post = cov_post @ (
                self.prior_precision @ self.prior_mean + n * (self.sim_precision @ x)
            )
            return MultivariateNormal(loc=mean_post, covariance_matrix=cov_post)

        elif self.prior_type == "mixture":
            # Gaussian‐mixture prior → Gaussian‐mixture posterior
            noise_cov = torch.inverse(self.sim_precision)  # [dim,dim]
            K = self.prior_weights.size(0)

            post_means = []
            post_covs = []
            for k in range(K):
                prior_cov_k = self.comp_covs[k]  # [dim,dim]
                prior_prec_k = torch.inverse(prior_cov_k)
                post_prec_k = prior_prec_k + n * self.sim_precision
                post_cov_k = torch.inverse(post_prec_k)
                post_mean_k = post_cov_k @ (
                    prior_prec_k @ self.comp_means[k] + n * (self.sim_precision @ x)
                )
                post_means.append(post_mean_k)
                post_covs.append(post_cov_k)

            post_means = torch.stack(post_means, dim=0)  # [K,dim]
            post_covs = torch.stack(post_covs, dim=0)  # [K,dim,dim]

            # update mixture weights via p(x|component)
            pred_dist = MultivariateNormal(
                loc=self.comp_means, covariance_matrix=self.comp_covs + noise_cov
            )
            log_px = pred_dist.log_prob(x.expand_as(self.comp_means))  # [K]
            log_w = torch.log(self.prior_weights) + log_px
            post_weights = torch.softmax(log_w, dim=0)  # [K]

            post_comp = MultivariateNormal(loc=post_means, covariance_matrix=post_covs)
            post_mix = Categorical(probs=post_weights)
            return MixtureSameFamily(
                mixture_distribution=post_mix, component_distribution=post_comp
            )

        elif self.prior_type == "uniform":
            # flat prior ⇒ posterior ∝ likelihood
            post_prec = n * self.sim_precision
            post_cov = torch.inverse(post_prec)
            post_mean = x
            return MultivariateNormal(loc=post_mean, covariance_matrix=post_cov)

        else:
            raise RuntimeError("Unknown prior type in posterior computation")

    def _sample_reference_posterior(
        self,
        num_samples,
        num_observation=None,
        observation=None,
    ):
        post = self._get_reference_posterior(
            num_observation=num_observation,
            observation=observation,
        )
        return post.sample((num_samples,))


class GaussianLinearHigh(Task):
    def __init__(
        self,
        p_dist: pdist.Distribution = None,
        dim: int = 20,
        prior_scale: float = 0.1,
        simulator_scale: float = 0.1,
    ):
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear High",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.dim = dim
        # simulator noise precision
        self.sim_precision = torch.inverse(simulator_scale * torch.eye(dim))

        # set or build prior
        if p_dist is None:
            self.prior_dist = MultivariateNormal(
                loc=torch.zeros(dim),
                precision_matrix=torch.inverse(prior_scale * torch.eye(dim)),
            )
        else:
            self.prior_dist = p_dist

        # detect prior type
        if isinstance(self.prior_dist, MultivariateNormal):
            self.prior_type = "gaussian"
            self.prior_precision = self.prior_dist.precision_matrix  # [dim,dim]
            self.prior_mean = self.prior_dist.loc  # [dim]
        elif isinstance(self.prior_dist, MixtureSameFamily):
            self.prior_type = "mixture"
            mix = self.prior_dist.mixture_distribution  # Categorical
            comp = (
                self.prior_dist.component_distribution
            )  # MultivariateNormal batch_shape=[K]
            self.prior_weights = mix.probs  # [K]
            self.comp_means = comp.loc  # [K,dim]
            self.comp_covs = comp.covariance_matrix  # [K,dim,dim]
        elif isinstance(self.prior_dist, Uniform):
            self.prior_type = "uniform"
        else:
            raise ValueError(f"Unsupported prior type: {type(self.prior_dist)}")

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls=None):
        def simulator(parameters):
            return pyro.sample(
                "data",
                MultivariateNormal(
                    loc=parameters.float(),
                    precision_matrix=self.sim_precision.float(),
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_reference_posterior(
        self,
        num_observation=None,
        observation=None,
    ):
        # must pass exactly one of num_observation or observation
        assert (num_observation is None) ^ (observation is None)
        if num_observation is not None:
            observation = self.get_observation(num_observation)

        # single-datum case
        x = observation.reshape(-1)  # [dim]
        n = 1

        if self.prior_type == "gaussian":
            # standard Gaussian–Gaussian update
            precision_post = self.prior_precision + n * self.sim_precision
            cov_post = torch.inverse(precision_post)
            mean_post = cov_post @ (
                self.prior_precision @ self.prior_mean + n * (self.sim_precision @ x)
            )
            return MultivariateNormal(loc=mean_post, covariance_matrix=cov_post)

        elif self.prior_type == "mixture":
            # Gaussian‐mixture prior → Gaussian‐mixture posterior
            noise_cov = torch.inverse(self.sim_precision)  # [dim,dim]
            K = self.prior_weights.size(0)

            post_means = []
            post_covs = []
            for k in range(K):
                prior_cov_k = self.comp_covs[k]  # [dim,dim]
                prior_prec_k = torch.inverse(prior_cov_k)
                post_prec_k = prior_prec_k + n * self.sim_precision
                post_cov_k = torch.inverse(post_prec_k)
                post_mean_k = post_cov_k @ (
                    prior_prec_k @ self.comp_means[k] + n * (self.sim_precision @ x)
                )
                post_means.append(post_mean_k)
                post_covs.append(post_cov_k)

            post_means = torch.stack(post_means, dim=0)  # [K,dim]
            post_covs = torch.stack(post_covs, dim=0)  # [K,dim,dim]

            # update mixture weights via p(x|component)
            pred_dist = MultivariateNormal(
                loc=self.comp_means, covariance_matrix=self.comp_covs + noise_cov
            )
            log_px = pred_dist.log_prob(x.expand_as(self.comp_means))  # [K]
            log_w = torch.log(self.prior_weights) + log_px
            post_weights = torch.softmax(log_w, dim=0)  # [K]

            post_comp = MultivariateNormal(loc=post_means, covariance_matrix=post_covs)
            post_mix = Categorical(probs=post_weights)
            return MixtureSameFamily(
                mixture_distribution=post_mix, component_distribution=post_comp
            )

        elif self.prior_type == "uniform":
            # flat prior ⇒ posterior ∝ likelihood
            post_prec = n * self.sim_precision
            post_cov = torch.inverse(post_prec)
            post_mean = x
            return MultivariateNormal(loc=post_mean, covariance_matrix=post_cov)

        else:
            raise RuntimeError("Unknown prior type in posterior computation")

    def _sample_reference_posterior(
        self,
        num_samples,
        num_observation=None,
        observation=None,
    ):
        post = self._get_reference_posterior(
            num_observation=num_observation,
            observation=observation,
        )
        return post.sample((num_samples,))


if __name__ == "__main__":
    gl = GaussianLinear()
    prior_dist = gl.get_prior_dist()
    simulator = gl.get_simulator()

    theta = prior_dist.sample((1000,))
    x = simulator(theta)
