import jax
import jax.numpy as jnp
from jax import random
from functools import partial, total_ordering

from jax.tree_util import register_pytree_node_class, tree_flatten
from jaxtyping import Array
from typing import Union, Any, Dict, Any, Callable
from abc import abstractmethod

from jax.scipy.stats import norm

from priorg.sim.methods.sdeint import sdeint


@total_ordering
class Constraint:
    """A constraint checks if a value satisfies the constraint."""

    def __contains__(self, val: Any) -> bool:
        # Should transform the value to satisfy the constraint.
        val_flatten, _ = tree_flatten(val)
        return all(self._is_contained(x) for x in val_flatten)

    def __eq__(self, __value: object) -> bool:
        return self.__class__ == __value.__class__

    def __lt__(self, __value: object) -> bool:
        return issubclass(self.__class__, __value.__class__)

    @abstractmethod
    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        pass

    def __repr__(self) -> str:
        return type(self).__name__.lower()

    def __str__(self) -> str:
        return self.__repr__()


class Real(Constraint):
    """A constraint that checks if a value is real."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.isreal(x).all()
        elif isinstance(x, Constraint):
            return x == self
        else:
            raise TypeError(f"Cannot check if {x} of type {type(x)} is real.")

class Interval(Real):
    """A constraint that checks if a value is in an interval."""

    def __init__(
        self,
        lower: float,
        upper: float,
        closed_left: bool = True,
        closed_right: bool = True,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.closed_left = closed_left
        self.closed_right = closed_right

    def _is_contained(self, x: Array) -> bool:
        if isinstance(x, Array):
            term1 = x >= self.lower if self.closed_left else x > self.lower
            term2 = x <= self.upper if self.closed_right else x < self.upper

            return super()._is_contained(x) and all(term1) and all(term2)
        else:
            is_real = super()._is_contained(x)
            is_interval = isinstance(x, Interval)
            term1 = x.lower >= self.lower if self.closed_left else x.lower > self.lower
            term2 = x.upper <= self.upper if self.closed_right else x.upper < self.upper
            return is_real and is_interval and term1 and term2


class StrictPositive(Interval):
    def __init__(self) -> None:
        super().__init__(0, jnp.inf, closed_left=False)

# Move these constraint definitions to the top, before the classes that use them
real = Real()
strict_positive = StrictPositive()

@register_pytree_node_class
class Empirical:
    """
    Empirical distribution based on discrete values.
    """

    def __init__(self, values: Array, probs: Array | None = None):
        self.values = jnp.atleast_1d(values)
        
        # Reinterpret the values as a batch of independent distributions
        self.num_values = self.values.shape[0]
        # Rest is interpreted as batch shape
        if values.ndim == 1:
            batch_shape = ()
            event_shape = ()
        else:
            batch_shape = self.values.shape[1:]
            event_shape = ()

        if probs is None:
            self.probs = None
        else:
            # assert probs.shape == values.shape, "probs shape mismatch"
            self.probs = jnp.atleast_1d(probs)

        self._batch_shape = batch_shape
        self._event_shape = event_shape

    @property
    def batch_shape(self) -> tuple:
        """Returns the shape over which parameters are batched."""
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """Returns the shape of a single sample (without batching)."""
        return self._event_shape

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        base_index = jnp.arange(0, self.num_values)
        if self.probs is not None:
            base_index = jnp.broadcast_to(base_index, self.probs.shape)
        index = random.choice(
            key, base_index, shape=shape + (1,) * len(self._event_shape), p=self.probs
        )

        samples = jnp.take_along_axis(self.values, index, axis=0)
        return samples

    def log_prob(self, value: Array) -> Array:
        value = jnp.asarray(value)
        mask = jnp.equal(value[..., None], self.values)
        indices = jnp.argmax(mask, axis=-self.values.ndim)
        valid = jnp.any(mask, axis=-self.values.ndim)
        if self.probs is not None:
            probs = self.probs
            while probs.ndim < indices.ndim:
                probs = probs[None, ...]
            while indices.ndim < probs.ndim:
                indices = indices[None, ...]

            log_probs = jnp.take_along_axis(jnp.log(probs), indices, axis=-1)
            log_probs = jnp.where(valid, log_probs, -jnp.inf)
        else:
            log_probs = jnp.where(valid, -jnp.log(self.num_values), -jnp.inf)
        return log_probs

    def prob(self, value: Array) -> Array:
        """Returns the probability density/mass function evaluated at value."""
        return jnp.exp(self.log_prob(value))

    @property
    def mean(self) -> Array:
        if self.probs is None:
            return jnp.mean(self.values, axis=0)
        else:
            return jnp.sum(self.values * self.probs, axis=0)

    @property
    def mode(self) -> Array:
        if self.probs is None:
            return jnp.bincount(self.values).argmax(axis=0)
        else:
            return self.values[jnp.argmax(self.probs)]

    @property
    def variance(self) -> Array:
        if self.probs is None:
            return jnp.var(self.values, axis=0)
        else:
            return jnp.sum((self.values - self.mean) ** 2 * self.probs)

    @property
    def stddev(self) -> Array:
        """Returns the standard deviation of the distribution."""
        return jnp.sqrt(self.variance)

    @property
    def entropy(self) -> Array:
        if self.probs is None:
            return -jnp.log(self.num_values)
        else:
            return -jnp.sum(self.probs * jnp.log(self.probs), axis=0)

    def cdf(self, value: Array) -> Array:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"Empirical(values={self.values.shape}, probs={None if self.probs is None else self.probs.shape})"

    def tree_flatten(self):
        return (self.values, self.probs), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        values, probs = children
        return cls(values, probs)



@register_pytree_node_class
class VESDE:
    def __init__(
        self,
        p0: Any,
        sigma_max: Union[Array, float] = 10.0,
        sigma_min: Union[Array, float] = 0.01,
    ) -> None:
        """Initialize VESDE (Variance Exploding SDE)
        
        Args:
            p0 (Distribution): Initial distribution
            sigma_max (Union[Array, float], optional): Maximum sigma. Defaults to 10.0.
            sigma_min (Union[Array, float], optional): Minimum sigma. Defaults to 0.01.
        """
        self.p0 = p0
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.batch_shape = p0.batch_shape
        self.event_shape = p0.event_shape
        
        # Compute constant used in diffusion
        self._const = jnp.sqrt(2 * jnp.log(sigma_max / sigma_min))
        
        # Define drift and diffusion functions
        shape = p0.event_shape
        d = shape[0] if len(shape) > 0 else 1
        self.drift = lambda t, x: jnp.zeros_like(x)
        self.diffusion = lambda t, x: (
            sigma_min * (sigma_max / sigma_min) ** t * self._const * jnp.ones_like(x)
        )

    def marginal_mean(self, ts: Array, x0=None, **kwargs) -> Array:
        """Compute marginal mean at times ts"""
        if x0 is None:
            mu0 = self.p0.mean
        else:
            mu0 = x0
        
        while ts.ndim < mu0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)
            
        ts, mu0 = jnp.broadcast_arrays(ts, mu0)
        return mu0

    def mean(self, ts: Array, x0=None, **kwargs) -> Array:
        """Compute mean at times ts"""
        shape = ts.shape
        ts = jnp.expand_dims(
            ts,
            axis=(-i for i in range(1, len(self.batch_shape) + len(self.event_shape) + 1)),
        )
        mu = self.marginal_mean(ts)
        return mu.reshape(shape + self.batch_shape + self.event_shape)

    def marginal_variance(self, ts: Array, x0=None, **kwargs) -> Array:
        """Compute marginal variance at times ts"""
        if x0 is None:
            var0 = self.p0.variance
        else:
            var0 = jnp.zeros_like(x0)
            
        while ts.ndim < var0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)
            
        ts, var0 = jnp.broadcast_arrays(ts, var0)
            
        vart = self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * ts)
        var = var0 + vart
        return var
    
    def marginal_stddev(self, t: Array, x0=None, **kwargs) -> Array:
        return jnp.sqrt(self.marginal_variance(t, x0, **kwargs))
    
    def stddev(self, t: Array) -> Array:
        return jnp.sqrt(self.variance(t))

    def variance(self, ts: Array, x0=None, **kwargs) -> Array:
        """Compute variance at times ts"""
        shape = ts.shape
        ts = jnp.expand_dims(
            ts,
            axis=(-i for i in range(1, len(self.batch_shape) + len(self.event_shape) + 1)),
        )
        var = self.marginal_variance(ts)
        return var.reshape(shape + self.batch_shape + self.event_shape)

    def stddev(self, ts: Array, x0=None, **kwargs) -> Array:
        """Compute standard deviation at times ts"""
        return jnp.sqrt(self.variance(ts, x0, **kwargs))

    def sample(self, key: Any, ts: Array, sample_shape=(), **kwargs) -> Array:
        """Sample from the SDE at times ts"""
        key1, key2 = jax.random.split(key)
        x0 = self.p0.sample(key1, sample_shape)
        
        # Flatten and split keys
        x0_flat = x0.reshape(-1, *self.event_shape)
        keys_flat = jax.random.split(key2, x0_flat.shape[0])
        
        # Handle ts dimensionality
        vmap_dim = 0 if ts.ndim > 1 else None
        if ts.ndim > 1:
            ts = ts.reshape(-1, ts.shape[-1])
            
        # Run SDE integration
        _sdeint = partial(sdeint, **kwargs)
        __sdeint = jax.vmap(_sdeint, in_axes=(0, None, None, 0, vmap_dim))
        ys = __sdeint(keys_flat, self.drift, self.diffusion, x0_flat, ts)
        
        # Reshape to correct shape
        return ys.reshape(sample_shape + self.batch_shape + ts.shape + self.event_shape)

    def log_prob(self, x: Array, t: Array, x0=None) -> Array:
        """Compute log probability of x at time t"""
        mu = self.mean(t, x0=x0)
        std = self.stddev(t, x0=x0)
        return jax.scipy.stats.norm.logpdf(x, mu, std)

    def tree_flatten(self):
        """PyTree flattening"""
        return ((self.p0, self.sigma_max, self.sigma_min), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """PyTree unflattening"""
        return cls(*children)


@register_pytree_node_class
class VPSDE:
    def __init__(
        self,
        p0: Any,
        beta_max: Union[Array, float] = 10.0,
        beta_min: Union[Array, float] = 0.1,
    ) -> None:
        self.p0 = p0
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.batch_shape = p0.batch_shape
        self.event_shape = p0.event_shape

        self.drift = lambda t, x: -0.5 * self.beta(t) * x
        self.diffusion = lambda t, x: jnp.sqrt(self.beta(t)) * jnp.ones_like(x)

    def beta(self, t: Array) -> Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_mean(self, ts: Array, x0=None, **kwargs) -> Array:
        if x0 is None:
            mu0 = self.p0.mean
        else:
            mu0 = x0

        while ts.ndim < mu0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)

        ts, mu0 = jnp.broadcast_arrays(ts, mu0)
        phi = jnp.exp(
            -0.25 * ts**2 * (self.beta_max - self.beta_min)
            - 0.5 * ts * self.beta_min
        )
        return phi * mu0

    def mean(self, ts: Array, x0=None, **kwargs) -> Array:
        shape = ts.shape
        ts = jnp.expand_dims(
            ts,
            axis=(-i for i in range(1, len(self.batch_shape) + len(self.event_shape) + 1)),
        )
        mu = self.marginal_mean(ts, x0=x0, **kwargs)
        return mu.reshape(shape + self.batch_shape + self.event_shape)

    def marginal_variance(self, ts: Array, x0=None, **kwargs) -> Array:
        if x0 is None:
            var0 = self.p0.variance
        else:
            var0 = jnp.zeros_like(x0)

        while ts.ndim < var0.ndim:
            ts = jnp.expand_dims(ts, axis=-1)

        ts, var0 = jnp.broadcast_arrays(ts, var0)
        phi = jnp.exp(
            -0.5 * ts**2 * (self.beta_max - self.beta_min) - ts * self.beta_min
        )
        return 1.0 + phi * (var0 - 1.0)

    def variance(self, ts: Array, x0=None, **kwargs) -> Array:
        shape = ts.shape
        ts = jnp.expand_dims(
            ts,
            axis=(-i for i in range(1, len(self.batch_shape) + len(self.event_shape) + 1)),
        )
        var = self.marginal_variance(ts, x0=x0, **kwargs)
        return var.reshape(shape + self.batch_shape + self.event_shape)

    def stddev(self, ts: Array, x0=None, **kwargs) -> Array:
        return jnp.sqrt(self.variance(ts, x0=x0, **kwargs))

    def marginal_stddev(self, ts: Array, x0=None, **kwargs) -> Array:
        return jnp.sqrt(self.marginal_variance(ts, x0=x0, **kwargs))

    def sample(self, key: Any, ts: Array, sample_shape=(), **kwargs) -> Array:
        key1, key2 = jax.random.split(key)
        x0 = self.p0.sample(key1, sample_shape)

        x0_flat = x0.reshape(-1, *self.event_shape)
        keys_flat = jax.random.split(key2, x0_flat.shape[0])

        vmap_dim = 0 if ts.ndim > 1 else None
        if ts.ndim > 1:
            ts = ts.reshape(-1, ts.shape[-1])

        _sdeint = partial(sdeint, **kwargs)
        __sdeint = jax.vmap(_sdeint, in_axes=(0, None, None, 0, vmap_dim))
        ys = __sdeint(keys_flat, self.drift, self.diffusion, x0_flat, ts)
        return ys.reshape(sample_shape + self.batch_shape + ts.shape + self.event_shape)

    def log_prob(self, x: Array, t: Array, x0=None) -> Array:
        mu = self.mean(t, x0=x0)
        std = self.stddev(t, x0=x0)
        return norm.logpdf(x, mu, std)

    def tree_flatten(self):
        return ((self.p0, self.beta_max, self.beta_min), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    


@register_pytree_node_class
class Normal():
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by

    Example::

        >>> key = random.PRNGKey(0)
        >>> m = Normal(jnp.array([0.0]), jnp.array([1.0]))
        >>> m.sample(key)  # normally distributed with loc=0 and scale=1
        array([-1.3348817], dtype=float32)

    Args:
        loc (float or ndarray): mean of the distribution (often referred to as mu)
        scale (float or ndarray): standard deviation of the distribution
            (often referred to as sigma)
    """

    arg_constraints = {"loc": real, "scale": strict_positive}
    support = real

    def __init__(self, loc: Array | float, scale: Array | float):
        loc = jnp.asarray(loc)
        scale = jnp.asarray(scale)
        self.loc, self.scale = jnp.broadcast_arrays(loc, scale)

        self._batch_shape = loc.shape
        self._event_shape = ()

    @property
    def batch_shape(self):
        return self._batch_shape
        
    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self) -> Array:
        return self.loc

    @property
    def mode(self) -> Array:
        return self.loc

    @property
    def median(self) -> Array:
        return self.loc

    @property
    def stddev(self) -> Array:
        return self.scale

    @property
    def variance(self) -> Array:
        return jnp.power(self.stddev, 2)

    @property
    def moment(self, n: int) -> Array:
        return self.scale * jnp.sqrt(2) * jnp.inverf(2 * n - 1)

    @property
    def fim(self) -> Array:
        mu = 1 / self.variance
        scale = 2 / self.variance
        mu_scale = jnp.stack([mu[..., None], scale[..., None]], axis=-1)
        return jnp.diag(mu_scale)

    def rsample(self, key, sample_shape: tuple = ()) -> Array:
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape)
        return self.loc + eps * self.scale
        
    def sample(self, key, sample_shape: tuple = ()) -> Array:
        return self.rsample(key, sample_shape)

    def log_prob(self, value) -> Array:
        return norm.logpdf(value, self.loc, self.scale)

    def cdf(self, value) -> Array:
        return norm.cdf(value, self.loc, self.scale)

    def icdf(self, value) -> Array:
        return norm.ppf(value, self.loc, self.scale)

    def entropy(self) -> Array:
        return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.scale)
        
    def tree_flatten(self):
        return (self.loc, self.scale), None
        
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        loc, scale = children
        return cls(loc, scale)

@register_pytree_node_class
class Independent:
    """
    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Args:
        base_dist: Base distribution object.
        reinterpreted_batch_ndims: The number of batch dimensions that should
            be considered as event dimensions.
    """

    def __init__(
        self,
        base_dist: Any,
        reinterpreted_batch_ndims: int,
    ):
        # Determine batch_shape and event_shape using the helper function
        batch_shape, event_shape, event_ndims, reinterpreted_batch_ndims = determine_shapes(
            base_dist, reinterpreted_batch_ndims
        )
        
        if not isinstance(base_dist, list) and not isinstance(base_dist, tuple):
            # Single distribution case
            self.base_dist = [base_dist]
        else:
            self.base_dist = base_dist

        self.event_ndims = event_ndims
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        
        for p in self.base_dist:
            p._batch_shape = batch_shape
            p._event_shape = event_shape

    @property
    def batch_shape(self) -> tuple:
        """Returns the shape over which parameters are batched."""
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """Returns the shape of a single sample (without batching)."""
        return self._event_shape

    @property
    def mean(self):
        return jnp.stack([b.mean for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def median(self):
        return jnp.stack([b.median for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def mode(self):
        # The mode does change and is not equal to the mode of the base distribution
        raise NotImplementedError()

    @property
    def variance(self):
        return jnp.stack([b.variance for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )
        
    @property
    def stddev(self):
        """Returns the standard deviation of the distribution."""
        return jnp.sqrt(self.variance)

    def rsample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        samples = jnp.stack(
            [p.rsample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            axis=-1,
        )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def sample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        if self.reinterpreted_batch_ndims > 0:
            samples = jnp.hstack(
                [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            )
        else:
            samples = jnp.stack(
                [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
                axis=-len(self.event_shape) - 1,
            )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        if len(self.base_dist) == 1:
            log_prob = self.base_dist[0].log_prob(value)
        else:
            if self.reinterpreted_batch_ndims > 0:
                split_value = jnp.split(value, self.event_ndims, axis=-1)[1:]
                log_prob = jnp.stack(
                    [
                        b.log_prob(v.reshape((-1,) + b.batch_shape + b.event_shape))
                        for b, v in zip(self.base_dist, split_value)
                    ], axis=-1
                )
                log_prob = jnp.reshape(
                    log_prob, value.shape[:-1] + self.batch_shape + (len(self.base_dist),)
                )
            else:
                split_value = jnp.split(
                    value, self.event_ndims, axis=-len(self.event_shape) - 1
                )[1:]
                log_prob = jnp.stack(
                    [b.log_prob(v) for b, v in zip(self.base_dist, split_value)],
                    axis=-len(self.event_shape) - 1,
                )
                log_prob = jnp.reshape(
                    log_prob,
                    value.shape[: -len(self.event_shape) - 1] + self.batch_shape,
                )

        # Sum the log probabilities along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                log_prob, axis=tuple(range(-self.reinterpreted_batch_ndims, 0))
            )
        else:
            return log_prob

    def prob(self, value):
        """Returns the probability density/mass function evaluated at value."""
        return jnp.exp(self.log_prob(value))

    def entropy(self):
        entropy = jnp.stack([b.entropy() for b in self.base_dist], axis=-1)

        # Sum the entropies along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                entropy, axis=tuple(range(-self.reinterpreted_batch_ndims, 0))
            )
        else:
            return entropy

    def __repr__(self) -> str:
        return f"Independent({self.base_dist}, reinterpreted_batch_ndims={self.reinterpreted_batch_ndims})"

    def tree_flatten(self):
        flat_components, tree_components = jax.tree_util.tree_flatten(self.base_dist)
        return (
            tuple(flat_components),
            [tree_components, self.reinterpreted_batch_ndims],
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tree_components, reinterpreted_batch_ndims = aux_data
        return cls(
            jax.tree_util.tree_unflatten(tree_components, children),
            reinterpreted_batch_ndims,
        )

@register_pytree_node_class
class TransformedDistribution:
    """
    A distribution transformed by a bijective transform.
    """
    
    arg_constraints: Dict[str, Constraint] = {}
    support: Constraint = Constraint()
    has_rsample = False
    multivariate = False
    
    def __init__(
        self,
        base_distribution: Any,
        transform: Callable,
        inverse_transform: Callable,
        log_det_jacobian: Callable,
    ):
        self.base_distribution = base_distribution
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.log_det_jacobian = log_det_jacobian
        
        self._batch_shape = base_distribution.batch_shape
        self._event_shape = base_distribution.event_shape
    
    @property
    def batch_shape(self) -> tuple:
        return self._batch_shape
    
    @property
    def event_shape(self) -> tuple:
        return self._event_shape
    
    @property
    def mean(self) -> Array:
        # This is generally not analytically available for transformed distributions
        raise NotImplementedError(f"{self.__class__} does not implement mean")
    
    @property
    def median(self) -> Array:
        # For bijective transforms, the median transforms
        try:
            return self.transform(self.base_distribution.median)
        except NotImplementedError:
            raise NotImplementedError(f"{self.__class__} does not implement median")
    
    @property
    def mode(self) -> Array:
        # Mode doesn't generally transform
        raise NotImplementedError(f"{self.__class__} does not implement mode")
    
    @property
    def variance(self) -> Array:
        # This is generally not analytically available for transformed distributions
        raise NotImplementedError(f"{self.__class__} does not implement variance")
    
    @property
    def stddev(self) -> Array:
        return jnp.sqrt(self.variance)
    
    def sample(self, key, sample_shape: tuple = tuple()) -> Array:
        base_samples = self.base_distribution.sample(key, sample_shape)
        return self.transform(base_samples)
    
    def rsample(self, key, sample_shape: tuple = tuple()) -> Array:
        if not self.base_distribution.has_rsample:
            raise NotImplementedError(f"{self.__class__} does not implement rsample")
        base_samples = self.base_distribution.rsample(key, sample_shape)
        return self.transform(base_samples)
    
    def log_prob(self, value: Array) -> Array:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.
        """
        x = self.inverse_transform(value)
        log_abs_det_jacobian = self.log_det_jacobian(value)
        return self.base_distribution.log_prob(x) - log_abs_det_jacobian
    
    def prob(self, value: Array) -> Array:
        return jnp.exp(self.log_prob(value))
    
    def cdf(self, value: Array) -> Array:
        x = self.inverse_transform(value)
        return self.base_distribution.cdf(x)
    
    def icdf(self, value: Array) -> Array:
        x = self.base_distribution.icdf(value)
        return self.transform(x)
    
    def entropy(self) -> Array:
        # Entropy of transformed distribution is generally not available analytically
        raise NotImplementedError(f"{self.__class__} does not implement entropy")
    
    def perplexity(self) -> Array:
        return jnp.exp(self.entropy())
    
    def __repr__(self) -> str:
        return f"TransformedDistribution(base_distribution={self.base_distribution})"
    
    # PyTree implementation
    def tree_flatten(self):
        return ((self.base_distribution, self.transform, self.inverse_transform, self.log_det_jacobian), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        base_distribution, transform, inverse_transform, log_det_jacobian = children
        return cls(base_distribution, transform, inverse_transform, log_det_jacobian)

def determine_shapes(
    base_dist: Any,
    reinterpreted_batch_ndims: int,
):
    if not isinstance(base_dist, list) and not isinstance(base_dist, tuple):
        # Single distribution case
        base_dist = [base_dist]

    # Extract batch shapes and event shapes from the list of base distributions
    batch_shapes = [getattr(b, 'batch_shape', ()) for b in base_dist]
    event_shapes = [getattr(b, 'event_shape', ()) for b in base_dist]
    
    assert all(reinterpreted_batch_ndims <= len(b) for b in batch_shapes) or all(reinterpreted_batch_ndims <= len(e) for e in event_shapes), "reinterpreted_batch_ndims must be greater than or equal to the batch shape of the base distribution."

    # Ensure that batch shapes are equal and calculate event_shape
    batch_shape, event_shape, event_ndims = calculate_shapes(
        batch_shapes, event_shapes, reinterpreted_batch_ndims
    )

    return tuple(batch_shape), tuple(event_shape), tuple(event_ndims), reinterpreted_batch_ndims

def calculate_shapes(batch_shapes, event_shapes, reinterpreted_batch_ndims):
    event_shape = list(event_shapes[0])
    if reinterpreted_batch_ndims > 0:
        new_event_shape = list(batch_shapes[0][-reinterpreted_batch_ndims:])
        if len(new_event_shape) > 0:
            for b in batch_shapes[1:]:
                if len(b) > 0:
                    new_event_shape[-1] += b[- 1]
                else:
                    new_event_shape[-1] += 1

        if len(event_shape) > 0:
            for e in event_shapes[1:]:
                if len(e) > 0:
                    event_shape[-1] += e[-1]
                else:
                    event_shape[-1] += 1

        batch_shape = tuple(batch_shapes[0][:-reinterpreted_batch_ndims])
        event_shape = tuple(new_event_shape) + tuple(event_shape)
        event_ndims = [0]
        for e in event_shapes:
            if len(e) == 0:
                event_ndims.append(event_ndims[-1] + 1)
            else:
                event_ndims.append(event_ndims[-1] + e[-1])
    else:
        new_batch_shape = list(batch_shapes[0])
        if len(new_batch_shape) > 0:
            for b in batch_shapes[1:]:
                if len(b) > 0:
                    new_batch_shape[-1] += b[-1]
                else:
                    new_batch_shape[-1] += 1
        else:
            new_batch_shape = (len(batch_shapes),)
        batch_shape = tuple(new_batch_shape)
        event_shape = tuple(event_shape)
        event_ndims = [0]
        for b in batch_shapes:
            if len(b) == 0:
                event_ndims.append(event_ndims[-1] + 1)
            else:
                event_ndims.append(event_ndims[-1] + b[-1])

    return batch_shape, event_shape, event_ndims

@register_pytree_node_class
class Uniform:
    """
    Uniform distribution without inheriting from Distribution base class.
    """
    
    arg_constraints: Dict[str, Constraint] = {"low": real, "high": real}
    support: Constraint = real
    has_rsample = False
    multivariate = False

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.low = jnp.where(low < high, low, high) - 1e-6
        self._batch_shape = jnp.shape(low)
        self._event_shape = ()

    @property
    def batch_shape(self) -> tuple:
        """Returns the shape over which parameters are batched."""
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """Returns the shape of a single sample (without batching)."""
        return self._event_shape

    @property
    def mean(self) -> Array:
        """Returns the mean of the distribution."""
        return (self.low + self.high) / 2.0

    @property
    def median(self) -> Array:
        """Returns the median of the distribution."""
        return (self.low + self.high) / 2.0

    @property
    def mode(self) -> Array:
        """Returns the mode of the distribution."""
        # Uniform distribution has no unique mode
        return (self.low + self.high) / 2.0  # Return midpoint as a convention

    @property
    def variance(self) -> Array:
        """Returns the variance of the distribution."""
        return (self.high - self.low) ** 2 / 12.0

    @property
    def stddev(self) -> Array:
        """Returns the standard deviation of the distribution."""
        return jnp.sqrt(self.variance)

    def sample(self, key: Any, sample_shape: tuple = ()) -> Array:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.uniform(key, shape, minval=self.low, maxval=self.high)

    def rsample(self, key: Any, sample_shape: tuple = ()) -> Array:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError(f"{self.__class__} does not implement rsample")

    def log_prob(self, value: Array) -> Array:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.
        """
        return jnp.log(
            jnp.where(
                (value >= self.low) & (value <= self.high),
                1.0 / (self.high - self.low),
                0.0,
            )
        )

    def prob(self, value: Array) -> Array:
        """
        Returns the probability density/mass function evaluated at
        `value`.
        """
        return jnp.exp(self.log_prob(value))

    def cdf(self, x: Array) -> Array:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.
        """
        return jnp.where(
            x < self.low,
            0.0,
            jnp.where(x > self.high, 1.0, (x - self.low) / (self.high - self.low)),
        )

    def icdf(self, q: Array) -> Array:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.
        """
        return self.low + q * (self.high - self.low)

    def entropy(self) -> Array:
        """
        Returns entropy of distribution, batched over batch_shape.
        """
        return jnp.log(self.high - self.low)

    def perplexity(self) -> Array:
        """
        Returns perplexity of distribution, batched over batch_shape.
        """
        return jnp.exp(self.entropy())

    def moment(self, n: int) -> Array:
        """
        Returns the nth non-central moment of the distribution.
        """
        if n == 0:
            return jnp.ones_like(self.low)
        elif n == 1:
            return self.mean
        else:
            # Formula for nth moment of uniform distribution
            return (self.high**(n+1) - self.low**(n+1)) / ((n+1) * (self.high - self.low))

    def __repr__(self) -> str:
        """String representation of the distribution."""
        args_string = f"low: {self.low if self.low.size == 1 else self.low.size}, high: {self.high if self.high.size == 1 else self.high.size}"
        return f"{self.__class__.__name__}({args_string})"

    # PyTree implementation
    def tree_flatten(self):
        return (
            (self.low, self.high),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        low, high = children
        return cls(low=low, high=high)


# Add this export at the end of the file to make these classes available for import
__all__ = [
    "Constraint", "Real", "Interval", "StrictPositive", 
    "VESDE", "VPSDE", "Empirical", "Normal", "Independent",
    "real", "strict_positive"
]

