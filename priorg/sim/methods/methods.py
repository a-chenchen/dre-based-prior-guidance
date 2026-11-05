import os
import sys
from copy import deepcopy

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from functools import partial
from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Now use imports relative to 'example'
from priorg.sim.distributions import Independent
from priorg.sim.distributions.continuous import Normal
from priorg.sim.distributions.discrete import Empirical
from priorg.sim.distributions.transformed_distribution import TransformedDistribution

# from priorg.sim.methods.guidance import repaint
from priorg.sim.methods.utils import VESDE, VPSDE, sdeint
from priorg.sim.nn.embedding_net import (
    XEmbeddingNet,
    XScoreDecoder,
    token_condition_mask,
)
from priorg.sim.nn.helpers import GaussianFourierEmbedding
from priorg.sim.nn.loss_fn import denoising_score_matching_loss
from priorg.sim.nn.tokenizer import ScalarTokenizer
from priorg.sim.nn.transformers import Transformer
from priorg.sim.utils.conditional_mask import get_condition_mask_fn


# Condition-mask functions under which the observation block is always fully
# conditioned. Only these are compatible with embed_x=True when the model does
# not decode scores for the observation entries (predict_x_score=False).
_X_CONDITIONED_MASK_FNS = frozenset({"posterior", "theta_random"})


class AllConditionalScoreModel:
    def __init__(
        self,
        params,
        model_fn,
        sde,
        sde_init_params,
        model_init_params,
        edge_mask_fn_params,
        z_score_params=None,
    ) -> None:
        self.params = params
        self.model_fn = model_fn
        self.sde = sde
        self.edge_mask_fn_params = edge_mask_fn_params
        self.z_score_params = z_score_params

        self.T_min = sde_init_params["T_min"]
        self.T_max = sde_init_params["T_max"]

        self.marginal_end_std = jnp.squeeze(
            sde.marginal_stddev(jnp.array([self.T_max]))
        )
        self.marginal_end_mean = jnp.squeeze(sde.marginal_mean(jnp.array([self.T_max])))

        # For sampling
        self.edge_mask = None
        self.edge_mask_fn = None
        self.meta_data = None
        self.score_fn = self.model_fn  # For score modifcations ...
        self.sampling_kwargs = {"num_steps": 500, "sampling_method": "sde"}

        # For pickle
        self.model_init_params = model_init_params
        self.sde_init_params = sde_init_params
        self.node_id = None
        self.condition_mask = None
        self.method = "score_transformer"
        self.backend = "jax"

    def _check_edge_mask(self, edge_mask, node_id, condition_mask, meta_data):
        if edge_mask is None:
            if self.edge_mask_fn is not None:
                edge_mask = self.edge_mask_fn(node_id, condition_mask, meta_data)
        return edge_mask

    def _check_for_meta_data(self, meta_data):
        if meta_data is None:
            if self.meta_data is not None:
                meta_data = self.meta_data
        return meta_data

    def _z_score_if_needed(self, x_o, node_id, condition_mask):
        if self.z_score_params is not None:
            z_score_fn = self.z_score_params["z_score_fn"]
            x_o = z_score_fn(x_o, node_id[condition_mask])
        return x_o

    def _un_z_score_if_needed(self, theta, node_id, condition_mask):
        if self.z_score_params is not None:
            un_z_score_fn = self.z_score_params["un_z_score_fn"]
            theta = un_z_score_fn(theta, node_id[~condition_mask])

        return theta

    def _check_x_o(self, x_o):
        """
        Check if x_o is provided, otherwise use the default value.

        Args:
            x_o: The value of x_o.

        Returns:
            The value of x_o.

        Raises:
            ValueError: If x_o is not provided and no default value is set.
        """
        if x_o is None:
            x_o = self.x_o
            if x_o is None:
                raise ValueError(
                    "Please provide x_o, either as argument or by calling set_default_x_o"
                )
        return x_o

    def _check_id_condition_mask(self, node_id, condition_mask):
        """
        Checks and retrieves the node ID and condition mask.

        If the node ID or condition mask is not provided, it retrieves the default values.

        Args:
            node_id: The node ID.
            condition_mask: The condition mask.

        Returns:
            node_id: The node ID.
            condition_mask: The condition mask.

        Raises:
            ValueError: If the node ID or condition mask is not provided.

        """
        if node_id is None:
            node_id = self.node_id
            if node_id is None:
                raise ValueError(
                    "Please provide node_id, either as argument or by calling set_default_node_id"
                )

        if condition_mask is None:
            condition_mask = self.condition_mask
            if condition_mask is not None:
                condition_mask = (
                    condition_mask  # .astype(jnp.bool_) TODO: Check if this is needed
                )
            else:
                raise ValueError(
                    "Please provide condition_mask, either as argument or by calling set_default_condition_mask"
                )
        return node_id, condition_mask

    def sample(
        self,
        num_samples,
        x_o=None,
        rng=None,
        node_id=None,
        condition_mask=None,
        **kwargs,
    ):
        """
        Samples from the model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The observed data.
            rng: The random number generator.
            node_id: The node ID.
            condition_mask: The condition mask.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated samples.

        """
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        x_o = self._check_x_o(x_o)
        return self._sample(
            num_samples,
            x_o=x_o,
            rng=rng,
            node_id=node_id,
            condition_mask=condition_mask,
            **kwargs,
        )

    def _sample(
        self,
        num_samples,
        x_o,
        num_steps=None,
        node_id=None,
        condition_mask=None,
        meta_data=None,
        edge_mask=None,
        rng=None,
        unique_nodes=False,  # If true, we assume all node_ids are unique and are JIT compatible
        return_conditioned_samples=False,  # If true, we preserves shapes and are JIT compatible
        with_bug=False,
        verbose=False,
        **kwargs,
    ):
        meta_data = self._check_for_meta_data(meta_data)
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask, meta_data)
        if x_o.shape[0] > 0:
            x_o = self._z_score_if_needed(x_o, node_id, condition_mask)

        sampling_kwargs = {**deepcopy(self.sampling_kwargs), **kwargs}
        if num_steps is None:
            num_steps = sampling_kwargs.pop("num_steps")
        else:
            if "num_steps" in sampling_kwargs:
                del sampling_kwargs["num_steps"]
        key1, key2 = jax.random.split(rng, 2)
        if not unique_nodes:
            unique_node_id = jnp.unique(node_id)
            if not with_bug:
                mean_end_per_node = jnp.array(
                    [
                        jnp.mean(self.marginal_end_mean[self.node_id == i])
                        for i in unique_node_id
                    ]
                )
                std_end_per_node = jnp.array(
                    [
                        jnp.mean(self.marginal_end_std[self.node_id == i])
                        for i in unique_node_id
                    ]
                )
            else:
                mean_end_per_node = jnp.array(
                    [
                        jnp.mean(self.marginal_end_std[self.node_id == i])
                        for i in unique_node_id
                    ]
                )
                std_end_per_node = jnp.array(
                    [
                        jnp.mean(self.marginal_end_std[self.node_id == i])
                        for i in unique_node_id
                    ]
                )
        else:
            mean_end_per_node = self.marginal_end_mean
            std_end_per_node = self.marginal_end_std

        x_T = (
            jax.random.normal(
                key1,
                (
                    num_samples,
                    node_id.shape[-1],
                ),
            )
            * std_end_per_node[node_id]
            + mean_end_per_node[node_id]
        )
        condition_mask = condition_mask.reshape(x_T.shape[-1])

        sampling_method = sampling_kwargs.pop("sampling_method")
        if verbose:
            print("Sampling method: ", sampling_method)
        if sampling_method == "sde":
            if unique_nodes:
                if x_o.shape[0] > 0:
                    indices = (
                        jnp.where(
                            condition_mask, jnp.arange(condition_mask.shape[0]), -1
                        )
                        % x_o.shape[0]
                    )
                    x_o_pad = x_o[indices]
                    x_T = x_T * (1 - condition_mask) + x_o_pad * condition_mask
            else:
                x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
            drift, diffusion = self._init_backward_sde(
                node_id, condition_mask, edge_mask, meta_data=meta_data
            )
            keys = jax.random.split(key2, (num_samples,))
            ys = jax.vmap(
                lambda *args: sdeint(*args, noise_type="diagonal", **sampling_kwargs),
                in_axes=(0, None, None, 0, None),
                out_axes=0,
            )(
                keys,
                drift,
                diffusion,
                x_T,
                jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
            )
            if not return_conditioned_samples:
                final_samples = ys[:, -1, ...][:, ~condition_mask]
            else:
                final_samples = ys[:, -1, ...]
            final_samples = final_samples.reshape((num_samples, -1))

        elif sampling_method == "ode":
            x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
            drift = self._init_backward_ode(
                node_id, condition_mask, edge_mask, meta_data=meta_data
            )
            ys = jax.vmap(
                lambda *args: _odeint(*args, **sampling_kwargs), in_axes=(None, 0, None)
            )(drift, x_T, jnp.linspace(0.0, self.T_max - self.T_min, num_steps))
            if not return_conditioned_samples:
                final_samples = ys[:, -1, ...][:, ~condition_mask]
            else:
                final_samples = ys[:, -1, ...]
            final_samples = final_samples.reshape((num_samples, -1))
        elif sampling_method == "repaint":
            resampling_steps = sampling_kwargs.pop("resampling_steps")

            @jax.vmap
            def sample_fn(key, x_T):
                return repaint(
                    self,
                    key,
                    condition_mask,
                    x_o,
                    x_T,
                    num_steps=num_steps,
                    node_id=node_id,
                    edge_mask=edge_mask,
                    meta_data=meta_data,
                    resampling_steps=resampling_steps,
                )

            keys = jax.random.split(key2, (num_samples,))
            final_samples = sample_fn(keys, x_T)
            final_samples = final_samples[:, ~condition_mask]
        elif sampling_method == "priorguide":
            raise NotImplementedError("priorguide sampling method not implemented yet")
        else:
            raise NotImplementedError(f"Unknown sampling method: {sampling_method}")

        final_samples = self._un_z_score_if_needed(
            final_samples, node_id, condition_mask
        )
        return final_samples

    def log_prob(self, theta, x_o=None, **kwargs):
        """
        Compute the log probability of theta given x_o.

        Args:
            theta: The value of theta.
            x_o: The value of x_o.
            **kwargs: Additional keyword arguments.

        Returns:
            The log probability.
        """
        x_o = self._check_x_o(x_o)
        return self._log_prob(theta, x_o=x_o, **kwargs)

    def _log_prob(
        self,
        val,
        x_o,
        num_steps=None,
        node_id=None,
        condition_mask=None,
        meta_data=None,
        edge_mask=None,
        **kwargs,
    ):
        # Add backward ode to compute log_prob
        assert condition_mask is not None, "Please provide a condition mask"
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        meta_data = self._check_for_meta_data(meta_data)
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask, meta_data)
        if x_o.shape[0] > 0:
            x_o = self._z_score_if_needed(x_o, node_id, condition_mask)

        sampling_kwargs = {**deepcopy(self.sampling_kwargs), **kwargs}
        if num_steps is None:
            num_steps = sampling_kwargs.pop("num_steps")
        else:
            if "num_steps" in sampling_kwargs:
                del sampling_kwargs["num_steps"]

        condition_mask = np.array(
            condition_mask
        )  # Make sure it's will be treated as static by jit

        q = self._init_cns(
            x_o,
            num_steps,
            node_id=node_id,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
            meta_data=meta_data,
            **kwargs,
        )
        return q.log_prob(val)

    def sample_batched(
        self,
        num_samples,
        x_o,
        node_id=None,
        condition_mask=None,
        edge_mask=None,
        meta_data=None,
        num_steps=None,
        rng=None,
        **kwargs,
    ):
        @jax.vmap
        def get_batched_samples(keys, x_os):
            samples = self.sample(
                num_samples,
                x_os,
                node_id=node_id,
                condition_mask=condition_mask,
                edge_mask=edge_mask,
                meta_data=meta_data,
                num_steps=num_steps,
                rng=keys,
                **kwargs,
            )
            return samples

        return get_batched_samples(jax.random.split(rng, x_o.shape[0]), x_o)

    def log_prob_batched(
        self,
        val,
        x_o,
        node_id=None,
        condition_mask=None,
        edge_mask=None,
        meta_data=None,
        num_steps=None,
        **kwargs,
    ):
        @jax.vmap
        def get_batched_log_probs(vals, x_os):
            log_probs = self.log_prob(
                vals,
                x_os,
                node_id=node_id,
                condition_mask=condition_mask,
                edge_mask=edge_mask,
                meta_data=meta_data,
                num_steps=num_steps,
                **kwargs,
            )
            return log_probs

        return get_batched_log_probs(val, x_o)

    def set_default_edge_mask_fn(self, edge_mask_fn):
        self.edge_mask_fn = edge_mask_fn

    def set_default_score_fn(self, score_fn):
        self.score_fn = score_fn

    def set_default_sampling_kwargs(self, **kwargs):
        self.sampling_kwargs = kwargs

    def set_default_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_default_condition_mask(self, condition_mask):
        """
        Sets the default condition mask.

        Args:
            condition_mask: The default condition mask.

        """
        self.condition_mask = condition_mask

    def set_default_node_id(self, node_id):
        """
        Sets the default node ID.

        Args:
            node_id: The default node ID.

        """
        self.node_id = node_id

    def _init_score(self, node_id, condition_mask, edge_mask, meta_data):

        def score_fn(t, x):
            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                meta_data=(
                    meta_data.reshape(-1, meta_data.shape[-1], 1)
                    if meta_data is not None
                    else None
                ),
                edge_mask=edge_mask,
            ).reshape(x.shape)

            return score

        return score_fn

    def _init_backward_sde(
        self, node_id=None, condition_mask=None, edge_mask=None, meta_data=None
    ):
        # Thats in general pretty shitty as it trickers recompilation every time we call sample
        def drift_backward(t, x):
            t = self.T_max - t

            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                meta_data=(
                    meta_data.reshape(-1, meta_data.shape[-1], 1)
                    if meta_data is not None
                    else None
                ),
                edge_mask=edge_mask,
            ).reshape(x.shape)
            drift = self.sde.drift(t, x) - self.sde.diffusion(t, x) ** 2 * score
            return -drift.reshape(x.shape) * (1 - condition_mask.reshape(x.shape))

        def diffusion_backward(t, x):
            t = self.T_max - t
            return self.sde.diffusion(t, x).reshape(x.shape) * (
                1 - condition_mask.reshape(x.shape)
            )

        return drift_backward, diffusion_backward

    def _init_backward_ode(
        self, node_id=None, condition_mask=None, edge_mask=None, meta_data=None
    ):
        def drift_backward(t, x):
            t = self.T_max - t
            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                edge_mask=edge_mask,
                meta_data=(
                    meta_data.reshape(-1, meta_data.shape[-1], 1)
                    if meta_data is not None
                    else None
                ),
            ).reshape(x.shape)
            dx = self.sde.drift(t, x) - 0.5 * self.sde.diffusion(t, x) ** 2 * score
            return -dx.reshape(x.shape) * (1 - condition_mask.reshape(x.shape))

        return drift_backward

    def _init_cns(
        self,
        x_o,
        num_steps,
        node_id=None,
        condition_mask=None,
        edge_mask=None,
        meta_data=None,
        **kwargs,
    ):

        drift = self._init_backward_ode(node_id, condition_mask, edge_mask, meta_data)

        def drift_cond(t, x):
            xs = jnp.zeros((len(node_id),))
            xs = xs.at[condition_mask].set(x_o.reshape(-1))
            xs = xs.at[~condition_mask].set(x.reshape(-1))
            f = drift(t, xs)
            f = f[~condition_mask]
            return f

        def f(x_T):
            y = odeint(
                drift_cond,
                x_T,
                jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
                **kwargs,
            )[-1]
            return y

        if node_id is None:
            q0 = Independent(
                Normal(
                    self.marginal_end_mean[~condition_mask],
                    self.marginal_end_std[~condition_mask],
                ),
                1,
            )
        else:
            q0 = Independent(
                Normal(
                    self.marginal_end_mean[node_id][~condition_mask],
                    self.marginal_end_std[node_id][~condition_mask],
                ),
                1,
            )
        q = TransformedDistribution(q0, f)
        return q

    def map(
        self,
        x_o,
        node_id=None,
        condition_mask=None,
        edge_mask=None,
        meta_data=None,
        num_init=1000,
        num_sampling_steps=None,
        init_learning_rate=1e-3,
        eps=0.1,
        rng=None,
        **kwargs,
    ):
        assert rng is not None, "Please provide a rng key"

        meta_data = self._check_for_meta_data(meta_data)
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask, meta_data)
        x_o = self._z_score_if_needed(x_o, node_id, condition_mask)

        if num_sampling_steps is None:
            num_sampling_steps = self.sampling_kwargs["num_steps"]

        score_fn = self._init_score(
            self.node_id, self.condition_mask, self.edge_mask, self.meta_data
        )

        latents_candidate = self.sample(
            num_init,
            x_o,
            node_id=node_id,
            condition_mask=condition_mask,
            meta_data=meta_data,
            edge_mask=edge_mask,
            rng=rng,
            num_steps=num_sampling_steps,
            **kwargs,
        )

        x_o_repeated = jnp.repeat(x_o.reshape((-1, x_o.shape[-1])), num_init, axis=0)
        samples = jnp.concatenate([latents_candidate, x_o_repeated], axis=-1)

        optimizer = optax.adam(init_learning_rate)
        opt_state = optimizer.init(samples)

        def update(opt_state, xs):
            grads = -score_fn(jnp.ones((1,)) * self.T_min, xs) * ~condition_mask
            updates, opt_state = optimizer.update(grads, opt_state)
            new_xs = optax.apply_updates(xs, updates)
            return new_xs, opt_state, grads

        def body_fn(state):
            xs, opt_state, grads = state
            new_xs, opt_state, grads = update(opt_state, xs)
            return new_xs, opt_state, grads

        def cond_fn(state):
            _, _, grads = state
            return jnp.quantile(jnp.linalg.norm(grads, axis=-1), 0.2) > eps

        state = update(opt_state, samples)
        xs, opt_state, grads = jax.lax.while_loop(cond_fn, body_fn, state)

        latents = xs[:, ~condition_mask]
        log_probs = self.log_prob(
            latents,
            x_o,
            node_id=node_id,
            condition_mask=condition_mask,
            meta_data=meta_data,
            edge_mask=edge_mask,
            num_steps=num_sampling_steps,
            **kwargs,
        )

        idx = log_probs.argmax()
        best_latent = latents[idx]

        return best_latent, latents, log_probs

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["model_fn"] = None
        state["sde"] = None
        state["score_fn"] = None
        state["edge_mask_fn"] = None
        if self.z_score_params is not None:
            state["z_score_params"]["z_score_fn"] = None
            state["z_score_params"]["un_z_score_fn"] = None
        return state

    def __setstate__(self, state):
        # from scoresbibm.methods.neural_nets import scalar_transformer_model
        # from scoresbibm.methods.score_transformer import get_z_score_fn
        # from scoresbibm.methods.sde import init_sde_related
        # from scoresbibm.utils.edge_masks import get_edge_mask_fn

        with jax.default_device(jax.devices("cpu")[0]):
            self.__dict__.update(state)
            self.sde, self.T_min, self.T_max, _, output_scale_fn = init_sde_related(
                **self.sde_init_params
            )
            _, self.model_fn = scalar_transformer_model(
                output_scale_fn=output_scale_fn, **self.model_init_params
            )
            self.score_fn = self.model_fn
            # The task is not needed to restore the model (edge masks are unused),
            # so it is not rebuilt here; this also lets models of tasks outside
            # get_task (e.g. the light-curve task) be unpickled.
            self.edge_mask_fn = None
            # get_edge_mask_fn(
            # self.edge_mask_fn_params["name"], task
            # )

            if not hasattr(self, "z_score_params"):
                self.z_score_params = None

            if self.z_score_params is not None:
                z_score_fn, un_z_score_fn = get_z_score_fn(
                    self.z_score_params["mean_per_node_id"],
                    self.z_score_params["std_per_node_id"],
                )
                self.z_score_params["z_score_fn"] = z_score_fn
                self.z_score_params["un_z_score_fn"] = un_z_score_fn


def scalar_transformer_model(
    num_nodes: int,
    token_dim: int = 40,
    condition_token_dim: int = 10,
    condition_token_init_scale: int = 0.01,
    condition_token_init_mean: int = 0.0,
    condition_mode: str = "concat",
    time_embedding_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 6,
    attn_size: int = 5,
    widening_factor: int = 4,
    num_hidden_layers: int = 1,
    act=jax.nn.gelu,
    skip_connection_attn: bool = True,
    skip_connection_mlp: bool = True,
    layer_norm: bool = True,
    output_scale_fn=None,
    base_mask=None,
    embed_x: bool = False,
    theta_dim: int = None,
    x_dim: int = None,
    num_x_tokens: int = 1,
    x_embed_hidden: Sequence[int] = (256, 128),
    x_embed_chunked: bool = False,
    x_embed_layer_norm: bool = True,
    predict_x_score: bool = False,
    x_decode_hidden: Sequence[int] = (128, 256),
    **kwargs,
):
    """Build the Simformer score model.

    By default every scalar of ``(theta, x)`` becomes its own token.  With
    ``embed_x=True`` the observation is instead condensed into ``num_x_tokens``
    tokens by a jointly-trained embedding network, following Gloeckler et al.
    (2024) Sec. 3.1 / App. A3.2.  In that mode the sequence length drops from
    ``theta_dim + x_dim`` to ``theta_dim + num_x_tokens``, but direct control
    over the condition state of individual observation scalars is lost -- the
    model must be trained on condition masks under which each observation block
    is either fully observed or fully latent.

    Args:
        embed_x: Enable the observation embedding network.
        theta_dim, x_dim: Required when ``embed_x=True`` (the model must know
            where the parameter block ends and the observation block begins).
        num_x_tokens: Number of tokens the observation is condensed into.
        x_embed_hidden: Hidden widths of the embedding MLP.
        x_embed_chunked: Split the observation into ``num_x_tokens`` contiguous
            blocks with one MLP each (keeps block-level conditioning) instead of
            one MLP over the whole vector.
        x_embed_layer_norm: LayerNorm the emitted observation tokens.
        predict_x_score: Emit scores for the observation entries by decoding the
            observation tokens.  Only needed when observation tokens can be
            latent; in the targeted-inference setting (observation always
            conditioned) leave this False and the model emits zeros there, which
            the loss masks out and the sampler freezes.
        x_decode_hidden: Hidden widths of the decoder MLP.
    """
    if output_scale_fn is None:
        output_scale_fn = lambda t, x: x

    if embed_x:
        if theta_dim is None or x_dim is None:
            raise ValueError(
                "embed_x=True requires theta_dim and x_dim to be provided."
            )
        if theta_dim + x_dim != num_nodes:
            raise ValueError(
                f"theta_dim + x_dim ({theta_dim} + {x_dim}) must equal "
                f"num_nodes ({num_nodes})."
            )

    if condition_mode == "concat":
        condition_token_dim = condition_token_dim
    elif condition_mode == "add":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = token_dim
    elif condition_mode == "none":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = 0

    def model(t, data, data_id, condition_mask, meta_data=None, edge_mask=base_mask):
        _, current_nodes, _ = data.shape  # (batch, nodes, 1)
        data_id = data_id.reshape(-1, current_nodes)
        condition_mask = condition_mask.reshape(-1, current_nodes)

        if embed_x:
            if meta_data is not None:
                raise NotImplementedError(
                    "meta_data is not supported together with embed_x=True."
                )
            # Parameters keep one token per scalar; the observation is condensed
            # into num_x_tokens tokens by the jointly-trained embedding network.
            tokenizer = ScalarTokenizer(token_dim, theta_dim + num_x_tokens)
            theta_tokens = tokenizer(
                data_id[:, :theta_dim], data[:, :theta_dim, :], None
            )

            x_tokens = XEmbeddingNet(
                token_dim=token_dim,
                x_dim=x_dim,
                num_x_tokens=num_x_tokens,
                hidden_sizes=x_embed_hidden,
                chunked=x_embed_chunked,
                act=act,
                layer_norm=x_embed_layer_norm,
            )(data[:, theta_dim:, 0])

            # Identifier for the observation tokens, mirroring the role of the
            # tokenizer's node-id embedding for the parameter tokens.
            x_token_id = hk.get_parameter(
                "x_token_id_embedding",
                shape=[1, num_x_tokens, token_dim],
                init=hk.initializers.TruncatedNormal(stddev=0.02),
            )
            x_tokens = x_tokens + x_token_id

            tokens = jnp.concatenate([theta_tokens, x_tokens], axis=-2)
            # An observation token is conditioned only if every scalar it
            # encodes is conditioned.
            condition_mask = token_condition_mask(
                condition_mask, theta_dim, x_dim, num_x_tokens, x_embed_chunked
            )
            current_nodes = theta_dim + num_x_tokens
            time = GaussianFourierEmbedding(time_embedding_dim)(t[..., None])
        else:
            # NOTE: module call order here is load-bearing -- Haiku draws init
            # keys in call order, so reordering these two changes the baseline
            # model's initial weights for a given seed.
            tokenizer = ScalarTokenizer(token_dim, num_nodes)
            time_embeder = GaussianFourierEmbedding(time_embedding_dim)

            # Embedding
            tokens = tokenizer(data_id, data, meta_data)
            time = time_embeder(t[..., None])

        # Conditioning
        if condition_mode != "none":
            condition_token = hk.get_parameter(
                "condition_token",
                shape=[1, 1, condition_token_dim],
                init=hk.initializers.RandomNormal(
                    condition_token_init_scale, condition_token_init_mean
                ),
            )
            condition_mask = condition_mask.reshape(-1, current_nodes, 1)
            condition_token = condition_mask * condition_token
            if condition_mode == "add":
                tokens = tokens + condition_token
            elif condition_mode == "concat":
                condition_token = jnp.broadcast_to(
                    condition_token, tokens.shape[:-1] + (condition_token_dim,)
                )
                tokens = jnp.concatenate([tokens, condition_token], -1)

        # Forward pass

        model = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            num_hidden_layers=num_hidden_layers,
            act=act,
            skip_connection_attn=skip_connection_attn,
            skip_connection_mlp=skip_connection_mlp,
            #           layer_norm=layer_norm,
        )

        h = model(tokens, context=time, mask=edge_mask)

        if embed_x:
            out_theta = hk.Linear(1)(h[:, :theta_dim, :])
            if predict_x_score:
                out_x = XScoreDecoder(
                    x_dim=x_dim,
                    num_x_tokens=num_x_tokens,
                    hidden_sizes=x_decode_hidden,
                    chunked=x_embed_chunked,
                    act=act,
                )(h[:, theta_dim:, :])
            else:
                # Observation tokens are always conditioned in this mode: the
                # loss masks these entries out and the sampler freezes them
                # (drift and diffusion are multiplied by 1 - condition_mask),
                # so the value here is never used.
                out_x = jnp.zeros(
                    (out_theta.shape[0], x_dim, 1), dtype=out_theta.dtype
                )
            out = jnp.concatenate([out_theta, out_x], axis=-2)
        else:
            out = hk.Linear(1)(h)

        out = output_scale_fn(t, out)
        return out

    init_fn, model_fn = hk.without_apply_rng(hk.transform(model))
    return init_fn, model_fn


def run_train_transformer_model(
    key,
    params,
    opt_state,
    data,
    node_id,
    meta_data,
    total_number_steps,
    batch_size,
    update,
    batch_sampler,
    loss_fn,
    print_every=100,
    val_every=100,
    validation_fraction=0.05,
    val_repeat=2,
    val_error_ratio=1.1,
    stop_early_count=5,
):
    """Trains a transformer model with early stopping based on validation performance.

    This function handles distributed training across multiple devices using JAX's pmap.
    It splits data into training and validation sets, monitors validation performance,
    and implements early stopping when validation loss deteriorates.

    Args:
        key: JAX random number generator key.
        params: Initial model parameters.
        opt_state: Initial optimizer state.
        data: Training data.
        node_id: Node identifiers for the data.
        meta_data: Additional metadata for the data (optional).
        total_number_steps: Total number of training steps to perform.
        batch_size: Batch size for training (will be divided across available devices).
        update: pmapped update function that performs parameter updates.
        batch_sampler: Function to sample batches from the training data.
        loss_fn: Loss function for validation.
        print_every: How often to print the training loss (in steps).
        val_every: How often to compute validation loss (in steps).
        validation_fraction: Fraction of data to use for validation.
        val_repeat: Number of times to repeat validation data for Monte Carlo estimation.
        val_error_ratio: Ratio of validation to training loss that triggers early stopping counter.
        stop_early_count: Number of consecutive triggers before stopping early.

    Returns:
        tuple: (params, opt_state) containing either:
            - The parameters with lowest validation loss and corresponding optimizer state if early
              stopping occurred.
            - The final parameters and optimizer state after training for total_number_steps.
    """
    # Set up stuff for multi-device training
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices

    # Validation loss
    data_val, data_train = jnp.split(
        data, [max(int(validation_fraction * data.shape[0]), 0)], axis=0
    )
    data_val = jnp.repeat(
        data_val, val_repeat, axis=0
    )  # Multiple Monte Carlo samples for validation loss
    if meta_data is not None and meta_data.ndim > 2:
        meta_data_val, meta_data_train = jnp.split(
            meta_data, [max(int(validation_fraction * data.shape[0]), 1)], axis=0
        )
        meta_data_val = jnp.repeat(
            meta_data_val, val_repeat, axis=0
        )  # Multiple Monte Carlo samples for validation loss
    else:
        meta_data_val = meta_data
        meta_data_train = meta_data

    sampler = partial(
        batch_sampler,
        data=data_train,
        node_id=node_id,
        meta_data=meta_data_train,
        num_devices=num_devices,
    )

    # Replicated for multiple devices
    replicated_params = jax.tree.map(lambda x: jnp.array([x] * num_devices), params)
    replicated_opt_state = jax.tree.map(
        lambda x: jnp.array([x] * num_devices), opt_state
    )

    early_stopping_counter = 0

    l_val = None
    l_train = None
    min_l_val = 1e10
    early_stopping_params = None

    for j in range(total_number_steps):
        key, key_batch, key_update, key_val = jax.random.split(key, 4)
        data_batch, node_id_batch, meta_data_batch = sampler(
            key_batch, batch_size_per_device
        )
        loss, replicated_params, replicated_opt_state = update(
            replicated_params,
            replicated_opt_state,
            jax.random.split(key_update, (num_devices,)),
            data_batch,
            node_id_batch,
            meta_data_batch,
        )
        # Train loss
        if j == 0:
            l_train = loss[0]
        else:
            l_train = 0.9 * l_train + 0.1 * loss[0]

        # Validation loss
        if validation_fraction > 0 and ((j % val_every) == 0) and j > 50:
            l_val = loss_fn(
                jax.tree.map(lambda x: x[0], replicated_params),
                key_val,
                data_val,
                node_id,
                meta_data_val,
            )

            if l_val / l_train > val_error_ratio:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if l_val < min_l_val:
                min_l_val = l_val
                early_stopping_params = jax.tree.map(lambda x: x[0], replicated_params)

        if early_stopping_counter > stop_early_count:
            return early_stopping_params, jax.tree.map(
                lambda x: x[0], replicated_opt_state
            )

        # Print
        if (j % print_every) == 0:
            print("Train loss: ", l_train, flush=True)
            if l_val is not None:
                print("Validation loss: ", l_val, early_stopping_counter, flush=True)

    params = jax.tree.map(lambda x: x[0], replicated_params)
    opt_state = jax.tree.map(lambda x: x[0], replicated_opt_state)

    del replicated_opt_state
    del replicated_params

    return params, opt_state


def get_method(name: str):
    """Get a method"""
    if "score_transformer" in name:
        return run_score_transformer
    else:
        raise NotImplementedError()


def run_score_transformer(task, data, method_cfg, rng=None):
    """
    Trains a score-based transformer model for diffusion-based simulation-based inference.

    This function initializes, trains, and returns a score-based transformer model that can sample
    from various conditional distributions. The model is trained on joint and conditional distributions
    using denoising score matching with a specified SDE (stochastic differential equation).

    Args:
        task: The simulation task object that defines the problem structure and data generation.
        data: Dictionary containing simulation data with keys 'theta' (parameters) and 'x' (observations),
              and optionally 'metadata'.
        method_cfg: Configuration object containing model architecture, SDE, and training parameters.
                   Should have nested attributes: device, sde, model, train, and posterior.
        rng: JAX random number generator key. If None, a new one will be created.

    Returns:
        AllConditionalScoreModel: A trained score-based diffusion model that can sample from
                                 conditional distributions with flexible conditioning patterns.
                                 The model is configured by default for posterior sampling.
    """
    device = method_cfg.device
    sde_params = dict(method_cfg.sde)
    model_params = dict(method_cfg.model)
    train_params = dict(method_cfg.train)

    # Data
    thetas, xs = data["theta"], data["x"]
    metadata = data.get("metadata", None)
    data = jnp.hstack([thetas, xs])
    node_id = task.get_node_id()
    theta_dim = task.get_theta_dim()
    x_dim = task.get_x_dim()
    data = data[..., None]
    if metadata is not None:
        metadata = metadata[..., None]

    # Z score
    if method_cfg.train.z_score_data:
        mean_per_node_id, std_per_node_id = mean_std_per_node_id(data, node_id)
        z_score_fn, un_z_score_fn = get_z_score_fn(mean_per_node_id, std_per_node_id)
        data = z_score_fn(data, node_id)

    # Initialize stuff
    sde, T_min, T_max, _weight_fn, output_scale_fn = init_sde_related(
        data, name=sde_params.pop("name"), **sde_params
    )
    weight_fn = lambda t: _weight_fn(t).reshape(-1, 1, 1)
    if not model_params.pop("use_output_scale_fn", True):
        output_scale_fn = None
    # The model needs the theta/x split to know where the observation block
    # starts when the observation embedding network is enabled.
    model_params.setdefault("theta_dim", theta_dim)
    model_params.setdefault("x_dim", x_dim)

    if model_params.get("embed_x", False) and not model_params.get(
        "predict_x_score", False
    ):
        _mask_name = str(train_params["condition_mask_fn"]["name"]).lower()
        if _mask_name not in _X_CONDITIONED_MASK_FNS:
            raise ValueError(
                f"embed_x=True with predict_x_score=False requires a condition "
                f"mask function under which the observation is always fully "
                f"conditioned, but got '{_mask_name}'. Use one of "
                f"{sorted(_X_CONDITIONED_MASK_FNS)}, or set predict_x_score=True "
                f"to also decode scores for the observation entries."
            )

    init_fn, model_fn = scalar_transformer_model(
        theta_dim + x_dim, output_scale_fn=output_scale_fn, **model_params
    )

    rng, rng_init = jax.random.split(rng)
    params = init_fn(
        rng_init,
        jnp.ones((10,)),
        data[:10],
        node_id,
        jnp.zeros_like(data[:10]),
        meta_data=metadata,
    )

    # Training params
    total_number_steps = int(
        max(
            min(
                data.shape[0] * train_params["total_number_steps_scaling"],
                train_params["max_number_steps"],
            ),
            train_params["min_number_steps"],
        )
    )
    batch_size = train_params["training_batch_size"]

    print_every = total_number_steps // 10
    val_every = total_number_steps // train_params["val_every"]
    learning_rate = train_params["learning_rate"]
    schedule = optax.linear_schedule(
        learning_rate,
        train_params["min_learning_rate"],
        total_number_steps // 2,
        total_number_steps // 2,
    )
    optimizer = optax.chain(
        optax.adaptive_grad_clip(train_params["clip_max_norm"]), optax.adam(schedule)
    )
    opt_state = optimizer.init(params)

    # Extract condition mask parameters, with debugging
    print("Original params:", dict(train_params["condition_mask_fn"]), flush=True)

    # Explicitly filter out any None/null values
    condition_mask_params = {}
    for k, v in dict(train_params["condition_mask_fn"]).items():
        if k != "name" and not k.startswith("_"):
            if v is not None and v != "null" and str(v).lower() != "null":
                condition_mask_params[k] = v

    print("Filtered params:", condition_mask_params, flush=True)
    edge_mask_params = dict(train_params["edge_mask_fn"])

    # Get condition and edge mask functions
    condition_mask_fn = get_condition_mask_fn(
        train_params["condition_mask_fn"]["name"], **condition_mask_params
    )

    edge_mask_fn = lambda node_id, condition_mask, *args, **kwargs: None

    # Training loop
    @jax.jit
    def loss_fn(params, key, data, node_id, meta_data):
        key_times, key_loss, key_condition = jax.random.split(key, 3)
        times = jax.random.uniform(
            key_times, (data.shape[0],), minval=T_min, maxval=T_max
        )

        # Structured conditioning
        condition_mask = condition_mask_fn(
            key_condition, data.shape[0], theta_dim, x_dim
        )
        if meta_data is None:
            edge_mask = jax.vmap(edge_mask_fn, in_axes=(None, 0))(
                node_id, condition_mask
            )
        else:
            edge_mask = jax.vmap(edge_mask_fn, in_axes=(None, 0, 0))(
                node_id, condition_mask, meta_data
            )

        loss = denoising_score_matching_loss(
            params,
            key_loss,
            times,
            data,
            loss_mask=condition_mask,
            model_fn=model_fn,
            mean_fn=sde.marginal_mean,
            std_fn=sde.marginal_stddev,
            weight_fn=weight_fn,
            rebalance_loss=train_params["rebalance_loss"],
            data_id=node_id,
            condition_mask=condition_mask,
            meta_data=meta_data,
            edge_mask=edge_mask,
        )
        return loss

    @partial(jax.pmap, axis_name="num_devices")
    def update(params, opt_state, key, data, node_id, meta_data):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, data, node_id, meta_data)

        loss = jax.lax.pmean(loss, axis_name="num_devices")
        grads = jax.lax.pmean(grads, axis_name="num_devices")

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    rng, rng_train = jax.random.split(rng)
    batch_sampler = task.get_batch_sampler()
    params, opt_state = run_train_transformer_model(
        rng_train,
        params,
        opt_state,
        data,
        node_id,
        metadata,
        total_number_steps,
        batch_size,
        update,
        batch_sampler,
        loss_fn,
        print_every=print_every,
        val_every=val_every,
        validation_fraction=train_params["validation_fraction"],
        val_repeat=train_params["val_repeat"],
        stop_early_count=train_params["stop_early_count"],
        val_error_ratio=train_params.get("val_error_ratio", 1.1),
    )

    sde_init_params = {
        "data": jax.device_put(data, jax.devices("cpu")[0]),
        **dict(method_cfg.sde),
    }
    model_init_params = {
        "num_nodes": theta_dim + x_dim,
        "theta_dim": theta_dim,
        "x_dim": x_dim,
        **dict(method_cfg.model),
    }
    edge_mask_params["task"] = task.name
    if method_cfg.train.z_score_data:
        z_score_params = {
            "mean_per_node_id": mean_per_node_id,
            "std_per_node_id": std_per_node_id,
            "z_score_fn": z_score_fn,
            "un_z_score_fn": un_z_score_fn,
        }
    else:
        z_score_params = None
    model = AllConditionalScoreModel(
        params,
        model_fn,
        sde,
        sde_init_params=sde_init_params,
        model_init_params=model_init_params,
        edge_mask_fn_params=edge_mask_params,
        z_score_params=z_score_params,
    )
    # Posterior as default
    default_conditon_mask = jnp.array([0] * theta_dim + [1] * x_dim, dtype=jnp.bool_)
    model.set_default_condition_mask(default_conditon_mask)
    model.set_default_node_id(node_id)
    model.set_default_edge_mask_fn(edge_mask_fn)

    model.set_default_sampling_kwargs(**method_cfg.posterior)
    return model


def init_sde_related(data, name="vpsde", **kwargs):
    """Initialize the sde and related functions."""
    if name.lower() == "vpsde":
        p0 = Independent(Empirical(data), 1)
        beta_min = kwargs.get("beta_min", 0.1)
        beta_max = kwargs.get("beta_max", 10.0)
        sde = VPSDE(p0, beta_min=beta_min, beta_max=beta_max)
        T_max = kwargs.get("T_max", 1.0)
        T_min = kwargs.get("T_min", 1e-5)
        scale_min = kwargs.get("scale_min", 1e-3)

        def weight_fn(t):
            t = t.reshape(-1, 1)
            return jnp.clip(
                1.0 - jnp.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t),
                a_min=1e-4,
            )

        def output_scale_fn(t, x):
            scale = jnp.clip(
                jnp.sqrt(
                    jnp.sum(
                        sde.marginal_variance(t[..., None], x0=jnp.ones_like(x)), -1
                    )
                ),
                scale_min,
            )
            return 1 / scale[..., None] * x
    elif name.lower() == "vesde":
        p0 = Independent(Empirical(data), 1)
        sigma_min = kwargs.get("sigma_min", 0.01)
        sigma_max = kwargs.get("sigma_max", 10.0)
        sde = VESDE(p0, sigma_min=sigma_min, sigma_max=sigma_max)
        T_max = kwargs.get("T_max", 1.0)
        T_min = kwargs.get("T_min", 1e-5)
        scale_min = kwargs.get("scale_min", 1e-3)

        # Train weight function
        def weight_fn(t):
            t = t.reshape(-1, 1)
            return sde.diffusion(t, jnp.ones((1,))) ** 2

        # Model output scale function
        def output_scale_fn(t, x):
            scale = jnp.clip(
                jnp.sqrt(
                    jnp.sum(
                        sde.marginal_variance(t[..., None], x0=jnp.ones_like(x)), -1
                    )
                ),
                scale_min,
            )
            return 1 / scale[..., None] * x

    else:
        raise NotImplementedError()

    return sde, T_min, T_max, weight_fn, output_scale_fn


def mean_std_per_node_id(data, node_ids):
    node_ids = node_ids.reshape(-1)
    mean = []
    std = []
    for i in range(node_ids.max() + 1):
        index = jnp.where(node_ids == i)
        mean.append(jnp.mean(data[:, index]))
        std.append(jnp.std(data[:, index]))
    mean, std = jnp.stack(mean), jnp.clip(jnp.stack(std), a_min=1e-2, a_max=None)
    mean = mean.reshape(-1, 1)
    std = std.reshape(-1, 1)
    return mean, std


def compute_guidance_z_score_params(task, model, n_samples, key):
    """Compute z-score normalisation parameters from task (unshifted) samples.

    Use this when the score model was trained *without* `z_score_data=True` but
    you still want to normalise the training data for the guidance models
    (log-ratio model or score field model).  Pass the returned dict as the
    ``z_score_params`` argument to ``train_log_ratio_model`` or
    ``train_score_field_model``.

    **Important:** the same ``z_score_params`` must also be passed to the
    corresponding sampler (``prior_guide_with_log_ratio_model`` /
    ``prior_guide_with_score_field``) so that the guidance model receives
    normalised inputs at inference time.  The easiest way to do this is to
    temporarily set ``model.z_score_params = z_score_params`` before calling
    the sampler.

    Args:
        task:      Prior-shift task with a ``sample_unshifted`` method.
        model:     ``AllConditionalScoreModel`` (provides ``node_id``).
        n_samples: Number of samples used to estimate mean/std.
        key:       JAX PRNG key.

    Returns:
        dict with keys ``mean_per_node_id``, ``std_per_node_id``,
        ``z_score_fn``, ``un_z_score_fn``.
    """
    import jax

    thetas, x_obs = task.sample_unshifted(n_samples, key=key)
    x_full = jnp.concatenate([jnp.asarray(thetas), jnp.asarray(x_obs)], axis=-1)
    x_full = x_full[..., None]  # (n_samples, D, 1)  — expected by mean_std_per_node_id
    mean_per_node_id, std_per_node_id = mean_std_per_node_id(x_full, model.node_id)
    z_score_fn, un_z_score_fn = get_z_score_fn(mean_per_node_id, std_per_node_id)
    return {
        "mean_per_node_id": mean_per_node_id,
        "std_per_node_id": std_per_node_id,
        "z_score_fn": z_score_fn,
        "un_z_score_fn": un_z_score_fn,
    }


def get_z_score_fn(data_mean_per_node_id, data_std_per_node_id):
    def z_score(data, node_id):
        shape = data.shape
        data = data.reshape(-1, len(node_id), 1)
        return (
            (data - data_mean_per_node_id[node_id]) / data_std_per_node_id[node_id]
        ).reshape(shape)

    def un_z_score(data, node_id):
        shape = data.shape
        data = data.reshape(-1, len(node_id), 1)
        return (
            data * data_std_per_node_id[node_id] + data_mean_per_node_id[node_id]
        ).reshape(shape)

    return z_score, un_z_score
