"""Embedding networks that condense high-dimensional observations into tokens.

This follows Gloeckler et al. (2024), *All-in-one simulation-based inference*
(Sec. 3.1 and Appendix A3.2): rather than treating every scalar of a
high-dimensional observation as its own variable/token, a specialized embedding
network compresses whole vectors into a single token (or a few tokens).  The
embedding network is trained end-to-end together with the score model, and can
afterwards be extracted and reused on its own -- e.g. as the observation
encoder of a density-ratio estimator.

    "Finally, specialized embedding networks, commonly used in SBI algorithms
     and trained end-to-end, can be efficiently integrated here by condensing
     complex data into a single token. This reduces computational complexity
     but loses direct control over dependencies and condition states for
     individual data elements."   -- Sec. 3.1

The encoder is a *pure function of the observation*, ``f(x) -> tokens``.  It
deliberately does not take the diffusion time ``t`` or the condition mask as
input: in the targeted-inference setting the observation tokens are always
conditioned, hence always clean.  That is exactly what makes the encoder
reusable outside the diffusion model.

Two granularities are supported:

``chunked=False`` (default, the paper's setting)
    The whole observation vector is passed through one MLP that emits
    ``num_x_tokens`` tokens.  With ``num_x_tokens=1`` this is literally
    "condense complex data into a single token".

``chunked=True``
    The observation is split into ``num_x_tokens`` contiguous blocks, each with
    its own MLP emitting one token.  This mirrors the paper's gravitational
    wave setup, where each detector's measurement becomes one token, and keeps
    *block-level* conditioning available (partial posteriors such as
    ``p(theta | x_block1)``).
"""

from typing import Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array


def chunk_boundaries(x_dim: int, num_chunks: int) -> Tuple[Tuple[int, int], ...]:
    """Split ``range(x_dim)`` into ``num_chunks`` contiguous, near-equal blocks.

    Leading chunks absorb the remainder, so e.g. ``(101, 4)`` yields sizes
    ``(26, 25, 25, 25)``.  Returned as static Python ints, so the boundaries can
    be used for slicing under ``jit``.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if num_chunks > x_dim:
        raise ValueError(
            f"num_chunks ({num_chunks}) cannot exceed x_dim ({x_dim})"
        )
    base, rem = divmod(x_dim, num_chunks)
    bounds = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return tuple(bounds)


class XEmbeddingNet(hk.Module):
    """Compress an observation vector into transformer tokens.

    Args:
        token_dim: Width of each emitted token (must match the tokenizer's
            ``output_dim`` so observation tokens can be concatenated with the
            parameter tokens).
        x_dim: Dimensionality of the observation vector.
        num_x_tokens: Number of tokens the observation is condensed into.
        hidden_sizes: Widths of the MLP hidden layers.
        chunked: If True, split the observation into ``num_x_tokens`` contiguous
            blocks and give each its own MLP.  If False, one MLP sees the whole
            vector and emits all tokens.
        act: Activation function.
        layer_norm: Apply LayerNorm to the emitted tokens.  Recommended: it puts
            the observation tokens on the same scale as the tokenizer's output
            and stabilises early training.
        name: Haiku module name.  Kept stable so the trained sub-tree can be
            located in the parameter dict and extracted.
    """

    def __init__(
        self,
        token_dim: int,
        x_dim: int,
        num_x_tokens: int = 1,
        hidden_sizes: Sequence[int] = (256, 128),
        chunked: bool = False,
        act=jax.nn.gelu,
        layer_norm: bool = True,
        name: str = "x_embedding_net",
    ):
        super().__init__(name=name)
        self.token_dim = token_dim
        self.x_dim = x_dim
        self.num_x_tokens = num_x_tokens
        self.hidden_sizes = tuple(hidden_sizes)
        self.chunked = chunked
        self.act = act
        self.layer_norm = layer_norm

    @hk.transparent
    def _mlp(self, x: Array, output_dim: int, tag: str) -> Array:
        h = x
        for i, width in enumerate(self.hidden_sizes):
            h = hk.Linear(width, name=f"{tag}_hidden_{i}")(h)
            h = self.act(h)
        return hk.Linear(output_dim, name=f"{tag}_out")(h)

    def __call__(self, x: Array) -> Array:
        """Encode observations.

        Args:
            x: ``(..., x_dim)`` observation values.

        Returns:
            ``(..., num_x_tokens, token_dim)`` observation tokens.
        """
        *leading_dims, x_dim = x.shape
        if x_dim != self.x_dim:
            raise ValueError(
                f"XEmbeddingNet expected x_dim={self.x_dim}, got {x_dim}"
            )
        x = x.reshape(-1, x_dim)

        if self.chunked:
            bounds = chunk_boundaries(self.x_dim, self.num_x_tokens)
            tokens = [
                self._mlp(x[:, lo:hi], self.token_dim, f"chunk_{i}")
                for i, (lo, hi) in enumerate(bounds)
            ]
            out = jnp.stack(tokens, axis=-2)
        else:
            out = self._mlp(x, self.num_x_tokens * self.token_dim, "global")
            out = out.reshape(-1, self.num_x_tokens, self.token_dim)

        if self.layer_norm:
            out = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="token_norm"
            )(out)

        return out.reshape(*leading_dims, self.num_x_tokens, self.token_dim)


class XScoreDecoder(hk.Module):
    """Map observation-token hidden states back to a score per observation scalar.

    Only needed when observation tokens may be *latent* (i.e. the model is asked
    for conditionals that generate ``x``, such as the joint or the likelihood).
    In the targeted-inference setting of Appendix A3.2 -- where the observation
    is always conditioned -- this decoder is never exercised, and the model can
    emit zeros for the observation entries instead (see ``predict_x_score``
    in ``scalar_transformer_model``).

    When ``chunked=True`` the decoding is local: each token is decoded back into
    exactly the block it was encoded from, so there is no global bottleneck.
    """

    def __init__(
        self,
        x_dim: int,
        num_x_tokens: int = 1,
        hidden_sizes: Sequence[int] = (128, 256),
        chunked: bool = False,
        act=jax.nn.gelu,
        name: str = "x_score_decoder",
    ):
        super().__init__(name=name)
        self.x_dim = x_dim
        self.num_x_tokens = num_x_tokens
        self.hidden_sizes = tuple(hidden_sizes)
        self.chunked = chunked
        self.act = act

    @hk.transparent
    def _mlp(self, x: Array, output_dim: int, tag: str) -> Array:
        h = x
        for i, width in enumerate(self.hidden_sizes):
            h = hk.Linear(width, name=f"{tag}_hidden_{i}")(h)
            h = self.act(h)
        return hk.Linear(output_dim, name=f"{tag}_out")(h)

    def __call__(self, h: Array) -> Array:
        """Decode observation-token hidden states.

        Args:
            h: ``(..., num_x_tokens, hidden_dim)`` transformer outputs for the
                observation tokens.

        Returns:
            ``(..., x_dim, 1)`` per-scalar scores.
        """
        *leading_dims, num_tokens, hidden_dim = h.shape
        if num_tokens != self.num_x_tokens:
            raise ValueError(
                f"XScoreDecoder expected {self.num_x_tokens} tokens, got {num_tokens}"
            )
        h = h.reshape(-1, num_tokens, hidden_dim)

        if self.chunked:
            bounds = chunk_boundaries(self.x_dim, self.num_x_tokens)
            parts = [
                self._mlp(h[:, i, :], hi - lo, f"chunk_{i}")
                for i, (lo, hi) in enumerate(bounds)
            ]
            out = jnp.concatenate(parts, axis=-1)
        else:
            flat = h.reshape(-1, num_tokens * hidden_dim)
            out = self._mlp(flat, self.x_dim, "global")

        return out.reshape(*leading_dims, self.x_dim, 1)


# ---------------------------------------------------------------------------
# Standalone reuse of a trained embedding network
# ---------------------------------------------------------------------------


def split_embedding_params(params, prefix: str = "x_embedding_net"):
    """Split a Haiku parameter dict into the embedding sub-tree and the rest.

    Args:
        params: Parameter dict of a trained score model.
        prefix: Module name of the embedding network.

    Returns:
        ``(embedding_params, remaining_params)``; both are plain dicts keyed by
        the original Haiku module paths.
    """

    def _is_embed(key: str) -> bool:
        return key == prefix or key.startswith(prefix + "/")

    embedding_params = {k: v for k, v in params.items() if _is_embed(k)}
    remaining_params = {k: v for k, v in params.items() if not _is_embed(k)}
    return embedding_params, remaining_params


def extract_x_embedding_net(
    params,
    token_dim: int,
    x_dim: int,
    num_x_tokens: int = 1,
    hidden_sizes: Sequence[int] = (256, 128),
    chunked: bool = False,
    act=jax.nn.gelu,
    layer_norm: bool = True,
    prefix: str = "x_embedding_net",
    flatten: bool = True,
):
    """Pull the trained embedding network out of a score model's parameters.

    The returned callable is a pure JAX function of the observation only, so it
    can be dropped into any downstream model (e.g. a density-ratio estimator)
    and either frozen or fine-tuned.

    Args:
        params: Parameter dict of the trained score model (``model.params``).
        token_dim, x_dim, num_x_tokens, hidden_sizes, chunked, act, layer_norm:
            Must match the configuration the score model was trained with.
        prefix: Module name of the embedding network inside ``params``.
        flatten: If True the returned embedding is flattened to
            ``(..., num_x_tokens * token_dim)``; otherwise it keeps the
            ``(..., num_x_tokens, token_dim)`` token layout.

    Returns:
        ``(apply_fn, embedding_params, embedding_dim)`` where
        ``apply_fn(embedding_params, x)`` encodes an observation batch.

    Raises:
        KeyError: If no parameters with ``prefix`` are found -- i.e. the model
            was not trained with ``embed_x=True``.
    """
    embedding_params, _ = split_embedding_params(params, prefix=prefix)
    if not embedding_params:
        raise KeyError(
            f"No parameters found under prefix '{prefix}'. Was this model "
            f"trained with embed_x=True? Available top-level keys: "
            f"{sorted(params.keys())}"
        )

    def _fn(x):
        tokens = XEmbeddingNet(
            token_dim=token_dim,
            x_dim=x_dim,
            num_x_tokens=num_x_tokens,
            hidden_sizes=hidden_sizes,
            chunked=chunked,
            act=act,
            layer_norm=layer_norm,
            name=prefix,
        )(x)
        if flatten:
            return tokens.reshape(*tokens.shape[:-2], num_x_tokens * token_dim)
        return tokens

    _, apply_fn = hk.without_apply_rng(hk.transform(_fn))

    embedding_dim = num_x_tokens * token_dim if flatten else token_dim
    return apply_fn, embedding_params, embedding_dim


def token_condition_mask(
    condition_mask: Array,
    theta_dim: int,
    x_dim: int,
    num_x_tokens: int,
    chunked: bool,
) -> Array:
    """Reduce a per-scalar condition mask to a per-token condition mask.

    An observation token counts as conditioned only if *every* scalar it
    encodes is conditioned -- a partially observed block cannot be represented
    by a clean embedding.

    Args:
        condition_mask: ``(batch, theta_dim + x_dim)`` per-scalar mask.
        theta_dim, x_dim, num_x_tokens, chunked: Model configuration.

    Returns:
        ``(batch, theta_dim + num_x_tokens)`` per-token mask, same dtype as the
        input.
    """
    dtype = condition_mask.dtype
    theta_mask = condition_mask[:, :theta_dim]
    x_mask = condition_mask[:, theta_dim:].astype(bool)

    if chunked:
        bounds = chunk_boundaries(x_dim, num_x_tokens)
        token_mask = jnp.stack(
            [jnp.all(x_mask[:, lo:hi], axis=-1) for lo, hi in bounds], axis=-1
        )
    else:
        all_conditioned = jnp.all(x_mask, axis=-1, keepdims=True)
        token_mask = jnp.broadcast_to(
            all_conditioned, (x_mask.shape[0], num_x_tokens)
        )

    return jnp.concatenate([theta_mask, token_mask.astype(dtype)], axis=-1)
