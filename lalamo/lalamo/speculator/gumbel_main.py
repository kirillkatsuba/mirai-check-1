import struct
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
from typing import Self, Optional

import jax
import jax.numpy as jnp

from .common import Speculator


def generate_gumbel_noise(key: jax.Array, shape: tuple) -> jax.Array:
    """
    Generates Gumbel(0, 1) noise.
    G = -log(-log(U)) where U ~ Uniform(0, 1)
    """
    # Use a small epsilon to prevent log(0)
    u = jax.random.uniform(key, shape=shape, minval=1e-10, maxval=1.0)
    return -jnp.log(-jnp.log(u))


def update_gumbels_coupling(
    old_scores: dict[int, float], 
    new_logits: dict[int, float], 
    rng_key: jax.Array, 
    top_k: int
) -> dict[int, float]:
    """
    Performs the Gumbel-Max Coupling update.
    
    Logic:
    1. Generate independent Gumbel noise for the new logits.
    2. Calculate New Score = Logit + Noise.
    3. Coupling: Take the MAX of the Old Score and New Score.
       S_final = max(S_old, S_new)
    
    This maintains the upper envelope of the Gumbel processes.
    """
    token_ids = list(new_logits.keys())
    logits_val = jnp.array(list(new_logits.values()))
    
    # 1. Generate noise
    noise = generate_gumbel_noise(rng_key, logits_val.shape)
    
    # 2. Add noise to logits (Gumbel-Max trick basics)
    new_scores_val = logits_val + noise
    
    # Map back to token IDs
    current_step_scores = dict(zip(token_ids, new_scores_val.tolist(), strict=True))

    # Union of all tokens seen so far
    all_keys = set(old_scores.keys()).union(current_step_scores.keys())

    # 3. The Coupling Step (Max-Aggregation)
    merged_scores = {}
    for k in all_keys:
        # Default to -1e9 (approx -infinity) if token not present in one of the sets
        val_old = old_scores.get(k, -1e9)
        val_new = current_step_scores.get(k, -1e9)
        
        # The core of the coupling: keep the highest Gumbel score seen
        merged_scores[k] = max(val_old, val_new)

    # Sort descending by score and keep only Top-K
    top_k_items = sorted(merged_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    
    return dict(top_k_items)


@dataclass(frozen=True, eq=False)
class GumbelSpeculator(Speculator):
    """
    A Context-Free (Unigram) Speculator using Gumbel-Max Coupling.
    It tracks the global 'best' Gumbel scores for tokens.
    """
    top_k: int
    
    # Internal storage using arrays for efficient serialization/interop
    # These represent a single "bucket" of global stats
    _keys: array[int]
    _values: array[float]

    def __post_init__(self) -> None:
        if not self.top_k > 0:
            raise ValueError(f"{self.top_k=} (must be > 0)")

    @classmethod
    def new(cls, top_k: int) -> Self:
        return cls(
            top_k,
            # Initialize keys to 0
            array("I", range(top_k)),
            # Initialize values to -1e9 (approx -inf) because Gumbel scores are real numbers (can be negative)
            # 0.0 is NOT a safe init value here.
            array("f", repeat(-1e9, top_k)),
        )

    def train(self, token_logits: Iterable[dict[int, float]], rng_key: Optional[jax.Array] = None) -> None:
        """
        Updates the global speculator state with a sequence of observations.
        
        :param token_logits: Iterable of dicts {token_id: logit_value}
        :param rng_key: JAX PRNG Key for Gumbel noise generation.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)

        # Reconstruct current state from flat arrays
        # Filter out initialization garbage (-1e9)
        old_scores = {}
        for k, v in zip(self._keys, self._values, strict=True):
            if v > -1e8: 
                old_scores[k] = v

        # Iterate over every observation in the batch/sequence
        for cur_logits in token_logits:
            rng_key, step_key = jax.random.split(rng_key)
            
            # Update state using Max-Coupling
            old_scores = update_gumbels_coupling(old_scores, cur_logits, step_key, self.top_k)

        # Write final state back to the arrays
        sorted_keys = list(old_scores.keys())
        sorted_vals = list(old_scores.values())
        
        # Pad with dummies if we haven't seen K unique tokens yet
        while len(sorted_keys) < self.top_k:
            sorted_keys.append(0)
            sorted_vals.append(-1e9)

        self._keys[:] = array("I", sorted_keys)
        self._values[:] = array("f", sorted_vals)

    def probs(self, seq: Iterable[int]) -> dict[int, float]:
        """
        Returns the current Gumbel scores.
        Note: The input 'seq' (context) is ignored because this is a Unigram model (No N-gram).
        """
        # Filter valid scores
        res = {}
        for k, v in zip(self._keys, self._values, strict=True):
            if v > -1e8:
                res[k] = v
        return res

    def serialize(self) -> bytes:
        # Format: [top_k (uint32)] [keys array] [values array]
        hdr = struct.pack("<I", self.top_k)
        return hdr + bytes(self._keys) + bytes(self._values)

    @classmethod
    def deserialize(cls, blob: bytes) -> Self:
        offset = 4
        (top_k,) = struct.unpack("<I", blob[:offset])

        # Calculate lengths
        # Keys are uint32 (4 bytes)
        keys_len = 4 * top_k
        keys = array("I", blob[offset : offset + keys_len])
        offset += keys_len

        # Values are float32 (4 bytes)
        vals_len = 4 * top_k
        values = array("f", blob[offset : offset + vals_len])
        offset += vals_len

        return cls(top_k, keys, values)