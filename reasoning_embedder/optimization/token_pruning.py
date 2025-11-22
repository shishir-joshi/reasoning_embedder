"""Token pruning utilities for ColBERT-style embeddings.

This module provides a small, well-tested set of utilities for pruning
token-level embeddings using attention-aware, length, random, or threshold
strategies. It supports both NumPy arrays and PyTorch tensors and is intended
to be simple and reusable.

API functions:
- apply_pruning_mask
- generate_pruning_mask
- prune_colbert_embeddings

The default assumptions:
- min_tokens is absolute (e.g., 1 means at least 1 token preserved)
- Random strategy uses seed=None by default and accepts a seed param for
  deterministic behavior when required.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def apply_pruning_mask(embeddings: ArrayLike, mask: ArrayLike) -> ArrayLike:
    """Apply a boolean mask to embeddings (numpy or torch).

    Args:
        embeddings: np.ndarray or torch.Tensor with shape (N, D).
        mask: boolean array/tensor of length N.

    Returns:
        The pruned embeddings of shape (K, D) where K <= N.
    """
    if isinstance(embeddings, np.ndarray):
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 1 or mask_arr.shape[0] != embeddings.shape[0]:
            raise ValueError("Mask must be 1D and match the number of tokens")
        return embeddings[mask_arr]

    if _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
        if torch.is_tensor(mask):
            m = mask.to(dtype=torch.bool)
        else:
            m = torch.tensor(np.asarray(mask, dtype=bool), device=embeddings.device)
        if m.ndim != 1 or m.shape[0] != embeddings.shape[0]:
            raise ValueError("Mask must be 1D and match the number of tokens (torch)")
        return embeddings[m]

    raise ValueError("Embeddings must be numpy array or torch tensor")


def _ensure_min_tokens(k: int, min_tokens: int, N: int) -> int:
    return max(min_tokens, min(k, N))


def generate_pruning_mask(
    attention_weights: Optional[ArrayLike] = None,
    keep_ratio: float = 0.6,
    strategy: str = "attention",
    N: Optional[int] = None,
    min_tokens: int = 1,
    preserve_indices: Optional[Iterable[int]] = None,
    seed: Optional[int] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """Generate a boolean mask to prune tokens.

    Supported strategies: 'attention', 'length', 'random', 'threshold', 'combined'.

    - attention: uses attention_weights to select top-k tokens.
    - length: keeps first k tokens.
    - random: randomly selects k tokens.
    - threshold: selects tokens above a percentile threshold derived from keep_ratio.
    - combined: prefers attention when available, otherwise falls back to length.

    Args:
        attention_weights: array-like of shape (N,) when applicable.
        keep_ratio: fraction of tokens to retain (0 < keep_ratio <= 1).
        strategy: selection strategy.
        N: total number of tokens. Required when attention_weights is None for some strategies.
        min_tokens: absolute minimum tokens to retain, irrespective of keep_ratio.
        preserve_indices: indices to always keep (special tokens).
        seed: random seed for 'random' strategy.

    Returns:
        boolean mask of length N with True for tokens to keep.
    """
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError("keep_ratio must be in (0, 1]")

    preserve_set = set(preserve_indices or [])
    if attention_weights is None and N is None:
        raise ValueError("When attention_weights is None, N must be provided")
    if attention_weights is not None:
        if isinstance(attention_weights, np.ndarray):
            att = attention_weights
        elif _TORCH_AVAILABLE and isinstance(attention_weights, torch.Tensor):
            att = attention_weights.cpu().detach().numpy()
        else:
            att = np.asarray(attention_weights)
        if att.ndim != 1:
            raise ValueError("attention_weights must be a 1D array")
        N = att.shape[0]

    assert N is not None
    k = int(np.ceil(keep_ratio * N))
    k = _ensure_min_tokens(k, min_tokens, N)
    mask = np.zeros(N, dtype=bool)

    if strategy == "attention":
        if attention_weights is None:
            raise ValueError("attention_weights required for attention strategy")
        # choose top-k indices by attention
        topk_idx = np.argpartition(-att, k - 1)[:k]
        mask[topk_idx] = True

    elif strategy == "length":
        mask[:k] = True

    elif strategy == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=k, replace=False)
        mask[idx] = True

    elif strategy == "threshold":
        if attention_weights is None:
            raise ValueError("attention_weights required for threshold strategy")
        # threshold is the quantile at (1-keep_ratio)
        thresh = np.quantile(att, 1.0 - keep_ratio)
        mask = att >= thresh
        # if below min tokens, fallback to top-k
        if mask.sum() < min_tokens:
            topk_idx = np.argpartition(-att, k - 1)[:k]
            mask = np.zeros(N, dtype=bool)
            mask[topk_idx] = True

    elif strategy == "combined":
        if attention_weights is not None:
            # attention then fallback
            topk_idx = np.argpartition(-att, k - 1)[:k]
            mask[topk_idx] = True
        else:
            mask[:k] = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Always preserve indices
    for idx in preserve_set:
        if 0 <= idx < N:
            mask[idx] = True

    # Ensure at least min_tokens are selected
    if mask.sum() < min_tokens:
        # Select top by attention if available otherwise by index
        if attention_weights is not None:
            topk_idx = np.argpartition(-att, min_tokens - 1)[:min_tokens]
            mask[topk_idx] = True
        else:
            mask[:min_tokens] = True

    if _TORCH_AVAILABLE and isinstance(attention_weights, torch.Tensor):
        # If input was torch, return torch mask on same device
        return torch.tensor(mask, device=attention_weights.device)
    return mask


def prune_colbert_embeddings(
    embeddings: ArrayLike,
    attention_weights: Optional[ArrayLike] = None,
    keep_ratio: float = 0.6,
    strategy: str = "attention",
    min_tokens: int = 1,
    preserve_indices: Optional[Iterable[int]] = None,
    return_mask: bool = False,
    seed: Optional[int] = None,
    N: Optional[int] = None,
) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """Prune ColBERT-style token embeddings.

    This wrapper supports numpy and torch arrays and returns pruned embeddings
    in the same type as input. When `return_mask=True`, the boolean mask is
    also returned.
    """
    if isinstance(embeddings, np.ndarray):
        N = embeddings.shape[0] if N is None else N
    elif _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
        N = embeddings.shape[0] if N is None else N
    else:
        raise ValueError("Embeddings must be numpy ndarray or torch tensor")

    if N == 0:
        # Nothing to prune
        mask = np.ones(0, dtype=bool)
        if _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
            mask = torch.tensor(mask, device=embeddings.device)
        if return_mask:
            return embeddings, mask
        return embeddings

    if keep_ratio >= 1.0:
        mask = np.ones(N, dtype=bool)
        if _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
            mask = torch.tensor(mask, device=embeddings.device)
        if return_mask:
            return embeddings, mask
        return embeddings

    # generate mask
    mask = generate_pruning_mask(
        attention_weights=attention_weights,
        keep_ratio=keep_ratio,
        strategy=strategy,
        N=N,
        min_tokens=min_tokens,
        preserve_indices=preserve_indices,
        seed=seed,
    )

    pruned = apply_pruning_mask(embeddings, mask)
    if return_mask:
        return pruned, mask
    return pruned
