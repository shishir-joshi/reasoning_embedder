"""Token pruning utilities for ColBERT-style embeddings.

This module provides utilities for pruning token-level embeddings at two stages:
1. Tokenization-time pruning: Mask input tokens before encoding
2. Post-encoding pruning: Reduce output embeddings after model forward pass

Tokenization-time (input pruning):
- apply_pruning_mask
- generate_pruning_mask
- prune_colbert_embeddings

Post-encoding (output embedding pruning):
- prune_embeddings_hierarchical - Semantic clustering-based pooling (production-ready)
- prune_embeddings_attention - Fast attention-based pruning
- prune_embeddings_batch - Batch processing wrapper

All functions support both NumPy arrays and PyTorch tensors.

The default assumptions:
- min_tokens is absolute (e.g., 1 means at least 1 token preserved)
- Random strategy uses seed=None by default and accepts a seed param for
  deterministic behavior when required.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union, List
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

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


# ============================================================================
# Post-Encoding Embedding Pruning (Output Space)
# ============================================================================


def prune_embeddings_hierarchical(
    embeddings: ArrayLike,
    pool_factor: float = 2.0,
    protected_tokens: int = 2,
    linkage_method: str = "average",
) -> ArrayLike:
    """Prune token embeddings using hierarchical clustering.

    Groups similar token embeddings via hierarchical clustering and averages
    within each cluster. This is the production-proven approach from ColBERT.

    Args:
        embeddings: Token embeddings [num_tokens, dim] (numpy or torch)
        pool_factor: Target reduction ratio (2.0 = 50% reduction)
        protected_tokens: Number of initial tokens to preserve (e.g., [CLS], [D])
        linkage_method: Scipy linkage method ('average', 'ward', 'complete')

    Returns:
        Pruned embeddings [num_pruned_tokens, dim] where num_pruned_tokens ≈ num_tokens/pool_factor
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for hierarchical pruning. Install: pip install scipy")

    is_torch = _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor)
    device = embeddings.device if is_torch else None

    # Convert to numpy for scipy
    if is_torch:
        embs_np = embeddings.cpu().detach().numpy()
    else:
        embs_np = np.asarray(embeddings)

    N, D = embs_np.shape

    # Edge case: too few tokens
    if N <= protected_tokens + 1:
        return embeddings

    # Separate protected and cluster-able tokens
    protected = embs_np[:protected_tokens]
    cluster_embs = embs_np[protected_tokens:]

    # Calculate target cluster count
    n_clusters = max(1, int(np.ceil(len(cluster_embs) / pool_factor)))

    if n_clusters >= len(cluster_embs):
        # No reduction needed
        return embeddings

    # Hierarchical clustering
    Z = linkage(cluster_embs, method=linkage_method)
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Average embeddings within each cluster
    pooled = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labels == cluster_id
        if cluster_mask.any():
            cluster_mean = cluster_embs[cluster_mask].mean(axis=0)
            pooled.append(cluster_mean)

    pooled = np.array(pooled) if pooled else np.zeros((0, D))

    # Concatenate protected + pooled
    result = np.vstack([protected, pooled]) if len(pooled) > 0 else protected

    # Convert back to torch if needed
    if is_torch:
        result = torch.tensor(result, dtype=embeddings.dtype, device=device)

    return result


def prune_embeddings_attention(
    embeddings: ArrayLike,
    attention_weights: Optional[ArrayLike] = None,
    keep_ratio: float = 0.6,
    protected_tokens: int = 2,
) -> ArrayLike:
    """Prune token embeddings using attention-based importance scores.

    Keeps tokens with highest attention scores. Faster than hierarchical
    but less semantic-aware. If no attention weights provided, uses
    self-attention (cosine similarity to mean).

    Args:
        embeddings: Token embeddings [num_tokens, dim]
        attention_weights: Optional importance scores [num_tokens]
        keep_ratio: Fraction of tokens to keep (0 < keep_ratio <= 1)
        protected_tokens: Number of initial tokens to preserve

    Returns:
        Pruned embeddings [num_kept_tokens, dim]
    """
    is_torch = _TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor)
    device = embeddings.device if is_torch else None

    if is_torch:
        embs = embeddings
    else:
        embs = np.asarray(embeddings)

    N, D = embs.shape if is_torch else embs.shape

    # Edge case
    if N <= protected_tokens + 1:
        return embeddings

    # Calculate importance if not provided
    if attention_weights is None:
        if is_torch:
            # Self-attention: similarity to mean
            mean_emb = embs.mean(dim=0, keepdim=True)
            importance = (embs @ mean_emb.T).squeeze(-1)
        else:
            mean_emb = embs.mean(axis=0, keepdims=True)
            importance = (embs @ mean_emb.T).squeeze(-1)
    else:
        if is_torch and not torch.is_tensor(attention_weights):
            importance = torch.tensor(attention_weights, device=device)
        elif not is_torch and not isinstance(attention_weights, np.ndarray):
            importance = np.asarray(attention_weights)
        else:
            importance = attention_weights

    # Separate protected tokens
    cluster_importance = importance[protected_tokens:]
    n_keep = max(1, int(np.ceil(len(cluster_importance) * keep_ratio)))

    # Select top-k by importance
    if is_torch:
        _, top_indices = torch.topk(cluster_importance, k=n_keep)
        # Adjust indices for protected offset
        top_indices = top_indices + protected_tokens
        # Combine protected + selected
        keep_indices = torch.cat([torch.arange(protected_tokens, device=device), top_indices])
        keep_indices = torch.sort(keep_indices)[0]  # maintain order
        result = embs[keep_indices]
    else:
        top_indices = np.argpartition(-cluster_importance, n_keep - 1)[:n_keep]
        top_indices = top_indices + protected_tokens
        keep_indices = np.concatenate([np.arange(protected_tokens), top_indices])
        keep_indices = np.sort(keep_indices)
        result = embs[keep_indices]

    return result


def prune_embeddings_batch(
    embeddings_batch: List[ArrayLike],
    doclens: List[int],
    strategy: str = "hierarchical",
    pool_factor: float = 2.0,
    keep_ratio: float = 0.6,
    protected_tokens: int = 2,
    attention_batch: Optional[List[ArrayLike]] = None,
) -> Tuple[List[ArrayLike], List[int]]:
    """Prune a batch of variable-length token embeddings.

    Args:
        embeddings_batch: List of token embeddings, each [num_tokens_i, dim]
        doclens: Original token counts for each item
        strategy: 'hierarchical' or 'attention'
        pool_factor: For hierarchical (2.0 = 50% reduction)
        keep_ratio: For attention (0.6 = keep 60%)
        protected_tokens: Preserve first N tokens
        attention_batch: Optional attention weights per item

    Returns:
        (pruned_batch, new_doclens) - pruned embeddings and updated lengths
    """
    pruned = []
    new_lens = []

    for i, embs in enumerate(embeddings_batch):
        attention = attention_batch[i] if attention_batch else None

        if strategy == "hierarchical":
            pruned_embs = prune_embeddings_hierarchical(
                embs, pool_factor=pool_factor, protected_tokens=protected_tokens
            )
        elif strategy == "attention":
            pruned_embs = prune_embeddings_attention(
                embs, attention_weights=attention, keep_ratio=keep_ratio,
                protected_tokens=protected_tokens
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        pruned.append(pruned_embs)
        new_lens.append(len(pruned_embs) if hasattr(pruned_embs, '__len__') else pruned_embs.shape[0])

    return pruned, new_lens
