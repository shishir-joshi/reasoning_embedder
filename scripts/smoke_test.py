from __future__ import annotations

import numpy as np


def main() -> None:
    # Basic import
    import reasoning_embedder  # noqa: F401

    # Prepare pipeline core transformation (no HF/network)
    from reasoning_embedder.data.prepare import process_documents

    entry = {
        "query": ["hello", "world"],
        "pos": [["inst", "1"], ["malformed"], "oops"],
        "neg": [["inst", "2"], ["inst", "already text"], ["bad"]],
    }
    id2doc = {"1": "doc one", "2": "doc two"}
    out = process_documents(entry, id2doc)
    assert out["query"] == "hello world"
    assert out["pos"] == [["inst", "doc one"]]
    assert out["neg"][0] == ["inst", "doc two"]

    # Token/embedding pruning utilities should import and run
    from reasoning_embedder.optimization.token_pruning import (
        generate_pruning_mask,
        prune_embeddings_attention,
        prune_embeddings_hierarchical,
    )

    rng = np.random.default_rng(0)
    token_ids = np.array([101, 5, 6, 7, 102], dtype=np.int64)
    mask = generate_pruning_mask(N=token_ids.shape[0], keep_ratio=1.0, strategy="length")
    assert mask.shape == token_ids.shape and mask.all()

    emb = rng.normal(size=(32, 16)).astype(np.float32)
    pruned_h = prune_embeddings_hierarchical(emb, pool_factor=2.0)
    assert pruned_h.ndim == 2 and pruned_h.shape[1] == emb.shape[1]

    attn = np.linspace(1.0, 0.2, emb.shape[0]).astype(np.float32)
    pruned_a = prune_embeddings_attention(emb, attention_weights=attn, keep_ratio=0.5)
    assert pruned_a.ndim == 2 and pruned_a.shape[1] == emb.shape[1]

    print("✓ smoke_test OK")


if __name__ == "__main__":
    main()
