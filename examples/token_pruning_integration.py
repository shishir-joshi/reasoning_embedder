"""Example integration for token pruning.

Shows how to call prune_colbert_embeddings in a small pipeline.
"""
import numpy as np
from reasoning_embedder.optimization.token_pruning import prune_colbert_embeddings


def demo():
    # Simulate token-level embeddings for a long document
    embeddings = np.random.randn(1024, 128).astype(np.float32)
    # Simulate attention scores (e.g., last-layer attention sum)
    attention = np.random.rand(1024)
    print("Original tokens:", embeddings.shape[0])

    pruned = prune_colbert_embeddings(
        embeddings,
        attention_weights=attention,
        keep_ratio=0.3,
        strategy='attention',
    )
    print("Pruned tokens:", pruned.shape[0])


if __name__ == "__main__":
    demo()
