import numpy as np
import sys
import math
import pytest

try:
    import torch
    _TORCH = True
except Exception:
    torch = None  # type: ignore
    _TORCH = False

from reasoning_embedder.optimization.token_pruning import (
    apply_pruning_mask,
    generate_pruning_mask,
    prune_colbert_embeddings,
    prune_embeddings_hierarchical,
    prune_embeddings_attention,
    prune_embeddings_batch,
)


class TestApplyPruningMask:
    def test_basic_masking_numpy(self):
        emb = np.arange(12).reshape(6, 2).astype(float)
        mask = np.array([True, False, True, False, True, False])
        pr = apply_pruning_mask(emb, mask)
        assert pr.shape == (3, 2)
        assert np.allclose(pr, emb[[0, 2, 4]])

    def test_basic_masking_torch(self):
        if not _TORCH:
            pytest.skip("torch not available")
        t = torch.arange(12).reshape(6, 2).float()
        mask = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)
        pr = apply_pruning_mask(t, mask)
        assert pr.shape == (3, 2)
        assert torch.equal(pr, t[mask])


class TestGeneratePruningMask:
    def test_attention_pruning(self):
        att = np.array([0.1, 0.9, 0.4, 0.05, 0.0])
        mask = generate_pruning_mask(attention_weights=att, keep_ratio=0.4, strategy='attention', min_tokens=1)
        # keep_ratio=0.4 -> k=2
        assert mask.sum() >= 1
        # ensure highest attention kept
        top = np.argmax(att)
        assert mask[top]

    def test_length_pruning(self):
        mask = generate_pruning_mask(keep_ratio=0.5, strategy='length', N=6)
        assert mask.sum() == 3
        assert mask[0] and mask[1] and mask[2]

    def test_random_pruning_deterministic(self):
        mask1 = generate_pruning_mask(keep_ratio=0.5, strategy='random', N=10, seed=42)
        mask2 = generate_pruning_mask(keep_ratio=0.5, strategy='random', N=10, seed=42)
        assert np.array_equal(mask1, mask2)

    def test_threshold_pruning(self):
        att = np.array([0.1, 0.9, 0.2, 0.7, 0.05])
        mask = generate_pruning_mask(attention_weights=att, keep_ratio=0.4, strategy='threshold', min_tokens=1)
        assert mask.sum() >= 1

    def test_combined_pruning_with_no_attention(self):
        mask = generate_pruning_mask(keep_ratio=0.25, strategy='combined', N=8)
        k = int(np.ceil(0.25 * 8))
        assert mask.sum() == k


class TestPruneColBERT:
    def test_prune_numpy_basic(self):
        emb = np.random.rand(20, 16)
        att = np.linspace(0, 1, 20)
        pr = prune_colbert_embeddings(emb, attention_weights=att, keep_ratio=0.5, strategy='attention')
        assert pr.shape[1] == 16
        assert pr.shape[0] <= emb.shape[0]

    def test_prune_torch_basic(self):
        if not _TORCH:
            pytest.skip('torch not available')
        t = torch.rand(20, 16)
        att = torch.linspace(0, 1, 20)
        pr = prune_colbert_embeddings(t, attention_weights=att, keep_ratio=0.5, strategy='attention')
        assert pr.shape[1] == 16
        assert pr.shape[0] <= t.shape[0]

    def test_preserve_indices(self):
        emb = np.random.rand(12, 8)
        att = np.random.rand(12)
        pr = prune_colbert_embeddings(emb, attention_weights=att, keep_ratio=0.3, preserve_indices=[0, 11])
        # preserved indices must be present in pruned
        # compute mask
        pr_all, mask = prune_colbert_embeddings(emb, attention_weights=att, keep_ratio=0.3, preserve_indices=[0, 11], return_mask=True)
        assert mask[0] and mask[11]

    def test_min_tokens_respected(self):
        emb = np.random.rand(6, 8)
        att = np.zeros(6)
        pr = prune_colbert_embeddings(emb, attention_weights=att, keep_ratio=0.1, min_tokens=2)
        assert pr.shape[0] >= 2

    def test_keep_ratio_one_returns_same(self):
        emb = np.random.rand(10, 8)
        pr = prune_colbert_embeddings(emb, keep_ratio=1.0)
        assert pr.shape == emb.shape

    def test_empty_inputs(self):
        emb = np.zeros((0, 8))
        pr = prune_colbert_embeddings(emb, keep_ratio=0.5, N=0)
        assert pr.shape[0] == 0

    def test_memory_efficiency(self):
        emb = np.random.rand(200, 64)
        orig_bytes = emb.nbytes
        pr = prune_colbert_embeddings(emb, keep_ratio=0.2, strategy='random', seed=123, N=200)
        pr_bytes = pr.nbytes
        assert pr_bytes < orig_bytes


def test_invalid_keep_ratio_raises():
    emb = np.random.rand(5, 4)
    with pytest.raises(ValueError):
        prune_colbert_embeddings(emb, keep_ratio=0.0)


def test_wrap_tokenize_with_pruning_basic():
    try:
        import torch
    except Exception:
        pytest.skip("torch not available")

    from reasoning_embedder.training.build import wrap_tokenize_with_pruning

    # Dummy tokenizer object with pad/cls/sep ids
    class DummyTok:
        pad_token_id = 0
        cls_token_id = 101
        sep_token_id = 102

    tok = DummyTok()

    # Dummy tokenize function that returns tensors of shape (B, L)
    def dummy_tokenize(texts, is_query=False, pad_document=True):
        B = len(texts)
        L = 8
        ids = torch.zeros((B, L), dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for i, t in enumerate(texts):
            # simple 'tokens' count equals number of words up to 6
            words = t.split()
            tokens = [tok.cls_token_id] + [i + 2 for i in range(min(6, len(words)))] + [tok.sep_token_id]
            for j, v in enumerate(tokens[:L]):
                ids[i, j] = v
                att[i, j] = 1
        return {"input_ids": ids, "attention_mask": att}

    wrapped = wrap_tokenize_with_pruning(dummy_tokenize, tok, keep_ratio=0.5, strategy="length", min_tokens=2)

    texts = ["one two three four five six", "a b c"]
    out_q = wrapped(texts, is_query=True)
    out_d = wrapped(texts, is_query=False)
    # Query should be unchanged
    assert torch.equal(out_q["input_ids"], dummy_tokenize(texts, is_query=True)["input_ids"])
    # Document should have pad tokens introduced (some tokens removed)
    ids_d = out_d["input_ids"]
    att_d = out_d["attention_mask"]
    assert att_d.shape == ids_d.shape
    # ensure at least one padded token was introduced in non-query sample
    assert (ids_d == tok.pad_token_id).sum() >= 1


# ============================================================================
# Post-Encoding Embedding Pruning Tests
# ============================================================================


class TestEmbeddingPruningHierarchical:
    def test_basic_hierarchical_numpy(self):
        pytest.importorskip("scipy")
        # Create embeddings with some similarity structure
        emb = np.random.randn(20, 16).astype(np.float32)
        # Add protected tokens (make them distinct)
        emb[:2] = np.array([[1.0] * 16, [-1.0] * 16])
        
        pruned = prune_embeddings_hierarchical(emb, pool_factor=2.0, protected_tokens=2)
        
        # Should have ~50% reduction
        assert pruned.shape[0] < emb.shape[0]
        assert pruned.shape[0] >= 2  # at least protected tokens
        assert pruned.shape[1] == emb.shape[1]  # dim unchanged
        
        # Protected tokens should be identical
        assert np.allclose(pruned[0], emb[0])
        assert np.allclose(pruned[1], emb[1])
    
    def test_hierarchical_torch(self):
        if not _TORCH:
            pytest.skip("torch not available")
        pytest.importorskip("scipy")
        
        t = torch.randn(30, 8)
        pruned = prune_embeddings_hierarchical(t, pool_factor=3.0, protected_tokens=2)
        
        assert isinstance(pruned, torch.Tensor)
        assert pruned.shape[0] < t.shape[0]
        assert pruned.shape[1] == t.shape[1]
        assert pruned.device == t.device
    
    def test_too_few_tokens(self):
        pytest.importorskip("scipy")
        emb = np.random.randn(3, 4).astype(np.float32)
        pruned = prune_embeddings_hierarchical(emb, pool_factor=2.0, protected_tokens=2)
        # Should return unchanged when N <= protected + 1
        assert pruned.shape == emb.shape


class TestEmbeddingPruningAttention:
    def test_basic_attention_numpy(self):
        emb = np.random.randn(20, 16).astype(np.float32)
        # Create attention weights favoring certain tokens
        att = np.linspace(0, 1, 20)
        
        pruned = prune_embeddings_attention(emb, attention_weights=att, keep_ratio=0.5, protected_tokens=2)
        
        # Should keep ~50% of non-protected tokens
        expected_kept = 2 + max(1, int(np.ceil((20 - 2) * 0.5)))
        assert pruned.shape[0] <= expected_kept + 2  # some flexibility
        assert pruned.shape[1] == emb.shape[1]
    
    def test_attention_torch(self):
        if not _TORCH:
            pytest.skip("torch not available")
        
        t = torch.randn(25, 12)
        att = torch.rand(25)
        
        pruned = prune_embeddings_attention(t, attention_weights=att, keep_ratio=0.6, protected_tokens=2)
        
        assert isinstance(pruned, torch.Tensor)
        assert pruned.shape[0] < t.shape[0]
        assert pruned.shape[1] == t.shape[1]
    
    def test_no_attention_fallback(self):
        # Should use self-attention (cosine similarity to mean)
        emb = np.random.randn(15, 8).astype(np.float32)
        pruned = prune_embeddings_attention(emb, keep_ratio=0.5, protected_tokens=1)
        
        assert pruned.shape[0] < emb.shape[0]
        assert pruned.shape[0] >= 1  # at least protected


class TestEmbeddingPruningBatch:
    def test_batch_hierarchical_numpy(self):
        pytest.importorskip("scipy")
        batch = [
            np.random.randn(20, 8).astype(np.float32),
            np.random.randn(15, 8).astype(np.float32),
            np.random.randn(25, 8).astype(np.float32),
        ]
        doclens = [20, 15, 25]
        
        pruned_batch, new_lens = prune_embeddings_batch(
            batch, doclens, strategy="hierarchical", pool_factor=2.0, protected_tokens=2
        )
        
        assert len(pruned_batch) == len(batch)
        assert len(new_lens) == len(batch)
        for i, (pr, orig) in enumerate(zip(pruned_batch, batch)):
            assert pr.shape[0] < orig.shape[0]
            assert pr.shape[1] == orig.shape[1]
            assert new_lens[i] == pr.shape[0]
    
    def test_batch_attention_torch(self):
        if not _TORCH:
            pytest.skip("torch not available")
        
        batch = [torch.randn(10, 4), torch.randn(8, 4), torch.randn(12, 4)]
        doclens = [10, 8, 12]
        att_batch = [torch.rand(10), torch.rand(8), torch.rand(12)]
        
        pruned_batch, new_lens = prune_embeddings_batch(
            batch, doclens, strategy="attention", keep_ratio=0.6, 
            protected_tokens=1, attention_batch=att_batch
        )
        
        assert len(pruned_batch) == 3
        for i, pr in enumerate(pruned_batch):
            assert isinstance(pr, torch.Tensor)
            assert pr.shape[0] < batch[i].shape[0]
