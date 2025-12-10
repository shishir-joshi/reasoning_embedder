import pytest
from datasets import Dataset, DatasetDict

from reasoning_embedder.training.data import preprocess_flatten_entry, prepare_splits
from reasoning_embedder.training.config import TrainingConfig
from reasoning_embedder.training.build import wrap_doc_with_embedding_pruning


def _make_entry(idx: int) -> dict:
    return {
        "query": ["reason", "ir", f"q{idx}"],
        "pos": [[f"inst {idx}", f"doc {idx}" ]],
        "neg": [[f"neg-inst {idx}", f"neg-doc {idx}"]],
    }


def test_preprocess_flatten_entry_normalizes_strings():
    entry = {
        "query": ["what", "is", "reasoning"],
        "pos": [["instruct", "document text"]],
        "neg": [["counter", "negative text"]],
    }

    out = preprocess_flatten_entry(entry)

    assert out["query"] == "what is reasoning"
    assert out["pos"] == "instruct document text"
    assert out["neg"] == "counter negative text"


def test_prepare_splits_filters_invalid_examples():
    dataset = Dataset.from_list(
        [
            _make_entry(0),
            {"query": ["bad"], "pos": [], "neg": []},
            _make_entry(1),
        ]
    )
    ds = DatasetDict({"train": dataset})

    train_ds, eval_ds = prepare_splits(ds, do_sample=False, sample_size=None, seed=7, eval_holdout=0.5)

    total_clean = len(train_ds) + (len(eval_ds) if eval_ds is not None else 0)
    assert total_clean == 2
    assert all(isinstance(row["pos"], str) and isinstance(row["neg"], str) for row in train_ds)


def test_prepare_splits_sampling_respects_sample_size():
    dataset = Dataset.from_list([_make_entry(i) for i in range(10)])
    ds = DatasetDict({"train": dataset})

    train_ds, eval_ds = prepare_splits(ds, do_sample=True, sample_size=4, seed=13, eval_holdout=0.2)

    assert 0 < len(train_ds) <= 4
    assert eval_ds is not None and 0 < len(eval_ds) <= 4


def test_wrap_doc_embedding_pruning_attention_reduces_tokens():
    torch = pytest.importorskip("torch")
    cfg = TrainingConfig(
        prune_embeddings=True,
        embedding_prune_strategy="attention",
        embedding_keep_ratio=0.5,
        protected_tokens=1,
    )

    base = torch.arange(24, dtype=torch.float32).view(6, 4)

    def dummy_doc(input_ids, attention_mask, keep_dims=True):
        batch = input_ids.shape[0]
        if keep_dims is False:
            return [base.clone() for _ in range(batch)]
        return torch.stack([base.clone() for _ in range(batch)], dim=0)

    wrapped = wrap_doc_with_embedding_pruning(dummy_doc, cfg)

    input_ids = torch.ones((2, 6), dtype=torch.long)
    attention_mask = torch.ones((2, 6), dtype=torch.long)

    original = dummy_doc(input_ids, attention_mask, keep_dims=False)
    pruned = wrapped(input_ids, attention_mask, keep_dims=False)

    assert len(original) == len(pruned)
    assert pruned[0].shape[1] == original[0].shape[1]
    assert pruned[0].shape[0] < original[0].shape[0]

    keep_dims_out = wrapped(input_ids, attention_mask, keep_dims=True)
    assert keep_dims_out.shape == (2,) + original[0].shape
