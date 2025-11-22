"""Small memory benchmark for token pruning on prepared_reasonir_hq.

This script loads a small sample of documents from the prepared ReasonIR HQ
dataset, encodes them at the token level using a small transformer, and
measures the in-memory size (bytes) of token embeddings before and after
pruning using `prune_colbert_embeddings`.

Usage:
    python examples/token_pruning_benchmark.py --dataset_path data/prepared_reasonir_hq --sample_size 10

Notes:
- This is a lightweight benchmark (small sample) to estimate memory savings.
- It uses a small transformer for token-level embeddings to keep runtime
  short and avoid OOMs. For production, use your ColBERT token embeddings.
"""
from __future__ import annotations

import argparse
import os
import math
import numpy as np
from typing import List

import datasets
from transformers import AutoTokenizer, AutoModel
import torch

from reasoning_embedder.optimization.token_pruning import prune_colbert_embeddings


def bytes_for_numpy(arr: np.ndarray) -> int:
    return arr.nbytes


def bytes_for_torch(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


def encode_tokens(text: str, tok, model, device='cpu'):
    # Token-level encoding using transformers AutoModel last_hidden_state
    inputs = tok(text, return_tensors='pt', truncation=True, padding=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # out.last_hidden_state shape = (1, L, hidden)
    emb = out.last_hidden_state[0].cpu()
    return emb


def sample_docs_from_prepared(dataset_path: str, sample_size: int) -> List[str]:
    # Try standard HF datasets load first
    try:
        ds = datasets.load_from_disk(dataset_path)
        if isinstance(ds, datasets.DatasetDict):
            if 'train' in ds:
                dset = ds['train']
            else:
                dset = list(ds.values())[0]
        else:
            dset = ds
        docs = []
        for ex in dset:
            pos = ex['pos'] if 'pos' in ex else []
            if isinstance(pos, list) and len(pos) > 0:
                item = pos[0]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    docs.append(item[1])
            if len(docs) >= sample_size:
                break
        return docs[:sample_size]
    except Exception:
        # Fallback: read arrow file with pyarrow directly
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            train_dir = os.path.join(dataset_path, 'train')
            # Prefer data-*.arrow or data-00000-of-00001.arrow
            arrow_files = [f for f in os.listdir(train_dir) if f.endswith('.arrow')]
            if not arrow_files:
                return []
            arrow_file = os.path.join(train_dir, sorted(arrow_files)[0])
            with ipc.open_file(arrow_file) as reader:
                table = reader.read_all()
            pos_col = table.column('pos').to_pylist()
            docs = []
            for pos in pos_col:
                if isinstance(pos, list) and len(pos) > 0:
                    item = pos[0]
                    # item should be [instruction, doc_text]
                    if isinstance(item, list) and len(item) >= 2:
                        docs.append(item[1])
                if len(docs) >= sample_size:
                    break
            return docs[:sample_size]
        except Exception:
            # Second fallback: try to fetch the source ReasonIR dataset from HF remote
            try:
                ds = datasets.load_dataset("reasonir/reasonir-data", "hq")
                dset = ds['train'] if 'train' in ds else list(ds.values())[0]
                docs = []
                for ex in dset:
                    pos = ex['pos'] if 'pos' in ex else []
                    if isinstance(pos, list) and len(pos) > 0:
                        item = pos[0]
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            docs.append(item[1])
                    if len(docs) >= sample_size:
                        break
                return docs[:sample_size]
            except Exception:
                return []


def memory_benchmark(dataset_path: str, sample_size: int = 10, keep_ratio: float = 0.5, strategy: str = 'attention'):
    device = 'cpu'
    # Use small transformer for token-level embeddings
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    
    docs = sample_docs_from_prepared(dataset_path, sample_size)
    if not docs:
        print(f"No docs found under {dataset_path} - nothing to benchmark")
        return

    total_before = 0
    total_after = 0
    total_tokens_before = 0
    total_tokens_after = 0

    for text in docs:
        token_emb = encode_tokens(text, tok, model, device=device)
        # token_emb shape (L, D)
        before_bytes = bytes_for_torch(token_emb)
        total_before += before_bytes
        total_tokens_before += token_emb.shape[0]
        # derive attention proxy from token norms
        att = token_emb.norm(dim=1).cpu().numpy()
        pruned, mask = prune_colbert_embeddings(token_emb, attention_weights=att, keep_ratio=keep_ratio, strategy=strategy, min_tokens=1, return_mask=True)
        if isinstance(pruned, torch.Tensor):
            after_bytes = bytes_for_torch(pruned)
            after_tokens = pruned.shape[0]
        else:
            after_bytes = bytes_for_numpy(pruned)
            after_tokens = pruned.shape[0]
        total_after += int(after_bytes)
        total_tokens_after += int(after_tokens)

    print("Token pruning benchmark results")
    print("---------------------------------")
    print(f"Docs benchmarked: {len(docs)}")
    print(f"Total tokens before: {total_tokens_before}")
    print(f"Total tokens after : {total_tokens_after}")
    print(f"Memory before (bytes): {total_before:,}")
    print(f"Memory after  (bytes): {total_after:,}")
    if total_before > 0:
        print(f"Memory reduction: {100.0 * (1 - total_after / total_before):.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', default='data/prepared_reasonir_hq')
    p.add_argument('--sample_size', type=int, default=10)
    p.add_argument('--keep_ratio', type=float, default=0.5)
    p.add_argument('--strategy', type=str, default='attention')
    args = p.parse_args()
    if not os.path.isdir(args.dataset_path):
        print('Dataset not found. Run reason-prepare first or set dataset_path.')
        return
    memory_benchmark(args.dataset_path, sample_size=args.sample_size, keep_ratio=args.keep_ratio, strategy=args.strategy)


if __name__ == '__main__':
    main()
