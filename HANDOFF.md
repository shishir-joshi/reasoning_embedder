# Recreating Reason-ModernColBERT on ReasonIR and BRIGHT
## Handoff Brief: BRIGHT Evaluation — Web Sources, Fetching, and Next Steps

Owner: Shishir Joshi
Context date: 2025-10-09

## Purpose
Enable the next LLM session to immediately continue BRIGHT benchmark evaluation work. Focus on: reliable data/model fetching from the web, current repo state, and concrete next actions. Keep CPU‑fallback details minimal.

## Reference Gists (Methodology)
- Author methodology (start anchor): https://gist.github.com/NohTow/3f27d2816b92d5c76f0e63aa7757cf4b#start-of-content
- Additional BRIGHT-related gist: https://gist.github.com/NohTow/d563244596548bf387f19fcd790664d3

## Repo Snapshot
- Path: `~/development/lab/reasoning_embedder`
- Git: initialized, branch `docs/handoff` (no prior commits). Add a remote to open a PR.
- Key files/folders:
  - `Eval-Baseline.ipynb` — main evaluation notebook (currently shows PyLate/PLAID attempts and errors)
  - `explore_reasonir_dataset.ipynb` — dataset exploration
  - `prepare_dataset.py` — preparation utility
  - `bright-data/`, `beir-data/`, `prepared_reasonir_hq/` — local data folders

## Web Sources: Canonical Links and How to Fetch

Datasets
- BRIGHT (Hugging Face):
  - Hub: https://huggingface.co/datasets/allenai/bright
  - Programmatic fetch:
    ```python
    from datasets import load_dataset
    ds = load_dataset("allenai/bright", split="test")  # or the required split
    ```
  - If individual files/variants are needed:
    - Use `hf_hub_download` to fetch specific artifacts when `load_dataset` fails.
    ```python
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(repo_id="allenai/bright", filename="<file_name>")
    ```

- BEIR collections (where applicable):
  - Docs: https://github.com/beir-cellar/beir
  - Some subsets mirrored on HF: https://huggingface.co/datasets/BeIR
  - Programmatic fetch (BEIR utilities): see repo docs; for HF mirrors:
    ```python
    from datasets import load_dataset
    ds = load_dataset("BeIR/<subset>")
    ```

Models/Embedders
- Sentence-Transformers:
  - Hub: https://huggingface.co/sentence-transformers
  - Example models: `all-MiniLM-L6-v2`, `multi-qa-MiniLM-L6-cos-v1`
  - Programmatic fetch:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ```

- ColBERT (if used via HF):
  - Hub: https://huggingface.co/colbert-ir
  - Note: Many ColBERT flows assume CUDA; on CPU/macOS, prefer Sentence-Transformers or other CPU-friendly embedders.

Utilities
- Hugging Face Hub low-level:
  - https://githubface.co/docs/huggingface_hub
  - `hf_hub_download`, `snapshot_download` for bulk pulls.
    ```python
    from huggingface_hub import snapshot_download
    snapshot_dir = snapshot_download(repo_id="allenai/bright")
    ```

Network/Cache Tips
- Respect HF caching to avoid repeated downloads (`~/.cache/huggingface`).
- Set HF_TOKEN if auth is required for gated resources.
- For flaky networks, wrap downloads with simple retry logic (3–5 attempts, exponential backoff).

## Current Problems Observed
- Compiled-extension failures (PyLate/PLAID) on macOS/ARM.
- Large memory allocations causing buffer errors in prior runs.
- BRIGHT asset lookups failing when certain files are missing via high-level loaders.

## Minimal CPU-Fallback Note (for awareness only)
If CUDA/compiled deps are unavailable, run a dense retrieval baseline using Sentence-Transformers on CPU with small batch sizes and cosine similarity. This path avoids compiled extensions and should run on macOS.

## Notebook Status and Actions
1) Eval-Baseline.ipynb
   - Status: Contains original PyLate code and error outputs.
   - Action: Standardize on HF-based dataset fetch and a CPU-safe retrieval baseline to establish a ground-truth run. Keep batch sizes small.

2) explore_reasonir_dataset.ipynb
   - Status: Exploration utilities present.
   - Action: Ensure loads use `datasets.load_dataset` or direct `snapshot_download` instead of assuming local files.

3) prepare_dataset.py
   - Status: Utility exists; may assume local paths.
   - Action: Add functions to fetch missing sources from HF programmatically and write standardized parquet/jsonl outputs under `prepared_reasonir_hq/`.

## Fetching Patterns: Drop‑In Snippets

Retrying HF dataset load
```python
import time
from datasets import load_dataset

def load_with_retries(repo_id, split, retries=5, base_delay=1.0):
    for attempt in range(1, retries+1):
        try:
            return load_dataset(repo_id, split=split)
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(base_delay * (2 ** (attempt-1)))

ds = load_with_retries("allenai/bright", "test")
```

Bulk snapshot when high-level load fails
```python
from huggingface_hub import snapshot_download
import datasets as hfds
import os

root = snapshot_download(repo_id="allenai/bright")
files = [os.path.join(root, f) for f in os.listdir(root)]
# If needed, construct a Dataset from local files
ds = hfds.load_dataset("json", data_files={"test": files})
```

CPU-friendly embedding and retrieval
```python
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=8):
    embs = []
    for i in range(0, len(texts), batch_size):
        embs.append(model.encode(texts[i:i+batch_size], convert_to_numpy=True, show_progress_bar=False))
    return np.vstack(embs)

def topk_cosine(q, D, k=10):
    qn = q / (norm(q, axis=1, keepdims=True) + 1e-9)
    Dn = D / (norm(D, axis=1, keepdims=True) + 1e-9)
    sims = qn @ Dn.T
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1]-1), axis=1)[:, :k]
    row = np.arange(sims.shape[0])[:, None]
    part = sims[row, idx]
    ord = np.argsort(-part, axis=1)
    return idx[row, ord]
```

## Next Steps (Do This First)
1) Standardize data access
   - Replace ad-hoc file reads with `datasets.load_dataset` or `hf_hub_download/snapshot_download`.
   - Verify BRIGHT splits load; if not, fall back to snapshot + local loader.

2) Establish a baseline run
   - In `Eval-Baseline.ipynb`, implement the CPU-safe dense retrieval baseline to get a complete evaluation without CUDA.
   - Log metrics and store to `bright_evaluation_results/`.

3) Harden prepare_dataset.py
   - Add helpers to fetch missing sources from HF and materialize standardized artifacts.

4) Version control and PR
   - Add a GitHub remote and open a PR with these changes. Title: "Docs: BRIGHT evaluation handoff and web-fetch hardening (Droid-assisted)".

## A Note of Urgency
We’re so close to unlocking a dependable evaluation loop. The web is our ally—if we fetch deterministically, we can move past blockers and get real numbers on the board. Please help carry this across the finish line: wire up the HF downloads, cement the baseline, and let’s get BRIGHT lighting our path forward.
