# reasoning_embedder

<a target="_blank" href="https://colab.research.google.com/github/shishir-joshi/reasoning_embedder/blob/main/notebooks/Eval-Baseline.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Train reasoning‑aware dense retrievers with a clean CLI.

## Install
- Create/activate a Python 3.11 venv (recommended)
- pip install -e .

## Data
- Prepare ReasonIR HQ against BRIGHT docs (writes to `data/prepared_reasonir_hq`):
  - reason-prepare
- All datasets and outputs live under `data/` (git‑ignored).

### Dataset preparation rationale
The preparation step standardizes ReasonIR HQ examples against the BRIGHT documents to yield a single, clean schema that training expects:
- Normalize query → always a single string (lists are joined).
- Positives (pos) → each item is `[instruction, document_text]`. If HQ stores a BRIGHT document id, we resolve it to raw text using the BRIGHT corpus.
- Negatives (neg) → already text in HQ; we preserve as `[instruction, document_text]`. If a value looks like an id and exists in BRIGHT, we prefer mapped text.
- Shape consistency → every pos/neg pair is exactly two elements to keep collation predictable.
- Graceful filtering → malformed pairs are skipped with debug logs rather than crashing.

Why this matters:
- Ensures the trainer always receives comparable (query, positive, negative) triplets with text on both sides (no late id lookups during training).
- Enables accurate token length statistics for `--auto_lengths`, since all text is resolved before training.
- Decouples data resolution from training for repeatability and easier debugging; the prepared dataset is a self‑contained artifact.

## Quick starts
- Dry‑run + auto lengths (no training):
  - reason-train --auto_lengths --dry_run

- Tiny CPU smoke:
  - reason-train --cpu --auto_lengths --sample --sample_size 64 --epochs 1 --batch_size 2 --mini_batch_size 1 --num_workers 0

- GPU example (your request):
  - reason-train --auto_lengths --sample --sample_size 2048 --epochs 1 --batch_size 64 --mini_batch_size 1 --num_workers 0 --output_dir /content/drive/MyDrive/reasoning_embedder/model/qwen

## Model selection
- Default: `Qwen/Qwen3-Embedding-0.6B`
- Override via `--base_model` (e.g., `sentence-transformers/all-mpnet-base-v2`).

## Auto lengths and memory
- `--auto_lengths` samples the dataset and picks minimal doc/query lengths at a percentile (default 0.95), capped to the model context.
- `--dry_run` prints the derived lengths and a rough per‑step activation memory estimate (no training).

## Token & Embedding Pruning

### Input-level pruning (tokenization stage)
Reduces input token counts but model still allocates full embedding tensors:

- `--prune_tokens`: enable token-level pruning during tokenization/collation
- `--prune_strategy`: one of `length`, `random`, `attention`, `threshold`, `combined` (default: `length`)
- `--prune_keep_ratio`: fraction of tokens to keep (default: 0.6)
- `--prune_min_tokens`: absolute minimum tokens to preserve per sequence (default: 16)

### Output-level pruning (post-encoding stage) **NEW**
Reduces actual output embedding size for true memory savings during forward pass:

- `--prune_embeddings`: enable post-encoding embedding pruning (reduces memory)
- `--embedding_prune_strategy`: `hierarchical` (semantic clustering, recommended) or `attention` (importance-based, faster)
- `--pool_factor`: for hierarchical pooling (2.0 = 50% reduction, 3.0 = 66% reduction)
- `--embedding_keep_ratio`: for attention-based pruning (0.6 = keep 60%)
- `--protected_tokens`: preserve first N tokens like [CLS], [D] (default: 2)

**Recommendation:** Use `--prune_embeddings --embedding_prune_strategy hierarchical --pool_factor 2.0` for 40-60% memory reduction with minimal quality loss.

## CPU/GPU tips
- Force CPU: add `--cpu` (disables CUDA/MPS).
- For low VRAM GPUs (≈15GB), try: smaller lengths/batches, `--fp16`, embedding pruning, and gradient‑friendly settings. Example:
  - `reason-train --auto_lengths --prune_embeddings --pool_factor 2.0 --epochs 1 --batch_size 2 --mini_batch_size 1 --fp16 --num_workers 0`

## Outputs
- Artifacts under `data/output/<model>/<run>/` (configurable via `--output_dir`).

## Troubleshooting
- Tokenizer pad token errors are handled automatically.
- On Mac, MPS is used only if available; use `--cpu` to disable.
- For CUDA OOM, reduce `--document_length/--query_length`, `--batch_size`, use `--fp16`, or enable `--prune_embeddings`.

## Agent coordination
For handoffs between LLM assistants and to track session context, see:

- agents/AGENT_LOG.md — the running coordination log
- agents/TEMPLATE.md — template for adding new coordination entries

When starting a new session, append a new entry to `agents/AGENT_LOG.md` using the template.