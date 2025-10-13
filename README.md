# reasoning_embedder

Train reasoning‑aware dense retrievers with a clean CLI.

## Install
- Create/activate a Python 3.11 venv (recommended)
- pip install -e .

## Data
- Prepare ReasonIR HQ against BRIGHT docs (writes to `data/prepared_reasonir_hq`):
  - reason-prepare
- All datasets and outputs live under `data/` (git‑ignored).

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

## CPU/GPU tips
- Force CPU: add `--cpu` (disables CUDA/MPS).
- For low VRAM GPUs (≈15GB), try: smaller lengths/batches, `--fp16`, and gradient‑friendly settings. Example:
  - reason-train --auto_lengths --epochs 1 --batch_size 2 --mini_batch_size 1 --fp16 --num_workers 0

## Outputs
- Artifacts under `data/output/<model>/<run>/` (configurable via `--output_dir`).

## Troubleshooting
- Tokenizer pad token errors are handled automatically.
- On Mac, MPS is used only if available; use `--cpu` to disable.
- For CUDA OOM, reduce `--document_length/--query_length`, `--batch_size`, or use `--fp16`.