# Changelog: feat/training-refactor

Date: 2025-10-11
Branch: `feat/training-refactor`
Target: merge into default branch (`main`)

## Summary

This branch delivers two main areas of work:

- Dataset preparation fixes for ReasonIR/BRIGHT to ensure correct `query`, `pos`, and `neg` structures.
- A modular, CLI-driven training pipeline for Reason-ModernColBERT using PyLate, including an explicit CPU fallback to avoid Apple Silicon MPS incompatibilities with CachedContrastive loss.

## Key Changes

1) Data processing

- File: `prepare_dataset.py`
  - Normalize `query` to string when provided as list.
  - Preserve `pos` as a two-element structure `[instruction, doc_text]` consistently.
  - Treat `neg` values as text when already provided, avoiding erroneous ID lookups.

2) Training refactor (modular design)

- New modules:
  - `training/config.py`: `TrainingConfig` dataclass with hyperparameters, device flags, and auto-derived fields (`run_name`, `output_dir`).
  - `training/data.py`: dataset loading, split prep, and filtering/preprocessing helpers.
  - `training/build.py`: builders for model, loss, trainer, and `SentenceTransformerTrainingArguments`.

- `train.py` rewritten as a clean CLI entry point:
  - Arguments: dataset/model paths, epochs, LR, batch sizes, sequence lengths, workers, `--bf16/--no-bf16`, `--fp16`, sampling controls, seed/holdout, and `--cpu`.
  - Uses PyLate `models.ColBERT` and `losses.CachedContrastive` with chunked embedding and in-batch negatives.
  - Saves artifacts to `output/<model_short>/<run_name>`.

3) Apple Silicon (MPS) compatibility

- PyLate’s `CachedContrastive` raises when MPS backend is available.
- Added `--cpu` flag and config `force_cpu`:
  - TrainingArguments: `no_cuda`, `use_cpu`, and `use_mps_device` set to force CPU.
  - Model builder passes `device='cpu'` to ColBERT when forced.
  - `train.py` defensively disables MPS via `torch.backends.mps.is_available = lambda: False` and sets `PYTORCH_ENABLE_MPS_FALLBACK=1` when `--cpu` is used.

## CLI Examples

Create venv (Python 3.11), install deps, and run a smoke test on CPU:

```bash
python3.11 -m venv .venv311
./.venv311/bin/python -m pip install --upgrade pip setuptools wheel
./.venv311/bin/pip install "numpy<2" torch transformers "sentence-transformers>=3.0.0" datasets accelerate pylate

# sample run (small, CPU)
./.venv311/bin/python train.py \
  --sample --sample_size 50 \
  --epochs 1 \
  --batch_size 8 \
  --mini_batch_size 4 \
  --num_workers 0 \
  --no-bf16 \
  --cpu
```

## Behavioral Notes

- Output folder: `output/<model_short>/<model_short>-ReasonIR` by default.
- With `--cpu`, MPS is disabled to prevent PyLate loss errors on Apple Silicon.
- For larger runs, increase `batch_size` and set `mini_batch_size` to fit memory.

## Commit Summary (vs origin/main)

```
5d5376e notebook
7936db0 workaround(mps): when --cpu, monkeypatch torch.backends.mps.is_available to False to satisfy PyLate loss
ca19300 train(args): also set use_cpu when --cpu is passed to fully force CPU in TrainingArguments
807039b fix(train): force device='cpu' in ColBERT when --cpu is set
07de2f0 chore(train): add --cpu flag; force CPU to avoid MPS incompatibility for pylate loss; expose no_cuda/use_mps_device
25d0657 refactor(training): modularize training per gist (config/data/build), switch to CachedContrastive + ColBERT lengths; add CLI
0f6448a fix: normalize query; correct pos/neg handling to text shape [instruction, doc_text] in prepare_dataset.py
```

## Known Issues / Follow-ups

- If running without `--cpu` on Apple Silicon, PyLate `CachedContrastive` may fail when MPS is available. Use `--cpu` until upstream adds MPS support.
- Consider adding a requirements file and CI job for smoke tests.
