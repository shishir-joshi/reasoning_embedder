# reasoning_embedder

Minimal quickstart for training reasoning‑aware embeddings.

Install (editable)
- pip install -e .

Prepare dataset (writes to data/prepared_reasonir_hq)
- reason-prepare

Dry‑run and auto‑select sensible lengths (no training)
- reason-train --auto_lengths --dry_run

Train with auto lengths (tiny smoke on CPU)
- reason-train --cpu --auto_lengths --sample --sample_size 64 --epochs 1 --batch_size 2 --mini_batch_size 1 --num_workers 0

Train with auto lengths on GPU (example)
- reason-train --auto_lengths --sample --sample_size 2048 --epochs 1 --batch_size 64 --mini_batch_size 1 --num_workers 0 --output_dir /content/drive/MyDrive/reasoning_embedder/model/qwen

Notes
- Defaults to base model: Qwen/Qwen3-Embedding-0.6B (override with --base_model)
- Data/artifacts live under data/ by default and are git‑ignored