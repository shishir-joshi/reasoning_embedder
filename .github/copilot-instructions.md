# Copilot Instructions for Reasoning Embedder

## Project Overview

This repository trains reasoning-aware dense retrievers using ColBERT-style late interaction models. The codebase provides a clean CLI wrapper around PyLate/ColBERT for training embedders on the ReasonIR dataset against BRIGHT documents.

## Architecture & Key Components

### Data Pipeline
- **Dataset preparation**: `reason-prepare` standardizes ReasonIR HQ examples against BRIGHT docs into `data/prepared_reasonir_hq/`
- **Schema**: Each training example becomes `(query, [instruction, document_text])` triplets for positives and negatives
- **ID resolution**: BRIGHT document IDs are resolved to full text during preparation, not training
- **All datasets/outputs**: Live under `data/` (git-ignored)

### Training Pipeline
- **Entry point**: `reason-train` CLI with extensive configuration options
- **Model architecture**: ColBERT via PyLate with configurable document/query lengths
- **Configuration**: `TrainingConfig` dataclass in `reasoning_embedder/training/config.py`
- **Auto-sizing**: `--auto_lengths` samples dataset to derive optimal sequence lengths
- **Memory estimation**: `--dry_run` shows estimated activation memory before training

### Key Modules
```
reasoning_embedder/
├── cli/train.py           # Main CLI with 25+ training flags
├── data/prepare.py        # Dataset preparation and ID resolution
├── training/
│   ├── config.py         # TrainingConfig with PEFT/LoRA/freezing options
│   ├── build.py          # Model/trainer/optimizer construction
│   └── data.py           # Dataset loading and splitting
└── models/compat.py      # Tokenizer padding compatibility
```

## Development Workflows

### Essential Commands
```bash
# Install (Python 3.11+ recommended)
pip install -e .

# Prepare data (required first step)
reason-prepare

# Quick smoke test
reason-train --cpu --auto_lengths --sample --sample_size 64 --epochs 1 --batch_size 2

# Memory estimation without training
reason-train --auto_lengths --dry_run

# Production training with LoRA
reason-train --auto_lengths --lora --lora_r 16 --batch_size 64 --epochs 3
```

### Testing & Validation
- **Notebooks**: `notebooks/` contain evaluation and exploration code
- **BRIGHT evaluation**: Use `Eval-Baseline.ipynb` for retrieval benchmarking
- **Dataset exploration**: `explore_reasonir_dataset.ipynb` and `explore_prepared_dataset.ipynb`
- **Evaluation outputs**: Results saved to `data/bright_evaluation_results/`

### BRIGHT Benchmarking Process
- **Reference methodology**: Based on [NohTow's gist](https://gist.github.com/NohTow/3f27d2816b92d5c76f0e63aa7757cf4b#start-of-content)
- **CPU-safe baseline**: Use Sentence Transformers with cosine similarity for reliable evaluation without CUDA dependencies
- **Memory considerations**: Keep batch sizes small (8-16) to avoid buffer errors on macOS/ARM
- **Evaluation flow**: Load BRIGHT test split → embed queries/documents → compute retrieval metrics → save results

## Project-Specific Patterns

### Configuration Management
- All training parameters flow through `TrainingConfig.finalize()` which sets derived fields
- CLI args map directly to config fields in `parse_args()`
- Output directories auto-generated: `data/output/<model>/<run_name>/`

### Memory & Device Handling
- **CPU fallback**: `--cpu` disables CUDA/MPS and sets environment variables
- **Auto memory**: `--auto_lengths` + `--dry_run` estimates activation memory before training
- **Mixed precision**: `--bf16` (default) or `--fp16` for memory efficiency
- **Gradient features**: `--grad_checkpoint`, `--grad_accum_steps` for large models

### Advanced Training Features
- **LoRA/PEFT**: `--lora` with configurable rank/alpha/dropout/target_modules
- **Layer freezing**: `--freeze_base` or `--train_last_n N` for selective training
- **8-bit optimization**: `--optimizer_8bit` for memory efficiency
- **Distributed**: `--gather_across_devices` for cross-device contrastive loss

### Data Flow Conventions
1. **Preparation phase**: ReasonIR HQ + BRIGHT → standardized triplets in `data/prepared_reasonir_hq/`
2. **Training phase**: Load prepared dataset → auto-size → split → train
3. **Output phase**: Artifacts saved to `data/output/<model>/<run>/` with final model in `final/`

## External Dependencies

### Core Libraries
- **PyLate**: ColBERT implementation and training utilities
- **Sentence Transformers**: Base model loading and training framework
- **Transformers/Accelerate**: Model backends and distributed training
- **Datasets (HF)**: Data loading from ReasonIR and BRIGHT

### Dataset Sources
- **ReasonIR HQ**: `load_dataset("reasonir/reasonir-data", "hq")`
- **BRIGHT**: `load_dataset("xlangai/BRIGHT", "documents")` or `load_dataset("allenai/bright")`
- **Fallback fetching**: Use `hf_hub_download` for problematic assets

### Web-Based Data Fetching Patterns
```python
# Robust dataset loading with retries
import time
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

def load_with_retries(repo_id, split, retries=5, base_delay=1.0):
    for attempt in range(1, retries+1):
        try:
            return load_dataset(repo_id, split=split)
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(base_delay * (2 ** (attempt-1)))

# Bulk snapshot when high-level load fails
def fetch_dataset_snapshot(repo_id):
    root = snapshot_download(repo_id=repo_id)
    files = [os.path.join(root, f) for f in os.listdir(root)]
    return load_dataset("json", data_files={"test": files})

# CPU-friendly retrieval baseline
def cpu_safe_retrieval(queries, documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    q_embs = model.encode(queries, batch_size=8, convert_to_numpy=True)
    d_embs = model.encode(documents, batch_size=8, convert_to_numpy=True)
    # Cosine similarity with normalization
    qn = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    dn = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-9)
    return qn @ dn.T
```

## Common Gotchas

### Data Preparation
- Always run `reason-prepare` before training - it's not optional
- Malformed pos/neg pairs are skipped with debug logs (check for low data counts)
- Document ID resolution happens during prep, not training (prevents runtime lookups)

### Training Configuration
- Default model `Qwen/Qwen3-Embedding-0.6B` - override with `--base_model`
- `--auto_lengths` is crucial for new datasets - prevents OOM and context overflow
- CPU training requires `--cpu` flag + smaller batch sizes for stability

### Memory Management
- Use `--dry_run` first to estimate memory requirements
- For CUDA OOM: reduce `--batch_size`, `--document_length`, or enable `--fp16`
- LoRA training (`--lora`) significantly reduces memory for large models

### BRIGHT Evaluation Issues
- **Compiled extension failures**: PyLate/PLAID may fail on macOS/ARM - use CPU fallback
- **Large memory allocations**: Buffer errors common on ARM - keep batch sizes ≤16
- **Asset loading failures**: When `load_dataset` fails, use `snapshot_download` + local loading
- **Network reliability**: HF downloads can be flaky - implement retry logic with exponential backoff

### CPU Fallback Strategy
When CUDA/compiled dependencies unavailable, use Sentence Transformers baseline:
- Set `HF_TOKEN` for gated resources, respect `~/.cache/huggingface` for caching
- Use small batch sizes (8-16) and cosine similarity for retrieval
- Avoid PyLate/PLAID compiled extensions that require CUDA

When implementing new features, follow the established pattern of CLI arg → TrainingConfig field → build.py implementation. Always test with `--dry_run` first and consider CPU compatibility for broader usability.