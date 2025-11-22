from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class TrainingConfig:
    # Data and model
    dataset_path: str = "data/prepared_reasonir_hq"
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Output/run naming
    run_name: Optional[str] = None
    output_dir: Optional[str] = None

    # Training hyperparameters
    num_train_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 256
    mini_batch_size: int = 32

    # Model/tokenization specifics
    document_length: int = 8192
    query_length: int = 128
    skiplist_words: List[str] = field(default_factory=list)

    # Trainer/runtime
    save_steps: int = 500
    logging_steps: int = 1
    dataloader_num_workers: int = 8
    bf16: bool = True
    fp16: bool = False
    force_cpu: bool = False

    # Length auto-tuning
    auto_lengths: bool = False
    length_percentile: float = 0.95
    length_sample: int = 1000

    # Utility
    dry_run: bool = False

    # Data handling
    sample_size: Optional[int] = None
    do_sample: bool = False
    eval_holdout: float = 0.01
    seed: int = 42

    # Distributed/optimization features
    grad_accum_steps: int = 1
    grad_checkpoint: bool = False
    gather_across_devices: bool = True

    # PEFT / LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Optimizers
    optimizer_8bit: bool = False

    # Freezing
    freeze_base: bool = False
    train_last_n: Optional[int] = None
    # Token pruning (collator/tokenization stage)
    prune_tokens: bool = False
    prune_strategy: str = "length"
    prune_keep_ratio: float = 0.6
    prune_min_tokens: int = 16
    prune_seed: Optional[int] = None

    def finalize(self) -> "TrainingConfig":
        """Fill derived fields like run_name and output_dir if missing."""
        short = self.base_model.split("/")[-1]
        if not self.run_name:
            self.run_name = f"{short}-ReasonIR"
        if not self.output_dir:
            self.output_dir = os.path.join("data", "output", short, self.run_name)
        return self
