from __future__ import annotations

from typing import Optional
import torch
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import losses, models, utils

from .config import TrainingConfig
from reasoning_embedder.models.compat import ensure_tokenizer_padding


def build_model(cfg: TrainingConfig):
    kwargs = dict(
        model_name_or_path=cfg.base_model,
        document_length=cfg.document_length,
        query_length=cfg.query_length,
        skiplist_words=cfg.skiplist_words,
    )
    if cfg.force_cpu:
        kwargs["device"] = "cpu"
    model = models.ColBERT(**kwargs)
    ensure_tokenizer_padding(model)
    # Cap max sequence lengths to the backbone's supported context to avoid CUDA device-side asserts
    try:
        mod = model._first_module()
        tok = getattr(mod, "tokenizer", None)
        auto = getattr(mod, "auto_model", None)
        # Detect backbone limits
        limits = []
        if tok is not None and getattr(tok, "model_max_length", None):
            limits.append(int(tok.model_max_length))
        if auto is not None and hasattr(getattr(auto, "config", None), "max_position_embeddings") and auto.config.max_position_embeddings:
            limits.append(int(auto.config.max_position_embeddings))
        if limits:
            max_ctx = max(8, min(limits))
            # Apply conservative caps
            mod.max_seq_length = max_ctx
            if tok is not None:
                try:
                    tok.model_max_length = max_ctx
                except Exception:
                    pass
    except Exception:
        pass
    return model


def build_loss(cfg: TrainingConfig, model):
    return losses.CachedContrastive(
        model=model,
        mini_batch_size=cfg.mini_batch_size,
        gather_across_devices=True,
        temperature=1.0,
    )


def build_args(cfg: TrainingConfig) -> SentenceTransformerTrainingArguments:
    # Prefer bf16 if requested; otherwise enable fp16 when CUDA is available
    fp16 = (cfg.fp16 and torch.cuda.is_available()) and (not cfg.force_cpu)
    bf16 = cfg.bf16 and (not cfg.force_cpu)
    # Only enable MPS flag on macOS with available MPS and when not forcing CPU
    try:
        import platform
        use_mps = (
            platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and not cfg.force_cpu
        )
    except Exception:
        use_mps = False
    return SentenceTransformerTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        fp16=fp16,
        bf16=bf16,
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        dataloader_num_workers=cfg.dataloader_num_workers,
        no_cuda=cfg.force_cpu,
        use_mps_device=use_mps,
        use_cpu=cfg.force_cpu,
    )


def build_trainer(model, args: SentenceTransformerTrainingArguments, train_dataset, data_collator) -> SentenceTransformerTrainer:
    return SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=build_loss(args, model) if False else None,  # placeholder to keep signature similar
        data_collator=data_collator,
    )


def create_trainer(cfg: TrainingConfig, train_dataset) -> SentenceTransformerTrainer:
    model = build_model(cfg)
    loss = build_loss(cfg, model)
    args = build_args(cfg)
    collator = utils.ColBERTCollator(model.tokenize)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        data_collator=collator,
    )
    # Extra safeguard: if forcing CPU, move model to cpu explicitly
    if cfg.force_cpu:
        try:
            model.to("cpu")
        except Exception:
            pass
    return trainer
