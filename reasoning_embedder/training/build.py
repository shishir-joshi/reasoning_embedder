from __future__ import annotations

from typing import Optional
import torch
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import losses, models, utils

from .config import TrainingConfig


def _ensure_padding_token(model) -> None:
    """Ensure tokenizer has a pad_token; add or alias if missing and resize embeddings.

    Works for arbitrary HF models wrapped by Sentence-Transformers/PyLate.
    """
    try:
        # sentence-transformers first module holds the tokenizer and auto_model
        mod = model._first_module()  # type: ignore[attr-defined]
        tok = getattr(mod, "tokenizer", None)
        auto = getattr(mod, "auto_model", None)
        if tok is None:
            return
        if tok.pad_token is None:
            if getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                if auto is not None and hasattr(auto, "resize_token_embeddings"):
                    auto.resize_token_embeddings(len(tok))
        # Some tokenizers use pad_token but pad_token_id is None; fix id
        if getattr(tok, "pad_token_id", None) is None and tok.pad_token is not None:
            tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
            if auto is not None and hasattr(auto, "resize_token_embeddings"):
                auto.resize_token_embeddings(len(tok))
    except Exception:
        # Be conservative: do not fail training if we can't introspect
        pass


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
    _ensure_padding_token(model)
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
    fp16 = cfg.fp16 and torch.cuda.is_available()
    bf16 = cfg.bf16
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
        use_mps_device=not cfg.force_cpu,
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
    return trainer
