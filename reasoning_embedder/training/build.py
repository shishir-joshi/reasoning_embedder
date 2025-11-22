from __future__ import annotations

from typing import Optional
import torch
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import losses, models, utils
from reasoning_embedder.optimization.token_pruning import generate_pruning_mask
from typing import Callable
try:
    import numpy as np
    import torch as _torch
except Exception:
    _torch = None
    np = None

from .config import TrainingConfig
from reasoning_embedder.models.compat import ensure_tokenizer_padding


def _maybe_enable_gradient_checkpointing(auto_model):
    try:
        if hasattr(auto_model, "gradient_checkpointing_enable"):
            auto_model.gradient_checkpointing_enable()
        elif hasattr(auto_model, "enable_input_require_grads"):
            auto_model.enable_input_require_grads()
    except Exception:
        pass


def _maybe_apply_lora(model, cfg: TrainingConfig):
    if not cfg.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
        mod = model._first_module()
        auto = getattr(mod, "auto_model", None)
        if auto is None:
            return model
        target = None
        if cfg.lora_target_modules:
            target = cfg.lora_target_modules
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target,
        )
        auto = get_peft_model(auto, peft_cfg)
        # Re-attach wrapped model
        setattr(mod, "auto_model", auto)
        return model
    except Exception:
        return model


def _maybe_freeze_layers(model, cfg: TrainingConfig):
    try:
        mod = model._first_module()
        auto = getattr(mod, "auto_model", None)
        if auto is None:
            return
        # Freeze all
        if cfg.freeze_base:
            for p in auto.parameters():
                p.requires_grad = False
        # Train last N layers (unfreeze them)
        if cfg.train_last_n is not None and cfg.train_last_n > 0:
            # Try common transformer stacks
            stacks = []
            for name in ("encoder.layer", "model.layers", "layers", "h"):
                cur = auto
                ok = True
                for part in name.split("."):
                    if hasattr(cur, part):
                        cur = getattr(cur, part)
                    else:
                        ok = False
                        break
                if ok and hasattr(cur, "__len__"):
                    stacks.append(cur)
                    break
            if stacks:
                layers = stacks[0]
                n = len(layers)
                k = max(0, min(n, cfg.train_last_n))
                for i, layer in enumerate(layers):
                    if i >= n - k:
                        for p in layer.parameters():
                            p.requires_grad = True
                    else:
                        for p in layer.parameters():
                            p.requires_grad = False
    except Exception:
        pass


def wrap_tokenize_with_pruning(
    tokenize_fn: Callable,
    tokenizer,
    keep_ratio: float,
    strategy: str,
    min_tokens: int,
    seed: Optional[int] = None,
) -> Callable:
    """Return a tokenize function that prunes non-query tokens by zeroing them.

    The returned callable keeps the same signature as the original `tokenize_fn`.
    It operates on the tokenized outputs (torch tensors or numpy arrays) and replaces
    pruned token positions with the tokenizer's pad_token_id and zeroes the attention mask.
    """
    pad_id = getattr(tokenizer, "pad_token_id", 0)

    def _wrapped(texts, is_query=False, pad_document=True):
        out = tokenize_fn(texts, is_query=is_query, pad_document=pad_document)
        if is_query:
            return out
        ids = out.get("input_ids")
        att = out.get("attention_mask")
        if ids is None or att is None:
            return out
        # operate on numpy or torch
        try:
            import torch as _t
        except Exception:
            _t = None

        if _t is not None and _t.is_tensor(ids):
            ids_t = ids
            att_t = att
            B, L = ids_t.shape
            for i in range(B):
                row_ids = ids_t[i]
                row_mask = att_t[i]
                active_idx = (row_mask != 0).nonzero(as_tuple=True)[0].tolist()
                N = len(active_idx)
                if N == 0:
                    continue
                preserve = [j for j in active_idx if row_ids[j].item() in (getattr(tokenizer, "cls_token_id", -1), getattr(tokenizer, "sep_token_id", -1))]
                try:
                    mask = generate_pruning_mask(
                        attention_weights=None,
                        keep_ratio=keep_ratio,
                        strategy=strategy,
                        N=N,
                        min_tokens=min_tokens,
                        preserve_indices=[active_idx.index(x) for x in preserve] if preserve else None,
                        seed=seed,
                    )
                except Exception:
                    mask = generate_pruning_mask(keep_ratio=keep_ratio, strategy="length", N=N, min_tokens=min_tokens)
                # mask might be numpy or torch tensor
                mask_list = mask.tolist() if hasattr(mask, "tolist") else list(mask)
                for idx_in_active, keep in enumerate(mask_list):
                    if not keep:
                        pos = active_idx[idx_in_active]
                        row_ids[pos] = pad_id
                        row_mask[pos] = 0
            out["input_ids"] = ids_t
            out["attention_mask"] = att_t
            return out

        # numpy path
        ids_np = ids
        att_np = att
        B, L = ids_np.shape
        for i in range(B):
            row_ids = ids_np[i]
            row_mask = att_np[i]
            active_idx = [j for j, v in enumerate(row_mask) if v]
            N = len(active_idx)
            if N == 0:
                continue
            preserve = [j for j in active_idx if row_ids[j] in (getattr(tokenizer, "cls_token_id", -1), getattr(tokenizer, "sep_token_id", -1))]
            try:
                mask = generate_pruning_mask(
                    attention_weights=None,
                    keep_ratio=keep_ratio,
                    strategy=strategy,
                    N=N,
                    min_tokens=min_tokens,
                    preserve_indices=[active_idx.index(x) for x in preserve] if preserve else None,
                    seed=seed,
                )
            except Exception:
                mask = generate_pruning_mask(keep_ratio=keep_ratio, strategy="length", N=N, min_tokens=min_tokens)
            mask_list = mask.tolist() if hasattr(mask, "tolist") else list(mask)
            for idx_in_active, keep in enumerate(mask_list):
                if not keep:
                    pos = active_idx[idx_in_active]
                    row_ids[pos] = pad_id
                    row_mask[pos] = 0
        out["input_ids"] = ids_np
        out["attention_mask"] = att_np
        return out

    return _wrapped


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
    # Optional features
    if cfg.grad_checkpoint:
        try:
            mod = model._first_module()
            auto = getattr(mod, "auto_model", None)
            if auto is not None:
                _maybe_enable_gradient_checkpointing(auto)
        except Exception:
            pass

    model = _maybe_apply_lora(model, cfg)
    _maybe_freeze_layers(model, cfg)
    return model


def build_loss(cfg: TrainingConfig, model):
    return losses.CachedContrastive(
        model=model,
        mini_batch_size=cfg.mini_batch_size,
        gather_across_devices=bool(cfg.gather_across_devices),
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
    optim = None
    if cfg.optimizer_8bit:
        optim = "adamw_bnb_8bit"
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
        gradient_accumulation_steps=max(1, int(cfg.grad_accum_steps)),
        optim=optim or "adamw_torch",
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
    # Wrap collator.tokenize with an optional pruning step to reduce token memory
    collator_tokenize = model.tokenize
    if cfg.prune_tokens:
        tok = getattr(model._first_module(), "tokenizer", None)
        collator_tokenize = wrap_tokenize_with_pruning(
            tokenize_fn=collator_tokenize,
            tokenizer=tok,
            keep_ratio=cfg.prune_keep_ratio,
            strategy=cfg.prune_strategy,
            min_tokens=cfg.prune_min_tokens,
            seed=cfg.prune_seed,
        )
    collator = utils.ColBERTCollator(collator_tokenize)

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
