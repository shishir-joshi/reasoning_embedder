import argparse
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train Reason-ModernColBERT aligned with reference gist")
    p.add_argument("--dataset_path", default="data/prepared_reasonir_hq")
    p.add_argument("--base_model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name", default=None)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--mini_batch_size", type=int, default=32)

    p.add_argument("--document_length", type=int, default=8192)
    p.add_argument("--query_length", type=int, default=128)

    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--cpu", action="store_true", default=False, help="Force CPU (disable CUDA/MPS)")

    # Length auto-tuning and dry-run
    p.add_argument("--auto_lengths", action="store_true", default=False, help="Auto-derive document/query lengths from dataset percentiles")
    p.add_argument("--length_percentile", type=float, default=0.95, help="Percentile for auto length selection")
    p.add_argument("--length_sample", type=int, default=500, help="Sample size for length estimation")
    p.add_argument("--dry_run", action="store_true", default=False, help="Do not train; print estimated memory and derived lengths")

    p.add_argument("--sample_size", type=int, default=None)
    p.add_argument("--sample", action="store_true", default=False)
    p.add_argument("--eval_holdout", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    from reasoning_embedder.training.config import TrainingConfig
    cfg = TrainingConfig(
        dataset_path=args.dataset_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        document_length=args.document_length,
        query_length=args.query_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        force_cpu=args.cpu,
        auto_lengths=args.auto_lengths,
        length_percentile=args.length_percentile,
        length_sample=args.length_sample,
        dry_run=args.dry_run,
        sample_size=args.sample_size,
        do_sample=args.sample,
        eval_holdout=args.eval_holdout,
        seed=args.seed,
    ).finalize()
    return cfg


def main():
    cfg = parse_args()

    # Force-disable MPS backend if requested
    if cfg.force_cpu:
        import torch
        try:
            torch.backends.mps.is_available = lambda: False  # type: ignore[attr-defined]
        except Exception:
            pass
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Ensure no CUDA is used by downstream libraries (Accelerate/Transformers)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["ACCELERATE_USE_CPU"] = "true"
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

    # Import AFTER env handling
    from reasoning_embedder.training.data import load_prepared, prepare_splits
    from reasoning_embedder.training.build import create_trainer

    # Ensure dataset exists, then load BEFORE auto-lengths so we can sample from it
    if not os.path.isdir(cfg.dataset_path):
        logger.error("Dataset directory not found at '%s'. Run reason-prepare first.", cfg.dataset_path)
        raise SystemExit(1)
    logger.info("Loading dataset from %s", cfg.dataset_path)
    dataset = load_prepared(cfg.dataset_path)

    # Optionally auto-derive lengths and/or dry-run memory estimation
    if cfg.auto_lengths or cfg.dry_run or cfg.document_length <= 0 or cfg.query_length <= 0:
        try:
            from transformers import AutoTokenizer, AutoConfig
            tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
            # Sample a subset
            ds = dataset["train"] if "train" in dataset else dataset
            n = min(len(ds), max(1, cfg.length_sample))
            step = max(1, len(ds) // n)
            queries = []
            docs = []
            for i in range(0, len(ds), step):
                ex = ds[i]
                q = ex.get("query", "")
                if isinstance(q, list):
                    q = " ".join(map(str, q))
                queries.append(q)
                # Pull one doc text from pos first, then neg
                dtext = ""
                for field in ("pos", "neg"):
                    arr = ex.get(field, []) or []
                    if isinstance(arr, list) and arr:
                        item = arr[0]
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            dtext = item[1]
                            break
                docs.append(dtext)
                if len(queries) >= n:
                    break
            q_lens = tok(queries, add_special_tokens=True, truncation=False, padding=False, return_length=True)["length"]
            d_lens = tok(docs, add_special_tokens=True, truncation=False, padding=False, return_length=True)["length"]
            def pct(arr, p):
                arr = sorted(arr)
                idx = int(max(0, min(len(arr) - 1, round(p * (len(arr) - 1)))))
                return int(arr[idx])
            q_len = pct(q_lens, cfg.length_percentile)
            d_len = pct(d_lens, cfg.length_percentile)
            # Respect model context window
            max_ctx = tok.model_max_length if tok.model_max_length and tok.model_max_length < 10**9 else 4096
            q_len = max(32, min(q_len, max_ctx))
            d_len = max(128, min(d_len, max_ctx))
            if cfg.auto_lengths or cfg.document_length <= 0 or cfg.query_length <= 0:
                cfg.query_length = q_len
                cfg.document_length = d_len
        except Exception as e:
            logger.warning("Auto length derivation failed: %s", e)

    if cfg.dry_run:
        try:
            from transformers import AutoConfig
            conf = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
            hidden = int(getattr(conf, "hidden_size", 768))
            layers = int(getattr(conf, "num_hidden_layers", 12))
            dt_bytes = 2 if (cfg.fp16 or cfg.bf16) and not cfg.force_cpu else 4
            toks = (cfg.document_length + cfg.query_length)
            per_mb = cfg.mini_batch_size * toks * hidden * layers * dt_bytes / (1024**2)
            est_mb = per_mb * 1.3  # overhead fudge
            logger.info("Dry-run: lengths (doc=%d, query=%d), est activation memory per step ~ %.1f MB (mini_batch=%d)", cfg.document_length, cfg.query_length, est_mb, cfg.mini_batch_size)
        except Exception as e:
            logger.info("Dry-run: lengths (doc=%d, query=%d)", cfg.document_length, cfg.query_length)
        return

    logger.info("Preparing splits and preprocessing examples...")
    train_dataset, _ = prepare_splits(dataset, cfg.do_sample, cfg.sample_size, cfg.seed, cfg.eval_holdout)
    logger.info("Train size: %d", len(train_dataset))

    logger.info("Building trainer for model %s", cfg.base_model)
    trainer = create_trainer(cfg, train_dataset)
    model = trainer.model

    logger.info("Starting training → output: %s", cfg.output_dir)
    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    try:
        model.save_pretrained(final_dir)
    except Exception:
        model.save(cfg.output_dir)
    logger.info("Training complete. Artifacts saved under %s", cfg.output_dir)


if __name__ == "__main__":
    main()