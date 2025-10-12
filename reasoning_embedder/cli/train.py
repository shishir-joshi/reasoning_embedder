import argparse
import logging
import os
from reasoning_embedder.training.config import TrainingConfig
from reasoning_embedder.training.data import load_prepared, prepare_splits
from reasoning_embedder.training.build import create_trainer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> TrainingConfig:
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

    p.add_argument("--sample_size", type=int, default=None)
    p.add_argument("--sample", action="store_true", default=False)
    p.add_argument("--eval_holdout", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

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
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

    if not os.path.isdir(cfg.dataset_path):
        logger.error("Dataset directory not found at '%s'. Run reason-prepare first.", cfg.dataset_path)
        raise SystemExit(1)

    logger.info("Loading dataset from %s", cfg.dataset_path)
    dataset = load_prepared(cfg.dataset_path)

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