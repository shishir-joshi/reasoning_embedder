import logging
import os
import torch
from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import evaluation, losses, models, utils

# --- Configuration ---

# Setup logging to be informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model to be fine-tuned. This is the base model for Reason-ModernColBERT.
BASE_MODEL_ID = "lightonai/GTE-ModernColBERT-v1"

# Path to the dataset created by the prepare_dataset.py script.
DATASET_PATH = "prepared_reasonir_hq"

# Directory where the final fine-tuned model will be saved.
OUTPUT_DIR = "output/reason-moderncolbert-finetuned"

# --- Training Hyperparameters ---
# These are set for a minimal but functional run. For best results, these should be tuned.
# The original Reason-ModernColBERT model card can be a good reference.
NUM_TRAIN_EPOCHS = 1  # A single epoch is good for a quick test run.
PER_DEVICE_TRAIN_BATCH_SIZE = 16  # Adjust based on your GPU memory.
PER_DEVICE_EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # A common default for fine-tuning transformers.
WARMUP_STEPS = 100  # Number of steps to warm up the learning rate.

# --- Data Sampling Options ----------------------------------------------------
# Enable these flags if you want to do a **quick sanity-check run** on a small
# subset of the data.  Disable (`SAMPLE_DATA = False`) for full-scale training.
SAMPLE_DATA: bool = True          # Toggle sampling on/off
SAMPLE_SIZE: int = 1_000          # Number of examples to keep when sampling

def main():
    """
    Main function to orchestrate the model fine-tuning process.
    """
    # --- 1. Load the Prepared Dataset ---
    logger.info(f"Loading dataset from disk at: {DATASET_PATH}")
    if not os.path.isdir(DATASET_PATH):
        logger.error(f"Dataset directory not found at '{DATASET_PATH}'.")
        logger.error("Please run the `prepare_dataset.py` script first to generate the dataset.")
        return

    # Load the dataset, which was processed and saved by the preparation script.
    dataset = load_from_disk(DATASET_PATH)
    
    # ---------------------------------------------------------------------- #
    # Split the dataset into train / validation (optionally with sampling)   #
    # ---------------------------------------------------------------------- #
    logger.info("Splitting dataset into training and evaluation sets...")

    train_split_raw = dataset["train"]

    if SAMPLE_DATA:
        logger.info(f"SAMPLING ENABLED → selecting up to {SAMPLE_SIZE} examples")
        # Shuffle first to ensure we get a representative subset
        train_split_raw = train_split_raw.shuffle(seed=42)
        if len(train_split_raw) > SAMPLE_SIZE:
            train_split_raw = train_split_raw.select(range(SAMPLE_SIZE))
            logger.info(f"Sampled dataset size: {len(train_split_raw)} examples")
        else:
            logger.warning(
                "Requested SAMPLE_SIZE (%d) larger than dataset (%d). "
                "Using full dataset.",
                SAMPLE_SIZE,
                len(train_split_raw),
            )

        # Use a slightly larger eval split for very small samples (10 %)
        dataset_splits = train_split_raw.train_test_split(test_size=0.1, seed=42)
    else:
        # Full dataset – keep a small held-out set (1 %)
        dataset_splits = train_split_raw.train_test_split(test_size=0.01, seed=42)

    train_dataset = dataset_splits['train']
    eval_dataset = dataset_splits['test']
    
    logger.info(f"Dataset loaded. Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

    # --- 2. Initialize the Model ---
    logger.info(f"Initializing model from base: {BASE_MODEL_ID}...")
    # We use the ColBERT class from pylate, which is designed for late-interaction models.
    model = models.ColBERT(model_name_or_path=BASE_MODEL_ID)

    # --- 3. Define Training Components ---
    
    # a) Loss Function: Contrastive loss is ideal for triplet data (query, positive, negative).
    # It works by pulling the (query, positive) pair closer together in the embedding space
    # while pushing the (query, negative) pair further apart.
    logger.info("Initializing Contrastive Loss function...")
    train_loss = losses.Contrastive(model=model, temperature=0.02)
    
    # b) Evaluator: This component will be used to evaluate the model's performance on the
    # validation set during training. The TripletEvaluator measures the accuracy of correctly
    # ranking the positive document higher than the negative one for a given query.
    logger.info("Initializing Triplet Evaluator...")
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["pos"],
        negatives=eval_dataset["neg"],
        name="reasonir-eval",
    )

    # c) Data Collator: This utility function is crucial for ColBERT models. It takes a list
    # of training examples and prepares them into a batch that the model can process.
    # This includes tokenizing the text and ensuring correct formatting.
    data_collator = utils.ColBERTCollator(model.tokenize)

    # d) Training Arguments: This object holds all the configuration for the training run.
    logger.info(f"Configuring training arguments. Output will be saved to: {OUTPUT_DIR}")
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        fp16=torch.cuda.is_available(),  # Use mixed precision for speed if a CUDA GPU is available
        bf16=False,  # Set to True if using an Ampere or newer NVIDIA GPU
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # This will ensure the best checkpoint is loaded at the end
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        run_name="finetune-reason-moderncolbert",
    )

    # --- 4. Initialize the Trainer ---
    # The SentenceTransformerTrainer orchestrates the entire training process.
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=data_collator,
    )

    # --- 5. Start Training ---
    logger.info("Starting model fine-tuning. This may take some time depending on your hardware.")
    trainer.train()
    
    # --- 6. Save the Final Model ---
    # The trainer saves the best checkpoint, but we can also save the final model state manually.
    logger.info(f"Training complete. Saving the final model to {OUTPUT_DIR}...")
    model.save(OUTPUT_DIR)
    logger.info("Model saved successfully.")

if __name__ == "__main__":
    main()
