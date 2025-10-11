import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_doc_and_ids(doc_pairs):
    """Extracts document IDs and content from a list of document pairs."""
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(str(dp['id']))
        documents.append(dp['content'])
    return documents, doc_ids

def process_documents(entry, id2doc):
    """
    Replace the integer/string IDs contained in the `pos` and `neg` fields
    of *entry* with their corresponding document text.

    Each element in `entry["pos"]` / `entry["neg"]` is expected to be a
    two-item list: `[instruction, doc_id]`.
      • If *instruction* is an empty string we store the **document text
        only**.
      • If *instruction* is non-empty we keep a two-element list
        `[instruction, document text]`.

    Any pairs that cannot be resolved (e.g. missing `doc_id`) are silently
    dropped but a debug message is logged for traceability.
    """
    try:
        for field in ("pos", "neg"):
            updated: list = []
            for pair in entry.get(field, []):
                # Guard against malformed data
                if (not isinstance(pair, (list, tuple))) or len(pair) != 2:
                    logger.debug("Malformed %s pair skipped: %s", field, pair)
                    continue

                instruction, doc_id = pair
                # Ensure we look up using a string key (BRIGHT ids are strings)
                doc_text = id2doc.get(str(doc_id))
                if not doc_text:
                    logger.debug("Document id '%s' not found for field '%s'.", doc_id, field)
                    continue

                if instruction:
                    updated.append([instruction, doc_text])
                else:
                    updated.append(doc_text)

            entry[field] = updated

    except Exception as exc:  # pragma: no cover
        # Catch any unexpected errors so that .map() continues processing
        logger.error("Failed to process entry: %s", exc, exc_info=True)

    return entry

def main():
    """Main function to process and save the dataset."""
    try:
        # Load the datasets
        logger.info("Loading hq_dataset and bright_docs...")
        hq_dataset = load_dataset("reasonir/reasonir-data", "hq")
        bright_docs = load_dataset("xlangai/BRIGHT", "documents")
        logger.info("Datasets loaded successfully.")

        # Process BRIGHT documents
        logger.info("Processing BRIGHT documents...")
        all_docs = []
        all_ids = []
        for task in bright_docs.keys():
            docs, ids = get_doc_and_ids(bright_docs[task])
            all_docs.extend(docs)
            all_ids.extend(ids)

        id2doc = dict(zip(all_ids, all_docs))
        logger.info(f"Processed {len(id2doc)} documents from BRIGHT.")

        # Process the hq_dataset
        logger.info("Processing hq_dataset...")
        processed_hq_dataset = hq_dataset.map(
            lambda x: process_documents(x, id2doc),
            desc="Resolving document ids",
        )
        logger.info("hq_dataset processed successfully.")

        # Save the processed dataset
        output_path = "prepared_reasonir_hq"
        logger.info(f"Saving processed dataset to {output_path}...")
        processed_hq_dataset.save_to_disk(output_path)
        logger.info("Dataset saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
