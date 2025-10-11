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
    Populate `pos` and `neg` columns with document text in a consistent shape
    and normalize the query field.

    Expectations from ReasonIR HQ formatting (as seen in the notebook):
    - entry["pos"] elements are [instruction, doc_id] where doc_id refers to
      a BRIGHT document id. We must map via `id2doc`.
    - entry["neg"] elements are [instruction, doc_text] where the second
      element is already the raw document text. No mapping needed.

    Output shape for both fields will be a list of two-item lists:
      [instruction (can be empty string), document_text]

    Additionally, if query is a list of tokens/segments, join into a single
    string.
    """
    try:
        # Normalize query to a string if it's a list
        q = entry.get("query")
        if isinstance(q, list):
            entry["query"] = " ".join(q)

        # Process positives: map id -> text
        updated_pos: list = []
        for pair in entry.get("pos", []):
            if (not isinstance(pair, (list, tuple))) or len(pair) != 2:
                logger.debug("Malformed pos pair skipped: %s", pair)
                continue
            instruction, doc_id = pair
            doc_text = id2doc.get(str(doc_id))
            if not doc_text:
                logger.debug("Document id '%s' not found for field 'pos'.", doc_id)
                continue
            # Preserve two-item list shape even if instruction is empty
            updated_pos.append([instruction if isinstance(instruction, str) else str(instruction), doc_text])
        entry["pos"] = updated_pos

        # Process negatives: second element is already text
        updated_neg: list = []
        for pair in entry.get("neg", []):
            if (not isinstance(pair, (list, tuple))) or len(pair) != 2:
                logger.debug("Malformed neg pair skipped: %s", pair)
                continue
            instruction, maybe_text = pair
            # If the second element looks like an id that exists in id2doc,
            # we still prefer the mapped text; otherwise treat as text.
            text = id2doc.get(str(maybe_text), None) or maybe_text
            updated_neg.append([instruction if isinstance(instruction, str) else str(instruction), text])
        entry["neg"] = updated_neg

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
