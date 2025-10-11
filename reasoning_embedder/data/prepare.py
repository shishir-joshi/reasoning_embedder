import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_doc_and_ids(doc_pairs):
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(str(dp['id']))
        documents.append(dp['content'])
    return documents, doc_ids


def process_documents(entry, id2doc):
    try:
        q = entry.get("query")
        if isinstance(q, list):
            entry["query"] = " ".join(q)

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
            updated_pos.append([instruction if isinstance(instruction, str) else str(instruction), doc_text])
        entry["pos"] = updated_pos

        updated_neg: list = []
        for pair in entry.get("neg", []):
            if (not isinstance(pair, (list, tuple))) or len(pair) != 2:
                logger.debug("Malformed neg pair skipped: %s", pair)
                continue
            instruction, maybe_text = pair
            text = id2doc.get(str(maybe_text), None) or maybe_text
            updated_neg.append([instruction if isinstance(instruction, str) else str(instruction), text])
        entry["neg"] = updated_neg

    except Exception as exc:  # pragma: no cover
        logger.error("Failed to process entry: %s", exc, exc_info=True)

    return entry


def main():
    try:
        logger.info("Loading hq_dataset and bright_docs...")
        hq_dataset = load_dataset("reasonir/reasonir-data", "hq")
        bright_docs = load_dataset("xlangai/BRIGHT", "documents")
        logger.info("Datasets loaded successfully.")

        logger.info("Processing BRIGHT documents...")
        all_docs = []
        all_ids = []
        for task in bright_docs.keys():
            docs, ids = get_doc_and_ids(bright_docs[task])
            all_docs.extend(docs)
            all_ids.extend(ids)

        id2doc = dict(zip(all_ids, all_docs))
        logger.info(f"Processed {len(id2doc)} documents from BRIGHT.")

        logger.info("Processing hq_dataset...")
        processed_hq_dataset = hq_dataset.map(
            lambda x: process_documents(x, id2doc),
            desc="Resolving document ids",
        )
        logger.info("hq_dataset processed successfully.")

        output_path = "prepared_reasonir_hq"
        logger.info(f"Saving processed dataset to {output_path}...")
        processed_hq_dataset.save_to_disk(output_path)
        logger.info("Dataset saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()