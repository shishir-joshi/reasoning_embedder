from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
from datasets import load_from_disk, Dataset


def load_prepared(path: str):
    return load_from_disk(path)


def _flatten_pair(pair) -> Optional[str]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    instr, text = pair
    instr = instr if isinstance(instr, str) else str(instr)
    text = text if isinstance(text, str) else str(text)
    if instr:
        return f"{instr} {text}".strip()
    return text


def preprocess_flatten_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure query is a single string
    q = entry.get("query")
    if isinstance(q, list):
        entry["query"] = " ".join(q)
    else:
        entry["query"] = str(q)

    # Take first pos/neg pair, flatten to string
    pos_list = entry.get("pos", []) or []
    neg_list = entry.get("neg", []) or []

    pos_str = _flatten_pair(pos_list[0]) if len(pos_list) > 0 else None
    neg_str = _flatten_pair(neg_list[0]) if len(neg_list) > 0 else None

    entry["pos"] = pos_str
    entry["neg"] = neg_str
    return entry


def prepare_splits(dataset, do_sample: bool, sample_size: Optional[int], seed: int, eval_holdout: float) -> Tuple[Dataset, Optional[Dataset]]:
    raw = dataset["train"]
    if do_sample and sample_size is not None:
        raw = raw.shuffle(seed=seed)
        if len(raw) > sample_size:
            raw = raw.select(range(sample_size))
        # For small samples, use 10% eval
        splits = raw.train_test_split(test_size=0.1, seed=seed)
    else:
        splits = raw.train_test_split(test_size=eval_holdout, seed=seed)

    train = splits["train"].map(preprocess_flatten_entry)
    eval_ds = splits["test"].map(preprocess_flatten_entry) if "test" in splits else None

    # Filter out malformed examples with missing strings
    train = train.filter(lambda x: isinstance(x.get("query"), str) and isinstance(x.get("pos"), str) and isinstance(x.get("neg"), str))
    if eval_ds is not None:
        eval_ds = eval_ds.filter(lambda x: isinstance(x.get("query"), str) and isinstance(x.get("pos"), str) and isinstance(x.get("neg"), str))

    return train, eval_ds
