import os
from typing import Any

import pytest


def _materialize_iterable_dataset(iterable, n: int):
    from datasets import Dataset

    rows = []
    for i, row in enumerate(iterable):
        if i >= n:
            break
        rows.append(row)
    return Dataset.from_list(rows)


def _build_mock_hq_dataset():
    from datasets import Dataset, DatasetDict

    # Keep schema consistent for HF Datasets (avoid mixing list and string types).
    # List-to-string conversion is tested in test_process_documents_skips_malformed_pairs.
    rows = [
        {
            "query": "what is photosynthesis",
            "pos": [["inst", "1"], ["inst", "2"]],
            "neg": [["inst", "3"], ["inst", "already text"]],
        },
        {
            "query": "who wrote hamlet",
            "pos": [["inst", "missing_doc_id"]],
            "neg": [],
        },
    ]
    return DatasetDict({"train": Dataset.from_list(rows)})


def test_process_documents_skips_malformed_pairs():
    from reasoning_embedder.data.prepare import process_documents

    entry = {
        "query": ["hello", "world"],
        "pos": [["inst", "1"], ["malformed"], "oops"],
        "neg": [["inst", "2"], ["malformed"], None],
    }
    id2doc = {"1": "doc one", "2": "doc two"}

    out = process_documents(entry, id2doc)
    assert out["query"] == "hello world"
    assert out["pos"] == [["inst", "doc one"]]
    # neg resolves known id, keeps unknown raw value when present and pair is well-formed
    assert out["neg"][0] == ["inst", "doc two"]


def _build_mock_bright_docs():
    from datasets import Dataset, DatasetDict

    docs = [
        {"id": "1", "content": "doc one text"},
        {"id": "2", "content": "doc two text"},
        {"id": "3", "content": "doc three text"},
    ]

    # BRIGHT documents dataset is iterated by task key
    return DatasetDict({"biology": Dataset.from_list(docs)})


@pytest.mark.parametrize("use_hf", [False, True])
def test_prepare_main_combined(tmp_path, monkeypatch, use_hf):
    """Combined test for prepare.py.

    - Default (use_hf=False): uses mocked datasets (fast, no network)
    - Optional (use_hf=True): loads small streaming slices from Hugging Face if
      REASONING_EMBEDDER_TEST_USE_HF=1 is set.

    This test validates that:
    - output dataset is saved to data/prepared_reasonir_hq
    - pos doc ids are resolved into doc text when available
    - neg second field is resolved when it matches a doc id, else preserved
    - query list is joined into string
    """

    env_flag = os.getenv("REASONING_EMBEDDER_TEST_USE_HF", "0")
    if use_hf and env_flag != "1":
        pytest.skip("Set REASONING_EMBEDDER_TEST_USE_HF=1 to run HF-backed test")

    from reasoning_embedder.data import prepare as prepare_mod

    def fake_load_dataset(repo_id: str, config_name: str, *args: Any, **kwargs: Any):
        if use_hf:
            from datasets import DatasetDict, load_dataset as hf_load_dataset

            try:
                if repo_id == "reasonir/reasonir-data" and config_name == "hq":
                    ds_dict = hf_load_dataset(repo_id, config_name, streaming=True)
                    split_name = next(iter(ds_dict.keys()))
                    ds = _materialize_iterable_dataset(ds_dict[split_name], n=25)
                    return DatasetDict({split_name: ds})

                if repo_id == "xlangai/BRIGHT" and config_name == "documents":
                    docs_dict = hf_load_dataset(repo_id, config_name, streaming=True)
                    task_names = list(docs_dict.keys())[:1]
                    materialized = {}
                    for task in task_names:
                        materialized[task] = _materialize_iterable_dataset(
                            docs_dict[task], n=50
                        )
                    return DatasetDict(materialized)

            except Exception as exc:
                pytest.skip(f"HuggingFace datasets not available: {exc}")

        # Mocked datasets
        if repo_id == "reasonir/reasonir-data" and config_name == "hq":
            return _build_mock_hq_dataset()
        if repo_id == "xlangai/BRIGHT" and config_name == "documents":
            return _build_mock_bright_docs()

        raise AssertionError(f"Unexpected dataset request: {repo_id} {config_name}")

    monkeypatch.setattr(prepare_mod, "load_dataset", fake_load_dataset)
    monkeypatch.chdir(tmp_path)

    prepare_mod.main()

    out_dir = tmp_path / "data" / "prepared_reasonir_hq"
    assert out_dir.exists(), "prepare.main() did not create output directory"

    from datasets import load_from_disk

    saved = load_from_disk(str(out_dir))

    # handle either DatasetDict or Dataset depending on save format
    if hasattr(saved, "keys") and "train" in saved:
        ds = saved["train"]
    else:
        ds = saved

    row0 = ds[0]

    # query list becomes string
    assert isinstance(row0["query"], str)

    # pos resolved to [instruction, doc_text]
    assert all(isinstance(x, list) and len(x) == 2 for x in row0["pos"])
    assert any("doc one text" in x[1] for x in row0["pos"]) or use_hf

    # neg keeps unknown second field, resolves known doc id
    assert all(isinstance(x, list) and len(x) == 2 for x in row0["neg"])
    if not use_hf:
        assert any(x[1] == "already text" for x in row0["neg"])
        assert any(x[1] == "doc three text" for x in row0["neg"])
