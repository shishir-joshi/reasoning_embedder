from __future__ import annotations

"""
Compatibility helpers for adapting HF models/tokenizers to training pipeline needs.

Currently provides:
- ensure_tokenizer_padding: guarantees a usable pad_token (alias to eos or add [PAD])
"""

from typing import Optional


def ensure_tokenizer_padding(model, prefer_eos: bool = True, new_pad_token: str = "[PAD]") -> None:
    """Ensure the wrapped sentence-transformers/PyLate model has a tokenizer with pad_token.

    - If tokenizer has no pad_token and has eos_token and prefer_eos=True, alias pad->eos.
    - Else add a new pad token and resize embeddings if auto_model exposes resize_token_embeddings.
    - Also fixes missing pad_token_id when pad_token exists but id is None.

    The function is safe to call multiple times and swallows unexpected errors.
    """
    try:
        mod = model._first_module()  # sentence-transformers internal: holds tokenizer and auto_model
        tok = getattr(mod, "tokenizer", None)
        auto = getattr(mod, "auto_model", None)
        if tok is None:
            return

        if getattr(tok, "pad_token", None) is None:
            if prefer_eos and getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": new_pad_token})
                if auto is not None and hasattr(auto, "resize_token_embeddings"):
                    auto.resize_token_embeddings(len(tok))

        if getattr(tok, "pad_token_id", None) is None and tok.pad_token is not None:
            tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
            if auto is not None and hasattr(auto, "resize_token_embeddings"):
                auto.resize_token_embeddings(len(tok))
    except Exception:
        # Non-fatal: different model wrappers may not expose expected internals
        pass
