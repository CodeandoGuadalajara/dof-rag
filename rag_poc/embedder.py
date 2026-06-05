"""Local ONNX embedding for pplx-embed-context-v1-0.6b with late chunking.

Chunks from the same document are concatenated with SEP tokens so the model
sees them in shared context.  After inference we split the token-level
embeddings back into per-chunk embeddings via late chunking.

All vectors are L2-normalised so that sqlite-vec Euclidean distance is
equivalent to cosine distance.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from rag_poc.config import EMBED_DIM
from rag_poc.model_manager import get_session, get_tokenizer

logger = logging.getLogger("rag_poc.embedder")


# ------------------------------------------------------------------
# Mean pooling  (float32)
# ------------------------------------------------------------------
def _mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


# ------------------------------------------------------------------
# Late chunking — split concatenated sequence by SEP positions
# ------------------------------------------------------------------
def _extract_chunks_from_concatenated(
    input_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    sep_token_id: int,
) -> list[list[torch.Tensor]]:
    """
    Extract individual chunk embeddings from a concatenated sequence.
    Input:  [chunk1][SEP][chunk2][SEP]...
    Output: list of docs, each doc is a list of chunk embedding tensors.
    """
    batch_size = input_ids.shape[0]
    all_docs: list[list[torch.Tensor]] = []

    for b in range(batch_size):
        valid = attention_mask[b].bool()
        sep_positions = ((input_ids[b] == sep_token_id) & valid).nonzero(as_tuple=True)[0]

        chunks: list[torch.Tensor] = []
        start = 0
        for pos in sep_positions:
            emb = _mean_pooling(
                token_embeddings[b, start:pos].unsqueeze(0),
                attention_mask[b, start:pos].unsqueeze(0),
            ).squeeze(0)
            chunks.append(emb)
            start = int(pos.item()) + 1

        # Last chunk (after final SEP or from start if no SEPs)
        last_valid = int(attention_mask[b].sum().item())
        if start < last_valid:
            emb = _mean_pooling(
                token_embeddings[b, start:last_valid].unsqueeze(0),
                attention_mask[b, start:last_valid].unsqueeze(0),
            ).squeeze(0)
            chunks.append(emb)
        elif not chunks:
            # Empty — zero vector (should not happen with real text)
            chunks.append(
                torch.zeros(token_embeddings.shape[-1], dtype=torch.float32)
            )

        all_docs.append(chunks)

    return all_docs


# ------------------------------------------------------------------
# Normalisation  (L2 → unit vector)
# ------------------------------------------------------------------
def _normalize(vec: torch.Tensor | list[float]) -> list[float]:
    """L2-normalise so Euclidean distance ≡ cosine distance."""
    if isinstance(vec, list):
        vec = torch.tensor(vec, dtype=torch.float32)
    norm = torch.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()


# ------------------------------------------------------------------
# Core API
# ------------------------------------------------------------------
def embed_documents(
    docs: list[list[str]],
    batch_size: int = 8,
) -> list[list[list[float]]]:
    """
    Embed documents using the contextual local ONNX model.

    Args:
        docs: docs[i] = list of text chunks for document i.
        batch_size: max documents per ONNX forward pass.

    Returns:
        embeddings[i][j] = normalised float vector for chunk j of doc i.
    """
    tokenizer = get_tokenizer()
    session = get_session()
    sep_token = tokenizer.sep_token

    all_embeddings: list[list[list[float]]] = []

    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]

        # 1. Concatenate chunks with SEP for each document
        doc_strings = [sep_token.join(chunks) for chunks in batch]

        # 2. Tokenize
        tokenized = tokenizer(
            doc_strings,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        inputs = {
            "input_ids": tokenized["input_ids"].astype(np.int64),
            "attention_mask": tokenized["attention_mask"].astype(np.int64),
        }

        # 3. ONNX inference
        output_names = [out.name for out in session.get_outputs()]
        onnx_outputs = session.run(output_names, inputs)
        last_hidden_state = onnx_outputs[0]  # (batch, seq_len, hidden_dim)

        # 4. Late chunking — split back into per-chunk embeddings
        batch_chunk_embeddings = _extract_chunks_from_concatenated(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            token_embeddings=torch.from_numpy(last_hidden_state),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
            sep_token_id=tokenizer.sep_token_id,
        )

        # 5. Normalise each chunk vector
        for doc_chunks in batch_chunk_embeddings:
            normalised = [_normalize(chunk_vec) for chunk_vec in doc_chunks]
            all_embeddings.append(normalised)

    # Validate dimensions
    for doc_idx, doc_embs in enumerate(all_embeddings):
        expected_n = len(docs[doc_idx])
        actual_n = len(doc_embs)
        if actual_n != expected_n:
            raise RuntimeError(
                f"Doc {doc_idx}: expected {expected_n} embeddings, got {actual_n}"
            )
        for chunk_idx, vec in enumerate(doc_embs):
            if len(vec) != EMBED_DIM:
                raise RuntimeError(
                    f"Doc {doc_idx} chunk {chunk_idx}: expected dim {EMBED_DIM}, got {len(vec)}"
                )

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string and normalise it."""
    result = embed_documents([[text]], batch_size=1)
    return result[0][0]


# ------------------------------------------------------------------
# Legacy compatibility
# ------------------------------------------------------------------
def get_provider_info() -> dict:
    return {
        "provider": "local_onnx",
        "model": "pplx-embed-context-v1-0.6b",
        "dimensions": EMBED_DIM,
    }
