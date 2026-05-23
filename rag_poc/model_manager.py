"""Download and cache the local pplx-embed ONNX model + tokenizer."""
from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from rag_poc.config import HF_REPO, MODEL_CACHE_DIR, ONNX_MODEL_FILE, ONNX_SUBDIR

logger = logging.getLogger("rag_poc.model_manager")

_tokenizer: AutoTokenizer | None = None
_session: ort.InferenceSession | None = None


def _ensure_model_cached() -> Path:
    """Download ONNX weights + tokenizer files if not already present."""
    cache = MODEL_CACHE_DIR / HF_REPO.replace("/", "--")
    onnx_dir = cache / ONNX_SUBDIR
    model_path = onnx_dir / ONNX_MODEL_FILE

    if model_path.exists():
        logger.debug("ONNX model already cached at %s", model_path)
        return cache

    logger.info("Downloading %s ONNX model (first run, ~1.2 GB)...", HF_REPO)
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{ONNX_SUBDIR}/**", "*.json", "*.txt", "*.py"],
        cache_dir=str(cache / ".hf_cache"),
        local_dir=str(cache),
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Expected ONNX model at {model_path} after download")
    logger.info("Model cached at %s", cache)
    return cache


def get_tokenizer() -> AutoTokenizer:
    """Return the (cached) AutoTokenizer."""
    global _tokenizer
    if _tokenizer is None:
        cache = _ensure_model_cached()
        _tokenizer = AutoTokenizer.from_pretrained(
            str(cache),
            trust_remote_code=True,
        )
    return _tokenizer


def get_session() -> ort.InferenceSession:
    """Return the (cached) ONNX Runtime session."""
    global _session
    if _session is None:
        cache = _ensure_model_cached()
        model_path = cache / ONNX_SUBDIR / ONNX_MODEL_FILE
        providers = ort.get_available_providers()
        # Prefer CPUExecutionProvider for stability; GPU can be added later
        if "CPUExecutionProvider" in providers:
            chosen = ["CPUExecutionProvider"]
        else:
            chosen = providers[:1]  # whatever is available
        logger.info("ONNX Runtime providers: %s", chosen)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=chosen,
        )
    return _session
