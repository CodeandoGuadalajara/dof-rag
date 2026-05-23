"""Configuration for the DOF RAG PoC."""
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DOF_MD_DIR = Path(os.environ.get("DOF_MD_DIR", "./dof_md"))
DB_PATH = Path(os.environ.get("RAG_DB_PATH", "./rag_poc/dof_rag.db"))
MODEL_CACHE_DIR = Path(os.environ.get("RAG_MODEL_CACHE", "./rag_poc/.model_cache"))

# ------------------------------------------------------------------
# Chunking (token-based approx: 1 token ≈ 3.5 chars in legal Spanish)
# ------------------------------------------------------------------
MAX_TOKENS = 800
OVERLAP_TOKENS = 50

# ------------------------------------------------------------------
# Embedding model (local ONNX)
# ------------------------------------------------------------------
HF_REPO = "perplexity-ai/pplx-embed-context-v1-0.6b"
ONNX_SUBDIR = "onnx"
ONNX_MODEL_FILE = "model.onnx"
EMBED_DIM = 1024

# ------------------------------------------------------------------
# Hybrid search
# ------------------------------------------------------------------
VECTOR_TOP_K = 20
FTS_TOP_K = 20
FINAL_TOP_K = 10
RRF_K = 60                  # reciprocal rank fusion constant
