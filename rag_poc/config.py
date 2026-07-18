"""Minimal configuration for the DOF chunker."""

# Chunking limits (tokens)
MAX_TOKENS = 800
OVERLAP_TOKENS = 50

# H2 compound documents often contain whole decretes/resolutions as a single
# H2 section. Keeping the H2 intact up to this limit preserves document-level
# coherence before falling back to paragraph-level splitting.
H2_MAX_TOKENS = 1_500
