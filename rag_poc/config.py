"""Minimal configuration for the DOF chunker."""

# Chunking limits (tokens)
MAX_TOKENS = 800
OVERLAP_TOKENS = 50

# H2 compound documents often contain whole decretes/resolutions as a single
# H2 section. Keeping the H2 intact up to this limit preserves document-level
# coherence before falling back to paragraph-level splitting. It is capped at
# the oversized threshold (MAX_TOKENS * 1.10) so that H2 chunks are never
# reported as oversized.
H2_MAX_TOKENS = int(MAX_TOKENS * 1.10)
