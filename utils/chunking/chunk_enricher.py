"""Context enrichment for chunks (Stage 3)."""

import re
from typing import Optional

from utils.chunking.chunk_config import EnrichmentConfig
from utils.chunking.chunk_model import Chunk

# Sentences starting with these indicate dependency on prior context
DEPENDENCY_MARKERS = re.compile(
    r"^(然而|但是|因此|所以|如上|如下|尽管如此|换言之|也就是说|"
    r"However|Therefore|Thus|As mentioned above|As shown|In contrast|Nevertheless)",
)


class ChunkEnricher:
    """Enriches chunks with context metadata: section chain, adjacent summaries, self-contained score."""

    def __init__(self, config: Optional[EnrichmentConfig] = None):
        self.config = config or EnrichmentConfig()

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich all chunks with context metadata.

        Args:
            chunks: List of chunks (output from Stage 2).

        Returns:
            Same chunks with enriched metadata.
        """
        if not chunks:
            return []

        self._set_section_chains(chunks)
        self._set_adjacent_summaries(chunks)
        self._set_self_contained_scores(chunks)
        return chunks

    def _set_section_chains(self, chunks: list[Chunk]) -> None:
        """Ensure section_path metadata is present."""
        for chunk in chunks:
            if "section_path" not in chunk.metadata:
                chunk.metadata["section_path"] = ""

    def _set_adjacent_summaries(self, chunks: list[Chunk]) -> None:
        """Set prev_summary and next_summary using extractive mode."""
        for i, chunk in enumerate(chunks):
            if self.config.use_extractive_summary:
                if i > 0:
                    chunk.prev_summary = self._extract_summary(chunks[i - 1].text, mode="first")
                if i < len(chunks) - 1:
                    chunk.next_summary = self._extract_summary(chunks[i + 1].text, mode="last")

    def _extract_summary(self, text: str, mode: str = "first") -> str:
        """Extract first or last sentence as summary."""
        sentences = re.split(r'(?<=[。！？.\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if mode == "first" and sentences:
            return sentences[0]
        elif mode == "last" and sentences:
            # Return last 2 sentences for richer context when available
            return " ".join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]
        return ""

    def _set_self_contained_scores(self, chunks: list[Chunk]) -> None:
        """Score each chunk's self-containment (0-1)."""
        for chunk in chunks:
            score = 0.5  # base score
            text = chunk.text.strip()

            # Has section path (higher score)
            if chunk.metadata.get("section_path"):
                score += 0.2

            # Starts with dependency marker → lower score
            # Check last non-empty line for content (skipping headings)
            lines = [l for l in text.split("\n") if l.strip()]
            check_line = lines[-1] if lines else text
            if DEPENDENCY_MARKERS.match(check_line):
                score -= 0.3

            # Very short → lower score (incomplete thought, < 15 chars)
            if len(text) < 15:
                score -= 0.2

            # Long and self-describing → higher score
            if len(text) > 200:
                score += 0.1

            chunk.self_contained_score = round(max(0.0, min(1.0, score)), 2)
