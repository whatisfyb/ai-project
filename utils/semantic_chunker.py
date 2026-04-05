"""Semantic chunking using embedding-based similarity (Stage 2)."""

import math
import re
from typing import Optional

from utils.chunk_model import Chunk, ChunkType
from utils.chunk_config import SemanticChunkConfig
from utils.embedding import embed_texts


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[。！？.])|(?<=[；;])|(?<=\n))"
)


class SemanticChunker:
    """Refines chunks by detecting semantic boundaries via embeddings."""

    def __init__(self, config: Optional[SemanticChunkConfig] = None,
                 *, min_chars: Optional[int] = None,
                 max_chars: Optional[int] = None):
        self.config = config or SemanticChunkConfig()
        if min_chars is not None:
            object.__setattr__(self.config, "min_chars", min_chars)
        if max_chars is not None:
            object.__setattr__(self.config, "max_chars", max_chars)

    def chunk(self, chunks: list[Chunk]) -> list[Chunk]:
        """Process chunks through semantic boundary detection.

        Args:
            chunks: Output from Stage 1 (structure-aware chunker).

        Returns:
            Semantically-refined chunks.
        """
        if not chunks:
            return []

        result = []
        for chunk in chunks:
            sub_chunks = self._process_chunk(chunk)
            result.extend(sub_chunks)

        result = self._merge_short(result)
        return result

    def _process_chunk(self, chunk: Chunk) -> list[Chunk]:
        """Split a single chunk by semantic similarity."""
        text = chunk.text

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [chunk]

        # Embed sentences in batches
        try:
            embeddings = self._embed_sentences(sentences)
        except Exception:
            # Embedding failure: chunk as-is
            return [chunk]

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_sim(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Compute cut threshold: 25th percentile of this chunk's similarities
        if similarities:
            sorted_sims = sorted(similarities)
            # For very few similarities, percentile picks same value — use a
            # fixed floor of 0.5 to avoid degenerate cases
            if len(sorted_sims) <= 2:
                threshold = 0.5
            else:
                idx = max(0, int(len(sorted_sims) * self.config.threshold_percentile / 100.0))
                threshold = sorted_sims[idx]
        else:
            threshold = 0.0

        # Find cut points
        cut_points = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                cut_points.append(i)

        # Build new chunks from cut points
        sentence_chunks = []
        start = 0
        for cut in cut_points + [len(sentences) - 1]:
            end = cut + 1
            combined = "".join(sentences[start:end])
            sentence_chunks.append(Chunk(
                text=combined.strip(),
                metadata=chunk.metadata.copy(),
                chunk_type=chunk.chunk_type,
                section_depth=chunk.section_depth,
            ))
            start = end

        # Split if still too long
        final = []
        for sc in sentence_chunks:
            final.extend(self._split_if_too_long(sc))
        return final

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        """Embed sentences using existing embedding module."""
        all_embeddings = []
        for i in range(0, len(sentences), self.config.batch_size):
            batch = sentences[i:i + self.config.batch_size]
            batch_embeddings = embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences at boundaries."""
        parts = SENTENCE_BOUNDARY.split(text)
        return [p for p in parts if p.strip()]

    def _merge_short(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge consecutive chunks shorter than min_chars if semantically related."""
        merged = []
        for chunk in chunks:
            if merged and len(merged[-1].text) < self.config.min_chars:
                prev = merged.pop()
                # Check semantic similarity before merging
                if self._are_similar(prev.text, chunk.text):
                    combined_text = prev.text + "\n" + chunk.text
                    merged.append(Chunk(
                        text=combined_text,
                        metadata=prev.metadata.copy(),
                        chunk_type=prev.chunk_type,
                        section_depth=prev.section_depth,
                    ))
                else:
                    merged.extend([prev, chunk])
            elif len(chunk.text) < self.config.min_chars and merged:
                last = merged.pop()
                # Check semantic similarity before merging
                if self._are_similar(last.text, chunk.text):
                    combined = last.text + " " + chunk.text
                    merged.append(Chunk(
                        text=combined,
                        metadata=last.metadata.copy(),
                        chunk_type=last.chunk_type,
                        section_depth=last.section_depth,
                    ))
                else:
                    merged.extend([last, chunk])
            else:
                merged.append(chunk)
        return merged

    def _are_similar(self, text_a: str, text_b: str, threshold: float = 0.5) -> bool:
        """Check if two texts are semantically similar enough to merge."""
        try:
            embs = embed_texts([text_a, text_b])
            return cosine_sim(embs[0], embs[1]) >= threshold
        except Exception:
            return True  # On failure, be conservative and merge

    def _split_if_too_long(self, chunk: Chunk) -> list[Chunk]:
        """Hard split at sentence boundary if chunk still exceeds max_chars."""
        if len(chunk.text) <= self.config.max_chars:
            return [chunk]

        sentences = self._split_sentences(chunk.text)
        if len(sentences) <= 1:
            return [chunk]

        result = []
        current = ""
        for sent in sentences:
            if len(current + sent) > self.config.max_chars and current:
                result.append(Chunk(
                    text=current.strip(),
                    metadata=chunk.metadata.copy(),
                    chunk_type=chunk.chunk_type,
                    section_depth=chunk.section_depth,
                ))
                current = sent
            else:
                current += sent
        if current:
            result.append(Chunk(
                text=current.strip(),
                metadata=chunk.metadata.copy(),
                chunk_type=chunk.chunk_type,
                section_depth=chunk.section_depth,
            ))
        if result:
            result[-1].metadata["warning"] = "chunk exceeded max_chars and hard-split"
        return result if result else [chunk]
