"""Pipeline orchestrator for the three-stage text splitting system."""

from typing import Optional

from utils.chunk_config import SplitterPipelineConfig
from utils.chunk_model import Chunk
from utils.structure_chunker import StructureChunker
from utils.semantic_chunker import SemanticChunker
from utils.chunk_enricher import ChunkEnricher
from utils.text_splitter import TextSplitter


class SplitterPipeline:
    """Orchestrates the three-stage chunking pipeline."""

    def __init__(self, config: Optional[SplitterPipelineConfig] = None):
        self.config = config or SplitterPipelineConfig()

    def run(self, text: str) -> list[Chunk]:
        """Run the full pipeline on raw text.

        Args:
            text: Raw text from PDF extraction.

        Returns:
            List of enriched Chunks ready for embedding storage.
        """
        if not text.strip():
            return []

        if self.config.use_legacy_splitter:
            return self._run_legacy(text)

        return self._run_pipeline(text)

    def _run_pipeline(self, text: str) -> list[Chunk]:
        """Run the three-stage pipeline."""
        # Stage 1: Structure-aware
        stage1 = StructureChunker(config=self.config.structure)
        chunks = stage1.chunk(text)

        if not chunks:
            return []

        # Stage 2: Semantic
        stage2 = SemanticChunker(config=self.config.semantic)
        chunks = stage2.chunk(chunks)

        if not chunks:
            return []

        # Stage 3: Enrich
        stage3 = ChunkEnricher(config=self.config.enrichment)
        chunks = stage3.enrich(chunks)

        return chunks

    def _run_legacy(self, text: str) -> list[Chunk]:
        """Use legacy TextSplitter."""
        splitter = TextSplitter()
        texts = splitter.split_text(text)
        return [Chunk(text=t, metadata={}) for t in texts]
