"""Configuration for the three-stage splitting pipeline."""

from dataclasses import dataclass, field


@dataclass
class StructureChunkConfig:
    """Config for Stage 1: Structure-Aware Chunking."""

    soft_max_chars: int = 3000
    min_heading_length: int = 3
    max_heading_length: int = 30
    separator: str = "\n\n"


@dataclass
class SemanticChunkConfig:
    """Config for Stage 2: Semantic Chunking."""

    min_chars: int = 100
    max_chars: int = 1500
    threshold_percentile: float = 25
    batch_size: int = 32
    embedding_model_fallback: bool = True


@dataclass
class EnrichmentConfig:
    """Config for Stage 3: Context Enrichment."""

    use_extractive_summary: bool = True  # True = extractive (free), False = LLM-based
    separator: str = " > "


@dataclass
class SplitterPipelineConfig:
    """Top-level config for the full pipeline."""

    structure: StructureChunkConfig = field(default_factory=StructureChunkConfig)
    semantic: SemanticChunkConfig = field(default_factory=SemanticChunkConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    use_legacy_splitter: bool = False
