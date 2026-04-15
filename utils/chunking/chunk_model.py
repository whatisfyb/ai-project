"""Chunk data model for the three-stage splitting pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    """Type of content a chunk contains."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    HEADING = "heading"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_type: ChunkType = field(default=ChunkType.UNKNOWN)
    section_depth: int = 0
    prev_summary: str = ""
    next_summary: str = ""
    self_contained_score: float = 0.0
