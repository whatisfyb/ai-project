"""Structure-aware text chunking (Stage 1 of the pipeline)."""

import re
from typing import Optional

from utils.chunk_model import Chunk, ChunkType
from utils.chunk_config import StructureChunkConfig

# Heading patterns, ordered by priority
HEADING_PATTERNS = [
    # Chapter: 第一章, 第2章
    re.compile(r"^(第[一二三四五六七八九十百\d]+[章篇])\s*(.*)"),
    # Numbered: 1.1, 1.2.3, Section 2
    re.compile(r"^(\d+(?:\.\d+)+)\s+([^\n]+)"),
    # ALL CAPS short line
    re.compile(r"^([A-Z][A-Z ]{2,29})$"),
    # Short line without terminal punctuation (except in CJK)
    re.compile(r"^([^。！？.!?]{3,30})$"),
]

# Table detection: pipe-separated or aligned columns
TABLE_PATTERN = re.compile(r"(?:\|[^|]*\|\n?){2,}")

# Code block: triple backticks
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")


class StructureChunker:
    """Splits text into chunks respecting natural document structure."""

    def __init__(self, config: Optional[StructureChunkConfig] = None):
        self.config = config or StructureChunkConfig()

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into structurally-aware chunks.

        Args:
            text: Raw text (typically from PDF extraction).

        Returns:
            List of Chunk objects with metadata.
        """
        if not text.strip():
            return []

        sections = self._split_by_headings(text)
        chunks = []
        current_path = []

        for section_text, heading in sections:
            if heading:
                current_path = self._update_path(current_path, heading)

            sub_chunks = self._split_section(section_text, heading)
            for sc in sub_chunks:
                sc.metadata["section_path"] = self.config.separator.join(current_path) if current_path else ""

            chunks.extend(sub_chunks)

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, Optional[str]]]:
        """Split text into (content, heading) pairs.

        Returns list of (section_text, heading) tuples.
        The first heading may be None if text starts before any heading.
        """
        lines = text.split("\n")
        sections: list[tuple[str, Optional[str]]] = []
        current_heading: Optional[str] = None
        current_lines: list[str] = []
        in_code_fence: bool = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                current_lines.append(line)
                continue

            if not in_code_fence and self._is_heading(stripped):
                if current_lines:
                    sections.append(("\n".join(current_lines), current_heading))
                    current_lines = []
                current_heading = stripped
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(("\n".join(current_lines), current_heading))

        if not sections:
            sections.append((text, None))

        return sections

    def _update_path(self, current_path: list[str], heading: str) -> list[str]:
        """Update section path based on heading."""
        number_match = re.search(r"(\d+(?:\.\d+)*)", heading)
        if number_match:
            depth = number_match.group(1).count(".")
            current_path = current_path[:depth] + [heading]
        else:
            if re.match(r"第[一二三四五六七八九十\d]+[章篇]", heading):
                current_path = [heading]
            else:
                current_path.append(heading)
        return current_path

    def _split_section(self, text: str, heading: Optional[str]) -> list[Chunk]:
        """Split a section into paragraphs, preserving special blocks."""
        paragraphs = self._split_paragraphs(text)
        chunks = []
        heading_str = heading or ""

        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue

            chunk_type = self._detect_type(stripped)

            if chunk_type == ChunkType.HEADING:
                continue

            prefix = heading_str + "\n" if (heading_str and not chunks) else ""
            chunk_text = prefix + stripped

            if len(chunk_text) > self.config.soft_max_chars:
                chunk_type = ChunkType.UNKNOWN

            chunks.append(Chunk(
                text=chunk_text,
                metadata={},
                chunk_type=chunk_type,
            ))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs on double newlines."""
        parts = text.split(self.config.separator)
        return [p for p in parts if p.strip()]

    def _detect_type(self, text: str) -> ChunkType:
        """Detect the type of a text block."""
        if self._is_heading(text):
            return ChunkType.HEADING
        if self._is_table(text):
            return ChunkType.TABLE
        if self._is_code_block(text):
            return ChunkType.CODE
        return ChunkType.PARAGRAPH

    def _is_heading(self, text: str) -> bool:
        """Check if text is a heading."""
        stripped = text.strip()
        if len(stripped) < self.config.min_heading_length:
            return False
        if len(stripped) > self.config.max_heading_length:
            return False
        if any(stripped.endswith(p) for p in "。！？."):
            return False
        if stripped.startswith("```") or stripped.startswith("|"):
            return False
        return any(p.match(stripped) for p in HEADING_PATTERNS)

    def _is_table(self, text: str) -> bool:
        """Check if text contains a table pattern."""
        return bool(TABLE_PATTERN.search(text))

    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code fence block."""
        return text.startswith("```") and text.endswith("```")
