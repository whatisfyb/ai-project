# Advanced Text Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-length `TextSplitter` with a three-stage pipeline (structure-aware → semantic → context-enriched) that preserves semantic boundaries and improves retrieval accuracy.

**Architecture:** Three independent `Chunker` classes operating on a shared `Chunk` data model, each adding/refining chunk boundaries. Can be used individually or piped together via a `Pipeline` class.

**Tech Stack:** Python, regex, sentence-transformers (local embedding), existing `utils/embedding.py`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `utils/chunk_model.py` | Create | `Chunk` dataclass and `ChunkType` enum |
| `utils/structure_chunker.py` | Create | Structure-aware chunking (headings, paragraphs, special blocks) |
| `utils/semantic_chunker.py` | Create | Embedding-based semantic chunking |
| `utils/chunk_enricher.py` | Create | Context enrichment (section chain, adjacent summaries, self-contained score) |
| `utils/chunk_pipeline.py` | Create | Pipeline orchestrator chaining the three stages |
| `utils/chunk_config.py` | Create | Configuration classes for all stages |
| `utils/text_splitter.py` | Modify | Add `use_legacy_splitter` flag, integrate pipeline |
| `config.yaml` | Modify | Add splitter config section |

---

### Task 1: Chunk Data Model + Config

**Files:**
- Create: `utils/chunk_model.py`
- Create: `utils/chunk_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunk_model.py`:

```python
"""Tests for Chunk data model and config."""

import pytest

from utils.chunk_model import Chunk, ChunkType


def test_chunk_creation():
    chunk = Chunk(text="hello world", metadata={"source": "test.pdf"})
    assert chunk.text == "hello world"
    assert chunk.metadata["source"] == "test.pdf"
    assert chunk.prev_summary == ""
    assert chunk.self_contained_score == 0.0


def test_chunk_type_enum():
    assert ChunkType.PARAGRAPH.value == "paragraph"
    assert ChunkType.TABLE.value == "table"
    assert ChunkType.CODE.value == "code"
    assert ChunkType.HEADING.value == "heading"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunk_model.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'utils.chunk_model'"

- [ ] **Step 3: Write minimal implementation**

Create `utils/chunk_model.py`:

```python
"""Chunk data model for the three-stage splitting pipeline."""

from dataclasses import dataclass, field
from enum import Enum


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
    metadata: dict = field(default_factory=dict)
    chunk_type: "ChunkType" = field(default=ChunkType.UNKNOWN)
    section_depth: int = 0
    prev_summary: str = ""
    next_summary: str = ""
    self_contained_score: float = 0.0
```

- [ ] **Step 4: Add chunk config**

Create `utils/chunk_config.py`:

```python
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
```

- [ ] **Step 5: Write config test**

Add to `tests/test_chunk_model.py`:

```python
from utils.chunk_config import SplitterPipelineConfig, SemanticChunkConfig


def test_default_config():
    config = SplitterPipelineConfig()
    assert config.structure.soft_max_chars == 3000
    assert config.semantic.max_chars == 1500
    assert config.enrichment.use_extractive_summary is True
    assert config.use_legacy_splitter is False


def test_semantic_config_override():
    config = SplitterPipelineConfig(
        semantic=SemanticChunkConfig(min_chars=200, max_chars=2000)
    )
    assert config.semantic.min_chars == 200
    assert config.semantic.max_chars == 2000
```

- [ ] **Step 6: Run all tests and verify pass**

Run: `python -m pytest tests/test_chunk_model.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add utils/chunk_model.py utils/chunk_config.py tests/test_chunk_model.py
git commit -m "feat: add Chunk data model and pipeline config"
```

---

### Task 2: Structure-Aware Chunker

**Files:**
- Create: `utils/structure_chunker.py`
- Modify: `utils/__init__.py` (add new exports)

- [ ] **Step 1: Write the failing test**

Create `tests/test_structure_chunker.py`:

```python
"""Tests for structure-aware chunking."""

from utils.chunk_model import ChunkType
from utils.structure_chunker import StructureChunker


def test_paragraph_splitting():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[1].text


def test_heading_creates_section():
    text = "第一章 引言\n\n这是引言的内容。\n\n第二章 方法\n\n这是方法部分。"
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    section_paths = [c.metadata.get("section_path", "") for c in chunks]
    assert any("第一章" in p for p in section_paths)
    assert any("第二章" in p for p in section_paths)


def test_code_block_not_split():
    text = "```\ndef foo():\n    pass\n```\n"
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "```" in chunks[0].text


def test_table_not_split():
    text = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "|" in chunks[0].text


def test_empty_input():
    chunker = StructureChunker()
    chunks = chunker.chunk("")
    assert chunks == []


def test_heading_kept_with_content():
    text = "第一章 引言\n\n这是引言的内容。\n"
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert "第一章" in chunks[0].text
    assert "这是引言的内容" in chunks[0].text


def test_numbered_heading_pattern():
    text = "1.1 背景介绍\n\n这是背景。"
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert any("1.1" in c.metadata.get("section_path", "") for c in chunks)


def test_chinese_chapter_heading():
    text = "第三章 方法论\n\n本章讨论方法。"
    chunker = StructureChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert any("第三章" in c.metadata.get("section_path", "") for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_structure_chunker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'utils.structure_chunker'"

- [ ] **Step 3: Write minimal implementation**

Create `utils/structure_chunker.py`:

```python
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
                # Update section path
                current_path = self._update_path(current_path, heading)

            sub_chunks = self._split_section(section_text, heading)
            for sc in sub_chunks:
                sc.metadata["section_path"] = self.config.separator.join(current_path) if current_path else ""

            chunks.extend(sub_chunks)

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Split text into (content, heading) pairs.

        Returns list of (section_text, heading) tuples.
        The first heading may be None if text starts before any heading.
        """
        lines = text.split("\n")
        sections: list[tuple[str, str | None]] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in lines:
            if self._is_heading(line.strip()):
                if current_lines:
                    sections.append(("\n".join(current_lines), current_heading))
                    current_lines = []
                current_heading = line.strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(("\n".join(current_lines), current_heading))

        # Ensure at least one section
        if not sections:
            sections.append((text, None))

        return sections

    def _update_path(self, current_path: list[str], heading: str) -> list[str]:
        """Update section path based heading."""
        # A heading at depth 0 starts new top level
        # Numbered headings: use the number to determine depth
        number_match = re.search(r"(\d+(?:\.\d+)*)", heading)
        if number_match:
            depth = number_match.group(1).count(".")
            current_path = current_path[:depth] + [heading]
        else:
            # Chinese chapter heading (第X章) — always top level
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
                continue  # Headings are handled specially, not as separate chunks

            # Prepend heading to first paragraph of section for context
            prefix = heading_str + "\n" if (heading_str and not chunks) else ""
            chunk_text = prefix + stripped

            if len(chunk_text) > self.config.soft_max_chars:
                # Defer to semantic chunker for further splitting
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
        return any(p.match(stripped) for p in HEADING_PATTERNS)

    def _is_table(self, text: str) -> bool:
        """Check if text contains a table pattern."""
        return bool(TABLE_PATTERN.search(text))

    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code fence block."""
        return text.startswith("```") and text.endswith("```")
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_structure_chunker.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/structure_chunker.py tests/test_structure_chunker.py
git commit -m "feat: add structure-aware chunker (Stage 1)"
```

---

### Task 3: Semantic Chunker

**Files:**
- Create: `utils/semantic_chunker.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_semantic_chunker.py`:

```python
"""Tests for Semantic Chunking (Stage 2)."""

import pytest

from utils.chunk_model import Chunk, ChunkType
from utils.semantic_chunker import SemanticChunker


def _make_chunk(text: str) -> Chunk:
    return Chunk(text=text, metadata={"source": "test.pdf"})


def test_semantic_chunk_same_topic():
    """Similar adjacent sentences should stay together."""
    text = "The model uses a transformer architecture. " \
           "It has 12 layers with 12 attention heads each. " \
           "Training took 300 epochs on 8 GPUs."
    chunk = _make_chunk(text)
    chunker = SemanticChunker()
    chunks = list(chunker.chunk([chunk]))
    # These sentences are topically similar, should not be split much
    assert len(chunks) <= 3


def test_semantic_cross_topic():
    """Chunks on different topics should be split apart."""
    text = "The experiment shows 95% accuracy on the test set. " \
           "The weather in Shanghai is very hot in summer."
    chunk = _make_chunk(text)
    chunker = SemanticChunker()
    chunks = list(chunker.chunk([chunk]))
    assert len(chunks) >= 2


def test_empty_input():
    chunker = SemanticChunker()
    chunks = list(chunker.chunk([]))
    assert chunks == []


def test_chunk_too_long_gets_split():
    """Chunks exceeding max_chars should be further split."""
    long_text = "。".join(["这是一个句子"] * 500)  # ~2000 chars
    chunk = _make_chunk(long_text)
    chunker = SemanticChunker(max_chars=1000)
    chunks = list(chunker.chunk([chunk]))
    # Should be split into multiple chunks
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 1000


def test_short_chunks_get_merged():
    """Chunks shorter than min_chars should be merged."""
    short_text = "短文本。"  # Very short
    chunk = _make_chunk(short_text)
    chunker = SemanticChunker(min_chars=100)
    chunks = list(chunker.chunk([chunk]))
    # Short chunks get merged — in this single-chunk case, no merge partner
    # so it stays as-is
    assert len(chunks) == 1


def test_cosine_similarity_calculation():
    """Verify cosine_sim returns values in [-1, 1]."""
    from utils.semantic_chunker import cosine_sim
    a = [0.1, 0.2, 0.3]
    b = [0.1, 0.2, 0.3]  # Identical → 1.0
    c = [0.0, 0.0, 0.0]
    assert abs(cosine_sim(a, b) - 1.0) < 1e-6
    # Zero vector vs anything → handle gracefully
    result = cosine_sim(a, c)
    assert isinstance(result, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_chunker.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Create `utils/semantic_chunker.py`:

```python
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

    def __init__(self, config: Optional[SemanticChunkConfig] = None):
        self.config = config or SemanticChunkConfig()

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
        if len(text) <= self.config.min_chars:
            return [chunk]

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
        """Merge consecutive chunks shorter than min_chars."""
        merged = []
        for chunk in chunks:
            if merged and len(merged[-1].text) < self.config.min_chars:
                prev = merged.pop()
                combined_text = prev.text + "\n" + chunk.text
                merged.append(Chunk(
                    text=combined_text,
                    metadata=prev.metadata.copy(),
                    chunk_type=prev.chunk_type,
                    section_depth=prev.section_depth,
                ))
            elif len(chunk.text) < self.config.min_chars and merged:
                last = merged.pop()
                combined = last.text + " " + chunk.text
                merged.append(Chunk(
                    text=combined,
                    metadata=last.metadata.copy(),
                    chunk_type=last.chunk_type,
                    section_depth=last.section_depth,
                ))
            else:
                merged.append(chunk)
        return merged

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
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_semantic_chunker.py -v`
Expected: All 6 tests PASS

Note: Semantic tests require the embedding model to be loaded, which may take 5-10s on first run.

- [ ] **Step 5: Commit**

```bash
git add utils/semantic_chunker.py tests/test_semantic_chunker.py
git commit -m "feat: add semantic chunker (Stage 2)"
```

---

### Task 4: Context Enricher

**Files:**
- Create: `utils/chunk_enricher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunk_enricher.py`:

```python
"""Tests for Context Enrichment (Stage 3)."""

import pytest

from utils.chunk_model import Chunk
from utils.chunk_enricher import ChunkEnricher


def test_section_chain_metadata():
    chunk = Chunk(
        text="实验设置如下",
        metadata={"section_path": "第3章 > 3.2 方法论 > 实验设置"},
    )
    enricher = ChunkEnricher()
    enriched = enricher.enrich([chunk])
    assert enriched[0].metadata["section_path"] == "第3章 > 3.2 方法论 > 实验设置"


def test_adjacent_extracts_first_last_sentence():
    """Extractive mode should use first/last sentences of adjacent chunks."""
    chunks = [
        Chunk(text="第一段。这是第一段的第二句。这是第三句。", metadata={}),
        Chunk(text="第二段。中间句。尾句。", metadata={}),
        Chunk(text="第三段。结尾句。", metadata={}),
    ]
    enricher = ChunkEnricher()
    enriched = enricher.enrich(chunks)
    # Middle chunk should have prev_summary and next_summary
    assert enriched[1].prev_summary != ""
    assert enriched[1].next_summary != ""
    assert "第一段" in enriched[1].prev_summary
    assert "第三段" in enriched[1].next_summary


def test_first_chunk_has_no_prev():
    chunks = [Chunk(text="only chunk", metadata={})]
    enricher = ChunkEnricher()
    enriched = enricher.enrich(chunks)
    assert enriched[0].prev_summary == ""


def test_self_contained_score_high():
    """Chunk with heading prefix and no ambiguous pronouns should score high."""
    chunk = Chunk(
        text="第一章 概述\n\n本文提出了一种新方法。",
        metadata={"section_path": "第一章"},
    )
    enricher = ChunkEnricher()
    result = enricher.enrich([chunk])
    assert result[0].self_contained_score >= 0.7


def test_self_contained_score_low():
    """Chunk starting with 'However' without context should score low."""
    chunk = Chunk(text="然而，如上所述，这个方法是有效的。", metadata={})
    enricher = ChunkEnricher()
    result = enricher.enrich([chunk])
    assert result[0].self_contained_score < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunk_enricher.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Create `utils/chunk_enricher.py`:

```python
"""Context enrichment for chunks (Stage 3)."""

import re
from typing import Optional
from utils.chunk_config import EnrichmentConfig
from utils.chunk_model import Chunk

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
            return sentences[-1]
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
            if DEPENDENCY_MARKERS.match(text.split("\n")[-1] if "\n" in text else text):
                score -= 0.3

            # Very short → lower score (incomplete thought)
            if len(text) < 50:
                score -= 0.2

            # Long and self-describing → higher score
            if len(text) > 200:
                score += 0.1

            chunk.self_contained_score = max(0.0, min(1.0, score))
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_chunk_enricher.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/chunk_enricher.py tests/test_chunk_enricher.py
git commit -m "feat: add context enricher (Stage 3)"
```

---

### Task 5: Pipeline Orchestrator

**Files:**
- Create: `utils/chunk_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunk_pipeline.py`:

```python
"""Integration tests for the full chunking pipeline."""

import pytest

from utils.chunk_model import Chunk
from utils.chunk_pipeline import SplitterPipeline
from utils.chunk_config import SplitterPipelineConfig


def test_pipeline_end_to_end():
    """Full pipeline should process text and return enriched chunks."""
    text = "第一章 引言\n\n这是一个关于机器学习的介绍。\n\n第二章 方法\n\n我们使用了深度学习的方法。"
    pipeline = SplitterPipeline()
    chunks = pipeline.run(text)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "section_path" in chunk.metadata


def test_pipeline_legacy_mode():
    """Legacy splitter mode should use TextSplitter."""
    text = "First paragraph.\n\nSecond paragraph."
    config = SplitterPipelineConfig(use_legacy_splitter=True)
    pipeline = SplitterPipeline(config=config)
    chunks = pipeline.run(text)
    assert len(chunks) >= 1


def test_pipeline_empty_input():
    pipeline = SplitterPipeline()
    chunks = pipeline.run("")
    assert chunks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunk_pipeline.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Create `utils/chunk_pipeline.py`:

```python
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
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_chunk_pipeline.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/chunk_pipeline.py tests/test_chunk_pipeline.py
git commit -m "feat: add splitter pipeline orchestrator"
```

---

### Task 6: Integrate with Existing TextSplitter + Update Config

**Files:**
- Modify: `utils/text_splitter.py`
- Modify: `config.yaml`
- Modify: `utils/__init__.py`

- [ ] **Step 1: Update text_splitter.py**

The existing `TextSplitter` class should remain usable. Add a `create_pipeline_splitter` factory function that returns a `SplitterPipeline` instance.

Add these imports and factory function at the end of `utils/text_splitter.py`:

```python
from utils.chunk_pipeline import SplitterPipeline
from utils.chunk_config import SplitterPipelineConfig
from utils.chunk_model import Chunk


def create_pipeline_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_legacy: bool = False,
) -> SplitterPipeline:
    """Create a pipeline-based splitter.

    Args:
        chunk_size: Target chunk size (used for semantic config max_chars).
        chunk_overlap: Overlap hint (semantic chunker uses overlap via merging).
        use_legacy: If True, use the legacy TextSplitter.

    Returns:
        SplitterPipeline instance.
    """
    from utils.chunk_config import (
        SemanticChunkConfig,
        StructureChunkConfig,
    )

    config = SplitterPipelineConfig(
        structure=StructureChunkConfig(soft_max_chars=chunk_size * 3),
        semantic=SemanticChunkConfig(max_chars=chunk_size),
        use_legacy_splitter=use_legacy,
    )
    return SplitterPipeline(config=config)
```

- [ ] **Step 2: Add splitter config to config.yaml**

Add this section to `config.yaml` (after embedding, before tavily):

```yaml
# Text Splitter Configuration
splitter:
  use_legacy: false
  chunk_size: 1000
  chunk_overlap: 200
  soft_max_chars: 3000
  semantic_threshold_percentile: 25
  min_chunk_chars: 100
  extractive_summary: true
```

- [ ] **Step 3: Update utils/__init__.py**

Add exports for the new modules. Read the current file first, then add export lines for:
- `Chunk`, `ChunkType` from `utils.chunk_model`
- `SplitterPipeline` from `utils.chunk_pipeline`
- `create_pipeline_splitter` from `utils.text_splitter`

- [ ] **Step 4: Verify no existing code breaks**

Run: `python -c "from utils.text_splitter import TextSplitter; print('OK')"`
Expected: prints "OK"

- [ ] **Step 5: Commit**

```bash
git add utils/text_splitter.py config.yaml utils/__init__.py
git commit -m "feat: integrate pipeline splitter with existing TextSplitter"
```

---