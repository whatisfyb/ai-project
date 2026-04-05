# Advanced Text Splitter Design

## Problem

Current `TextSplitter` wraps LangChain's `RecursiveCharacterTextSplitter` with fixed separators. This approach has four fatal flaws:

1. **Semantic truncation** — sentences and complete ideas are split mid-thought, making retrieved chunks contextless.
2. **Information loss** — fixed-length chunks ignore natural boundaries (paragraphs, sections, tables, code blocks), destroying structure and meaning.
3. **Retrieval inaccuracy** — chunks too short lack informat; chunks too long dilute signals with noise.
4. **Lost associations** — cross-referential content ("as mentioned above", "the above experiment") gets split into different chunks and can never be co-retrieved.

## Architecture

Three-stage pipeline, each stage is an independent module with a clear contract. Stages can be used individually or piped together.

```
PDF text → [Stage1: Structure-Aware] → [Stage2: Semantic] → [Stage3: Context Enrich] → Vector Store
```

## Stage 1: Structure-Aware Chunking

**Module**: `utils/structure_chunker.py`

Identifies natural structural boundaries in PDF text through rules and heuristics:

### Heading Detection

Regex patterns that match common heading formats:
- Numbered: `1.1`, `第三章`, `Section 2`, `PART A`
- Style-based: short lines (≤30 chars), all-caps lines, lines without terminal punctuation

Each head becomes a section boundary and is prepended to all chunks within that section as context prefix.

### Special Block Preservation

- Tables: detected by multi-line tabular patterns (pipe-separated or aligned columns)
- Code blocks: markdown fences (`\`\`\``) or indented blocks
- Lists: consecutive lines starting with `-`, `*`, `\d+.`

These special blocks **never** get split. If a block exceeds a soft size limit (configurable, default 3000 chars), it is passed through to Stage 2 for further processing.

### Paragraph Boundary

`\n\n` and runs of whitespace (tabs, multiple spaces) serve as natural paragraph separators.

### Output

A `Chunk` data model:

```python
@dataclass
class Chunk:
    text: str                          # chunk content
    metadata: dict                     # source, section_path, type, position
    section_depth: int = 0             # heading hierarchy depth
    prev_summary: str = ""             # enriched in Stage 3
    next_summary: str = ""             # enriched in Stage 3
    self_contained_score: float = 0.0  # 0-1, enriched in Stage 3
```

The `StructureChunker` takes a `str` (raw PDF text) and returns `list[Chunk]`.

### Config

| Param | Default | Description |
|---|---|---|
| `soft_max_chars` | 3000 | Blocks exceeding this are deferred to Stage 2 |
| `heading_patterns` | (see impl) | Regex list for heading detection |

## Stage 2: Semantic Chunking

**Module**: `utils/semantic_chunker.py`

Takes the output of Stage 1 and further refines chunks using embedding-based semantic similarity.

### Process

1. Sentence splitting via sentence boundary regex (`。`, `.`, `！`, `！`, `？`, `；`, `;`, `\n`)
2. Batch embedding of all sentences using existing `utils/embedding.py` (sentence-transformers, local)
3. Compute pairwise cosine similarity between adjacent sentences
4. Cut points are where similarity falls below the threshold (configurable, default = 25th percentile of similarity distribution for that document)
5. Merge adjacent chunks that are too short (below configurable min_chars, default 100)
6. Re-split chunks that are still too long at their nearest sentence boundary (respects soft_max from config, not a hard cut)

### Output

`list[Chunk]` — same data model, carries forward original metadata.

### Config

| Param | Default | Description |
|---|---|---|
| `min_chars` | 100 | Minimum chunk length (smaller ones get merged) |
| `max_chars` | 1500 | Maximum recommended chunk length |
| `threshold_percentile` | 25 | Cut at similarities below this percentile |
| `batch_size` | 32 | Embedding batch size |

## Stage 3: Context Enrichment

**Logic**: integrated into the Chunk enrichment pipeline, `utils/chunk_enricher.py`

Adds context metadata to each chunk without changing its core text:

### Parent Section Chain

Each chunk inherits its section's full heading hierarchy (e.g. `"第3章 > 3.2 方法论 > 实验设置"`). Stored in `metadata["section_path"]`.

### Adjacent Chunk Summary

For each chunk, generate a 1-2 sentence summary of the previous and next chunk using an LLM (configurable, can also be simple extractive — first sentence). Stored in `prev_summary` and `next_summary`. Default mode: extractive (first/last sentence) for zero cost; LLM mode optional.

### Self-Contained Score

Heuristic 0-1 score evaluating how readable the chunk is in isolation:
- High score: has section heading prefix, no pronoun ambiguity, self-contained claim
- Low score: starts with "however", "therefore", references prior content without context

## Integration with Existing Pipeline

The current pipeline is:
```
Document → TextSplitter.split_documents() → embeddings → vector store
```

New pipeline replaces `TextSplitter` with the three-stage pipeline:
```
PDF text → StructureChunker → SemanticChunker → ChunkEnricher → embeddings → vector store
```

The original `TextSplitter` remains as a fallback option (accessible via `use_legacy_splitter=True` config flag).

## Error Handling

- Empty documents → return empty list, no errors
- Non-PDF text input → Stage 1 still works on raw text (heading detection may have lower accuracy)
- Embedding failures → fall back to structure-only chunking (Stage 2 skip)
- Oversized blocks (>max_chars after all stages) → hard split at sentence boundary with metadata warning

## Testing

- Unit tests: heading detection accuracy, special block preservation, semantic threshold calculation, adjacent summary extraction
- Integration tests: end-to-end pipeline with a sample PDF, verify chunk count, average chunk length, and semantic coherence of boundaries
- Benchmark: compare retrieval accuracy (hits@k) of new vs legacy splitter on a fixed query set
