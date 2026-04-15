"""文本分块器"""

from typing import TYPE_CHECKING, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.core.config import Settings

if TYPE_CHECKING:
    from utils.chunking.chunk_pipeline import SplitterPipeline


class TextSplitter:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[list[str]] = None,
    ):
        """初始化分块器

        Args:
            chunk_size: 每个 chunk 的最大字符数
            chunk_overlap: chunk 之间的重叠字符数
            separators: 分隔符列表，按优先级排序
        """
        settings = Settings()

        self.chunk_size = chunk_size or getattr(settings, "chunk_size", 1000)
        self.chunk_overlap = chunk_overlap or getattr(settings, "chunk_overlap", 200)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        """分块文本

        Args:
            text: 输入文本

        Returns:
            chunk 列表
        """
        return self._splitter.split_text(text)

    def split_documents(self, documents: list) -> list:
        """分块 LangChain Document 对象列表

        Args:
            documents: LangChain Document 列表

        Returns:
            分块后的 Document 列表
        """
        return self._splitter.split_documents(documents)


def create_default_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> TextSplitter:
    """创建默认分块器

    Args:
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: chunk 之间的重叠字符数

    Returns:
        TextSplitter 实例
    """
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_pipeline_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_legacy: bool = False,
) -> "SplitterPipeline":
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
        SplitterPipelineConfig,
        StructureChunkConfig,
    )
    from utils.chunk_pipeline import SplitterPipeline

    config = SplitterPipelineConfig(
        structure=StructureChunkConfig(soft_max_chars=chunk_size * 3),
        semantic=SemanticChunkConfig(max_chars=chunk_size),
        use_legacy_splitter=use_legacy,
    )
    return SplitterPipeline(config=config)