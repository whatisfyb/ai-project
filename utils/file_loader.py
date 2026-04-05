"""通用文件加载器 - 使用 LangChain Document Loader"""

from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
)
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader


class FileLoader:
    """通用文件加载器"""

    # LangChain loader 映射
    _LOADERS = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".json": TextLoader,
        ".yaml": TextLoader,
        ".yml": TextLoader,
        ".xml": TextLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".csv": CSVLoader,
        ".toml": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

    # 简单文本类型 (无需特殊 loader)
    _TEXT_TYPES = {".txt", ".md", ".json", ".yaml", ".yml", ".xml", ".html", ".htm"}

    @classmethod
    def load(cls, file_path: str) -> str:
        """加载文件内容

        Args:
            file_path: 文件路径

        Returns:
            文件文本内容
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        if suffix not in cls._LOADERS:
            raise ValueError(f"不支持的文件类型: {suffix}。支持: {cls.supported_types()}")

        loader_cls = cls._LOADERS[suffix]

        # TextLoader 和 CSVLoader 需要 encoding 参数
        if suffix in cls._TEXT_TYPES and loader_cls == TextLoader:
            loader = loader_cls(str(path), encoding="utf-8")
        elif suffix == ".csv":
            loader = loader_cls(str(path), encoding="utf-8")
        else:
            loader = loader_cls(str(path))

        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    @classmethod
    def load_documents(cls, file_path: str) -> list:
        """加载文件返回 LangChain Document 对象

        Args:
            file_path: 文件路径

        Returns:
            LangChain Document 列表
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        if suffix not in cls._LOADERS:
            raise ValueError(f"不支持的文件类型: {suffix}。支持: {cls.supported_types()}")

        loader_cls = cls._LOADERS[suffix]

        if suffix in cls._TEXT_TYPES and loader_cls == TextLoader:
            loader = loader_cls(str(path), encoding="utf-8")
        elif suffix == ".csv":
            loader = loader_cls(str(path), encoding="utf-8")
        else:
            loader = loader_cls(str(path))

        return loader.load()

    @classmethod
    def supported_types(cls) -> list[str]:
        """获取支持的文件类型列表"""
        return sorted(cls._LOADERS.keys())