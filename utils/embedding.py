"""Embedding 工具 - 本地 sentence-transformers 模型"""

from typing import Optional
from sentence_transformers import SentenceTransformer

from utils.config import Settings


_embedding_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """获取 Sentence Transformer 模型"""
    global _embedding_model
    if _embedding_model is None:
        settings = Settings()
        model_name = settings.embedding.model
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_embedding_model():
    """获取 embedding 模型"""
    return _get_model()


def embed_text(text: str) -> list[float]:
    """获取单条文本的 embedding"""
    model = _get_model()
    return model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量获取文本的 embeddings"""
    model = _get_model()
    return model.encode(texts).tolist()


def get_embedding_dimension() -> int:
    """获取 embedding 维度"""
    settings = Settings()
    return settings.embedding.dimension


def reset_embedding_model():
    """重置 Embedding 实例"""
    global _embedding_model
    _embedding_model = None
