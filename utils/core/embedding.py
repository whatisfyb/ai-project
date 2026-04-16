"""Embedding 工具 - 本地 sentence-transformers 模型

注意：sentence_transformers 依赖 torch，可能导致循环导入。
因此使用延迟导入，只在实际调用时才导入。
"""

from typing import Optional

from utils.core.config import Settings


_embedding_model: Optional["SentenceTransformer"] = None


def _get_model() -> "SentenceTransformer":
    """获取 Sentence Transformer 模型（延迟导入）"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
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


def preload_embedding_model() -> bool:
    """预加载 Embedding 模型

    Returns:
        是否加载成功
    """
    try:
        _get_model()
        # 预热：做一次 dummy encode
        _embedding_model.encode("warmup")
        return True
    except Exception as e:
        print(f"Embedding 模型预加载失败: {e}")
        return False
