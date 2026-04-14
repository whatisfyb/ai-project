"""模型预加载模块"""

from utils.embedding import preload_embedding_model
from utils.reranker import preload_reranker_model


def preload_models(embedding: bool = True, reranker: bool = True) -> dict[str, bool]:
    """预加载所有模型

    Args:
        embedding: 是否预加载 embedding 模型
        reranker: 是否预加载 reranker 模型

    Returns:
        各模型加载状态
    """
    results = {}

    if embedding:
        print("正在预加载 Embedding 模型...")
        results["embedding"] = preload_embedding_model()
        print(f"  Embedding 模型: {'成功' if results['embedding'] else '失败'}")

    if reranker:
        print("正在预加载 Reranker 模型...")
        results["reranker"] = preload_reranker_model()
        print(f"  Reranker 模型: {'成功' if results['reranker'] else '失败'}")

    return results


def is_models_loaded() -> dict[str, bool]:
    """检查模型是否已加载"""
    from utils.embedding import _embedding_model
    from utils.reranker import _reranker_model

    return {
        "embedding": _embedding_model is not None,
        "reranker": _reranker_model is not None,
    }
