"""Developer utilities

This module provides backward-compatible imports for all utilities.
All imports are re-exported from their new locations in submodules.
"""

# 全局配置 LangSmith（在任何 LLM 调用之前）
from utils.core.langsmith import configure_langsmith
configure_langsmith()

# ============ Core ============
from utils.core.config import Settings
from utils.core.llm import get_llm_model, get_all_llm_models, reset_llm_models
from utils.core.embedding import (
    get_embedding_model,
    embed_text,
    embed_texts,
    get_embedding_dimension,
    reset_embedding_model,
)

# ============ Protocol ============
from utils.protocol.jsonrpc import (
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCErrorCodes,
    JSONRPC_VERSION,
    make_request,
    make_success_response,
    make_error_response,
)

# ============ Retrieval ============
from utils.retrieval.vector_store import VectorStore
from utils.retrieval.retriever import Retriever, create_retriever, MultiRetriever

# ============ Chunking ============
from utils.chunking.chunk_model import Chunk, ChunkType
from utils.chunking.text_splitter import TextSplitter, create_default_splitter, create_pipeline_splitter
from utils.chunking.chunk_pipeline import SplitterPipeline

# ============ Document ============
from utils.document.file_loader import FileLoader

__all__ = [
    # Core - Config
    "Settings",
    # Core - LLM
    "get_llm_model",
    "get_all_llm_models",
    "reset_llm_models",
    # Core - Embedding
    "get_embedding_model",
    "embed_text",
    "embed_texts",
    "get_embedding_dimension",
    "reset_embedding_model",
    # Core - LangSmith
    "configure_langsmith",
    # Protocol - JSON-RPC
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCErrorCodes",
    "JSONRPC_VERSION",
    "make_request",
    "make_success_response",
    "make_error_response",
    # Retrieval
    "VectorStore",
    "Retriever",
    "create_retriever",
    "MultiRetriever",
    # Chunking
    "Chunk",
    "ChunkType",
    "TextSplitter",
    "create_default_splitter",
    "create_pipeline_splitter",
    "SplitterPipeline",
    # Document
    "FileLoader",
]
