"""Token 计量工具

使用 tiktoken 进行本地 token 计算。
"""

from typing import Union

import tiktoken


# 默认使用 cl100k_base（GPT-4/GPT-3.5-turbo 编码）
DEFAULT_ENCODING = "cl100k_base"

# 缓存 encoding 对象
_encoding_cache = {}


def get_encoding(encoding_name: str = DEFAULT_ENCODING):
    """获取 tiktoken encoding 对象（带缓存）"""
    if encoding_name not in _encoding_cache:
        _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoding_cache[encoding_name]


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """计算文本的 token 数量

    Args:
        text: 要计算的文本
        encoding_name: tiktoken 编码名称

    Returns:
        token 数量
    """
    if not text:
        return 0

    encoding = get_encoding(encoding_name)
    return len(encoding.encode(text))


def count_messages_tokens(
    messages: list,
    encoding_name: str = DEFAULT_ENCODING
) -> int:
    """计算消息列表的 token 数量

    参考 OpenAI 的消息格式计算方式：
    - 每条消息有固定的格式开销
    - 角色名称和内容都计入 token

    Args:
        messages: 消息列表，支持 LangChain Message 或 dict 格式
        encoding_name: tiktoken 编码名称

    Returns:
        总 token 数量
    """
    if not messages:
        return 0

    encoding = get_encoding(encoding_name)
    total = 0

    # 每条消息的固定开销（格式化需要额外 token）
    # 参考：https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    TOKENS_PER_MESSAGE = 3  # <|start|>{role}\n{content}<|end|>\n
    TOKENS_PER_NAME = 1    # name 字段如果有

    for msg in messages:
        # 支持 dict 和 LangChain Message 对象
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            name = msg.get("name", "")
        else:
            # LangChain Message 对象
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "") or ""
            name = ""

        # 计算内容的 token
        total += TOKENS_PER_MESSAGE
        total += len(encoding.encode(role))
        total += len(encoding.encode(content))

        if name:
            total += TOKENS_PER_NAME
            total += len(encoding.encode(name))

    # 对话结束标记
    total += 3

    return total


def estimate_max_tokens(
    messages: list,
    model_context_window: int,
    encoding_name: str = DEFAULT_ENCODING,
    reserved_output_tokens: int = 4096,
) -> int:
    """估算可用的最大输出 token 数

    Args:
        messages: 当前消息列表
        model_context_window: 模型的上下文窗口大小
        encoding_name: tiktoken 编码名称
        reserved_output_tokens: 为输出预留的 token 数

    Returns:
        可用的输出 token 数（如果输入已超限，返回 0）
    """
    input_tokens = count_messages_tokens(messages, encoding_name)
    available = model_context_window - input_tokens - reserved_output_tokens
    return max(0, available)


class TokenCounter:
    """Token 计数器（会话级）

    用于追踪会话中的 token 使用情况。
    """

    def __init__(
        self,
        context_window: int = 128000,
        warning_threshold: float = 0.8,
        encoding_name: str = DEFAULT_ENCODING,
    ):
        """初始化

        Args:
            context_window: 模型上下文窗口大小
            warning_threshold: 警告阈值（百分比）
            encoding_name: tiktoken 编码名称
        """
        self.context_window = context_window
        self.warning_threshold = warning_threshold
        self.encoding_name = encoding_name
        self._total_tokens = 0
        self._message_count = 0

    def add_message(self, content: str) -> int:
        """添加消息并返回其 token 数

        Args:
            content: 消息内容

        Returns:
            该消息的 token 数
        """
        tokens = count_tokens(content, self.encoding_name)
        self._total_tokens += tokens
        self._message_count += 1
        return tokens

    def add_messages(self, messages: list) -> int:
        """添加多条消息并返回总 token 数

        Args:
            messages: 消息列表

        Returns:
            总 token 数
        """
        tokens = count_messages_tokens(messages, self.encoding_name)
        self._total_tokens += tokens
        self._message_count += len(messages)
        return tokens

    @property
    def total_tokens(self) -> int:
        """获取总 token 数"""
        return self._total_tokens

    @property
    def usage_ratio(self) -> float:
        """获取使用比例（0-1）"""
        return min(1.0, self._total_tokens / self.context_window)

    @property
    def is_near_limit(self) -> bool:
        """是否接近限制"""
        return self.usage_ratio >= self.warning_threshold

    @property
    def is_over_limit(self) -> bool:
        """是否超过限制"""
        return self._total_tokens >= self.context_window

    def remaining_tokens(self) -> int:
        """获取剩余可用 token"""
        return max(0, self.context_window - self._total_tokens)

    def reset(self):
        """重置计数"""
        self._total_tokens = 0
        self._message_count = 0

    def get_status(self) -> dict:
        """获取状态信息

        Returns:
            包含 token 统计信息的字典
        """
        return {
            "total_tokens": self._total_tokens,
            "message_count": self._message_count,
            "context_window": self.context_window,
            "usage_ratio": round(self.usage_ratio * 100, 1),
            "remaining_tokens": self.remaining_tokens(),
            "is_near_limit": self.is_near_limit,
            "is_over_limit": self.is_over_limit,
        }
