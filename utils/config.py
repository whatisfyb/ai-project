"""配置管理 - 从 config.yaml 加载所有配置"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _parse_token_value(value: int | str | None) -> int | None:
    """解析 token 值，支持 K/M 单位

    Examples:
        "200K" -> 200000
        "1M" -> 1000000
        128000 -> 128000
    """
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip().upper()
        multipliers = {
            "K": 1000,
            "M": 1000000,
        }
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                num = float(value[:-1])
                return int(num * mult)
        # 尝试直接解析为数字
        return int(value)

    return None


def _get_config_path() -> Path:
    """获取 config.yaml 路径"""
    config_file = Path("config.yaml")
    if not config_file.is_absolute():
        # 相对于此文件的位置
        config_dir = Path(__file__).parent.parent
        config_file = config_dir / config_file
    return config_file


@lru_cache
def _load_config() -> dict[str, Any]:
    """加载 config.yaml"""
    config_path = _get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> dict[str, Any]:
    """获取全局配置"""
    return _load_config()


class SingleLLMConfig:
    """单个 LLM 模型配置"""

    # 默认上下文窗口大小
    DEFAULT_CONTEXT_WINDOW = 128_000
    DEFAULT_MAX_OUTPUT_TOKENS = 8_192

    def __init__(
        self,
        name: str = "",
        model: str = ...,
        api_key: str = ...,
        base_url: str = ...,
        temperature: float = 0.7,
        timeout: int = 120,
        context_window: int | str | None = None,
        max_output_tokens: int | str | None = None,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self._context_window = _parse_token_value(context_window)
        self._max_output_tokens = _parse_token_value(max_output_tokens)

    @property
    def context_window(self) -> int:
        """上下文窗口大小（tokens）"""
        return self._context_window or self.DEFAULT_CONTEXT_WINDOW

    @property
    def max_output_tokens(self) -> int:
        """最大输出 tokens"""
        return self._max_output_tokens or self.DEFAULT_MAX_OUTPUT_TOKENS


class LLMSettings:
    """LLM 配置"""
    def __init__(self, data: dict):
        self._data = data

    def get_models(self) -> list[SingleLLMConfig]:
        """获取模型列表"""
        models_data = self._data.get("llm", {}).get("models", [])
        if not models_data:
            raise ValueError("未配置任何 LLM 模型，请检查 config.yaml")
        return [SingleLLMConfig(**m) for m in models_data]


class EmbeddingSettings:
    """Embedding 配置"""
    def __init__(self, data: dict):
        self._data = data.get("embedding", {})

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")

    @property
    def base_url(self) -> str:
        return self._data.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    @property
    def model(self) -> str:
        return self._data.get("model", "text-embedding-v4")

    @property
    def dimension(self) -> int:
        return self._data.get("dimension", 1536)


class TavilySettings:
    """Tavily 配置"""
    def __init__(self, data: dict):
        self._data = data.get("tavily", {})

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")


class FirecrawlSettings:
    """Firecrawl 配置"""
    def __init__(self, data: dict):
        self._data = data.get("firecrawl", {})

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")


class LangSmithSettings:
    """LangSmith 配置"""
    def __init__(self, data: dict):
        self._data = data.get("langsmith", {})

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")

    @property
    def tracing(self) -> bool:
        return self._data.get("tracing", False)

    @property
    def project(self) -> str:
        return self._data.get("project", "default")


class CompactSettings:
    """上下文压缩配置"""
    def __init__(self, data: dict):
        self._data = data.get("compact", {})

    @property
    def auto_enabled(self) -> bool:
        """是否启用自动压缩"""
        return self._data.get("auto_enabled", True)

    @property
    def threshold_pct(self) -> float:
        """触发阈值（百分比）"""
        return self._data.get("threshold_pct", 0.80)

    @property
    def keep_recent(self) -> int:
        """保留最近消息数"""
        return self._data.get("keep_recent", 10)

    @property
    def buffer_tokens(self) -> int:
        """缓冲区 tokens（支持 K/M 单位）"""
        value = self._data.get("buffer_tokens", 10_000)
        return _parse_token_value(value) or 10_000

    @property
    def output_reserve(self) -> int:
        """输出预留 tokens（支持 K/M 单位）"""
        value = self._data.get("output_reserve", 8_000)
        return _parse_token_value(value) or 8_000


class Settings:
    """应用全局配置"""

    def __init__(self):
        data = _load_config()
        self.llm = LLMSettings(data)
        self.embedding = EmbeddingSettings(data)
        self.tavily = TavilySettings(data)
        self.firecrawl = FirecrawlSettings(data)
        self.langsmith = LangSmithSettings(data)
        self.compact = CompactSettings(data)


# ============ 上下文管理 ============

class ContextThresholds:
    """上下文阈值配置"""

    def __init__(
        self,
        context_window: int,
        max_output_tokens: int,
        buffer_tokens: int = 10_000,
        warning_pct: float = 0.80,
        compact_pct: float = 0.90,
        block_pct: float = 0.95,
    ):
        # 有效窗口 = 总窗口 - 输出预留 - 缓冲区
        self.effective_window = context_window - max_output_tokens - buffer_tokens
        self.warning_threshold = int(self.effective_window * warning_pct)
        self.compact_threshold = int(self.effective_window * compact_pct)
        self.block_threshold = int(self.effective_window * block_pct)


class ContextStatus:
    """上下文状态"""

    def __init__(self, current_tokens: int, thresholds: ContextThresholds):
        self.current_tokens = current_tokens
        self.thresholds = thresholds

    @property
    def percent_used(self) -> float:
        """使用百分比"""
        return min(100, (self.current_tokens / self.thresholds.effective_window) * 100)

    @property
    def tokens_remaining(self) -> int:
        """剩余 tokens"""
        return max(0, self.thresholds.effective_window - self.current_tokens)

    @property
    def should_warn(self) -> bool:
        """是否应该警告"""
        return self.current_tokens >= self.thresholds.warning_threshold

    @property
    def should_compact(self) -> bool:
        """是否应该压缩"""
        return self.current_tokens >= self.thresholds.compact_threshold

    @property
    def is_blocked(self) -> bool:
        """是否阻塞（必须压缩）"""
        return self.current_tokens >= self.thresholds.block_threshold

    def get_action(self) -> str:
        """获取建议动作"""
        if self.is_blocked:
            return "block"
        elif self.should_compact:
            return "compact"
        elif self.should_warn:
            return "warn"
        return "ok"


def get_context_thresholds(model_config: SingleLLMConfig) -> ContextThresholds:
    """获取模型的上下文阈值"""
    return ContextThresholds(
        context_window=model_config.context_window,
        max_output_tokens=model_config.max_output_tokens,
    )


def check_context_status(current_tokens: int, model_config: SingleLLMConfig) -> ContextStatus:
    """检查上下文状态"""
    thresholds = get_context_thresholds(model_config)
    return ContextStatus(current_tokens, thresholds)


# ============ 便捷函数 ============

@lru_cache
def get_settings_instance() -> Settings:
    """获取全局 Settings 实例（缓存）"""
    return Settings()


def get_default_model_config() -> SingleLLMConfig:
    """获取默认模型配置（第一个配置的模型）"""
    settings = get_settings_instance()
    models = settings.llm.get_models()
    return models[0] if models else SingleLLMConfig()
