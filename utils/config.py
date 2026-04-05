"""配置管理 - 从 config.yaml 加载所有配置"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


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
    def __init__(
        self,
        name: str = "",
        model: str = ...,
        api_key: str = ...,
        base_url: str = ...,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout


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


class Settings:
    """应用全局配置"""
    def __init__(self):
        data = _load_config()
        self.llm = LLMSettings(data)
        self.embedding = EmbeddingSettings(data)
        self.tavily = TavilySettings(data)
        self.firecrawl = FirecrawlSettings(data)
        self.langsmith = LangSmithSettings(data)
