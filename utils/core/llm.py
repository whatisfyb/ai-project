"""LLM 工厂 - 支持多模型实例"""

from typing import Optional

from langchain_openai import ChatOpenAI

from utils.core.config import Settings, SingleLLMConfig


_llm_instances: dict[str, ChatOpenAI] = {}
_settings_instance: Settings | None = None


def _get_settings() -> Settings:
    """获取 Settings 单例"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def _create_llm(config: SingleLLMConfig) -> ChatOpenAI:
    """根据配置创建 LLM 实例"""
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        timeout=config.timeout,
    )


def get_llm_model(name: Optional[str] = None) -> ChatOpenAI:
    """获取 LLM 模型实例

    Args:
        name: 模型标识名 (可选)
              - 不填: 返回配置中的第一个模型
              - 填了: 返回对应名称的模型

    Returns:
        ChatOpenAI 实例

    Raises:
        ValueError: 未配置任何模型时抛出
        KeyError: 指定名称的模型不存在
    """
    global _llm_instances

    settings = _get_settings()
    models = settings.llm.get_models()

    if not models:
        raise ValueError("未配置任何 LLM 模型，请检查 config.yaml")

    if not name:
        # 返回配置中的第一个模型
        name = _get_model_key(models[0])

    if name in _llm_instances:
        return _llm_instances[name]

    # 创建新实例
    for cfg in models:
        key = _get_model_key(cfg)
        if key == name or cfg.name == name:
            _llm_instances[key] = _create_llm(cfg)
            return _llm_instances[key]

    raise KeyError(f"LLM model '{name}' not found in configuration")


def _get_model_key(cfg: SingleLLMConfig) -> str:
    """获取模型的唯一标识"""
    return cfg.name if cfg.name else cfg.model


def get_all_llm_models() -> dict[str, ChatOpenAI]:
    """获取所有已配置的 LLM 模型实例

    Raises:
        ValueError: 未配置任何模型时抛出
    """
    global _llm_instances

    settings = _get_settings()
    models = settings.llm.get_models()

    if not models:
        raise ValueError("未配置任何 LLM 模型，请检查 config.yaml")

    for cfg in models:
        key = _get_model_key(cfg)
        if key not in _llm_instances:
            _llm_instances[key] = _create_llm(cfg)

    return _llm_instances.copy()


def reset_llm_models():
    """重置所有 LLM 实例 (用于测试或配置更改)"""
    global _llm_instances, _settings_instance
    _llm_instances = {}
    _settings_instance = None


def get_llm_config(name: str | None = None) -> SingleLLMConfig:
    """获取 LLM 模型配置

    Args:
        name: 模型标识名 (可选)

    Returns:
        SingleLLMConfig 配置对象

    Raises:
        ValueError: 未配置任何模型时抛出
        KeyError: 指定名称的模型不存在
    """
    settings = _get_settings()
    models = settings.llm.get_models()

    if not models:
        raise ValueError("未配置任何 LLM 模型，请检查 config.yaml")

    if not name:
        return models[0]

    for cfg in models:
        if cfg.name == name or _get_model_key(cfg) == name:
            return cfg

    raise KeyError(f"LLM model '{name}' not found in configuration")
