"""A2A 配置 - 外部 Agent 配置管理"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class AgentEndpoint(BaseModel):
    """外部 Agent 端点配置"""
    url: str
    description: str = ""
    api_key: Optional[str] = None


class A2AConfig(BaseModel):
    """A2A 配置"""
    agents: dict[str, AgentEndpoint] = {}


def load_a2a_config() -> A2AConfig:
    """从 config.yaml 加载 A2A 配置

    Returns:
        A2AConfig 实例
    """
    config_path = Path("config.yaml")

    if not config_path.exists():
        return A2AConfig()

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        a2a_data = data.get("a2a", {})
        agents_data = a2a_data.get("agents", {})

        agents = {}
        for name, endpoint in agents_data.items():
            if isinstance(endpoint, dict):
                agents[name] = AgentEndpoint(**endpoint)

        return A2AConfig(agents=agents)
    except Exception:
        return A2AConfig()


# 全局配置缓存
_config: A2AConfig | None = None


def get_a2a_config() -> A2AConfig:
    """获取 A2A 配置（单例）"""
    global _config
    if _config is None:
        _config = load_a2a_config()
    return _config


def get_agent_endpoint(agent_name: str) -> AgentEndpoint | None:
    """获取指定 Agent 的端点配置"""
    return get_a2a_config().agents.get(agent_name)


def reload_a2a_config() -> A2AConfig:
    """重新加载配置"""
    global _config
    _config = load_a2a_config()
    return _config
