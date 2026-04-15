"""LangSmith Tracing 配置"""

import os

from utils.core.config import Settings


def configure_langsmith():
    """配置 LangSmith tracing (通过环境变量)"""
    settings = Settings()
    ls = settings.langsmith

    if not ls.api_key:
        return

    os.environ["LANGSMITH_API_KEY"] = ls.api_key
    os.environ["LANGSMITH_TRACING"] = str(ls.tracing).lower()
    os.environ["LANGSMITH_PROJECT"] = ls.project
