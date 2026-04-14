"""Main Agent 入口"""

import sys
import warnings

# 抑制 jieba 的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def main():
    """主入口"""
    from agent.main.tui import run_tui
    run_tui()


if __name__ == "__main__":
    main()
