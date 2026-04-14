"""Main Agent 入口"""

import sys

def main():
    """主入口"""
    # 默认使用 TUI
    if len(sys.argv) > 1 and sys.argv[1] == "--repl":
        # 传统 REPL 模式
        from agent.main.repl import run_repl
        run_repl()
    else:
        # TUI 模式（默认）
        from agent.main.tui import run_tui
        run_tui()


if __name__ == "__main__":
    main()
