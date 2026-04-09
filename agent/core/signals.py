"""中断信号管理"""

import threading

# 全局中断事件
_interrupt_event = threading.Event()


def set_interrupt():
    """设置中断信号"""
    _interrupt_event.set()


def clear_interrupt():
    """清除中断信号"""
    _interrupt_event.clear()


def is_interrupted() -> bool:
    """检查是否被中断"""
    return _interrupt_event.is_set()
