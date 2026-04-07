"""Weather Skill - 获取天气信息"""

from typing import Any


def run(city: str, days: int = 1, **kwargs) -> dict[str, Any]:
    """获取天气信息

    Args:
        city: 城市名称
        days: 预报天数，默认 1
        **kwargs: 额外参数

    Returns:
        天气信息字典
    """
    if not city or not city.strip():
        return {
            "success": False,
            "error": "城市名称不能为空",
            "data": None,
        }

    # 这里可以接入真实的天气 API
    # 目前返回模拟数据用于演示
    return {
        "success": True,
        "city": city,
        "days": days,
        "data": {
            "current": {
                "temperature": "25°C",
                "condition": "晴朗",
                "humidity": "60%",
            },
            "forecast": [
                {"day": f"第{i+1}天", "temp": f"{20 + i}°C", "condition": "晴"}
                for i in range(days)
            ],
        },
    }
