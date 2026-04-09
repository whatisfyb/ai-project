"""Weather Skill - 获取天气信息"""

import json
import subprocess
from typing import Any


# 城市到坐标的映射（常用城市）
CITY_COORDS = {
    "北京": (39.9042, 116.4074, "Asia/Shanghai"),
    "上海": (31.2304, 121.4737, "Asia/Shanghai"),
    "广州": (23.1291, 113.2644, "Asia/Shanghai"),
    "深圳": (22.5431, 114.0579, "Asia/Shanghai"),
    "杭州": (30.2741, 120.1551, "Asia/Shanghai"),
    "成都": (30.5728, 104.0668, "Asia/Shanghai"),
    "武汉": (30.5928, 114.3055, "Asia/Shanghai"),
    "西安": (34.3416, 108.9398, "Asia/Shanghai"),
    "重庆": (29.4316, 106.9123, "Asia/Shanghai"),
    "南京": (32.0603, 118.7969, "Asia/Shanghai"),
    "天津": (39.3434, 117.3616, "Asia/Shanghai"),
    "苏州": (31.2989, 120.5853, "Asia/Shanghai"),
    "长沙": (28.2282, 112.9388, "Asia/Shanghai"),
    "青岛": (36.0671, 120.3826, "Asia/Shanghai"),
    "大连": (38.9140, 121.6147, "Asia/Shanghai"),
    "厦门": (24.4798, 118.0894, "Asia/Shanghai"),
    "福州": (26.0753, 119.2965, "Asia/Shanghai"),
    "郑州": (34.7466, 113.6253, "Asia/Shanghai"),
    "沈阳": (41.8057, 123.4328, "Asia/Shanghai"),
    "哈尔滨": (45.8038, 126.5340, "Asia/Shanghai"),
}


def _curl_get(url: str) -> dict | None:
    """使用 curl 获取 URL 内容（解决 Python SSL 问题）"""
    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


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

    city = city.strip()

    # 获取城市坐标
    if city in CITY_COORDS:
        lat, lon, tz = CITY_COORDS[city]
    else:
        # 尝试通过 geocoding API 获取坐标
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_data = _curl_get(geo_url)
        if not geo_data or not geo_data.get("results"):
            return {
                "success": False,
                "error": f"未找到城市: {city}",
                "data": None,
            }
        result = geo_data["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        tz = result.get("timezone", "Asia/Shanghai")

    # 获取天气数据
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
        f"&timezone={tz}"
        f"&forecast_days={min(days, 7)}"
    )
    data = _curl_get(url)

    if not data:
        return {
            "success": False,
            "error": "获取天气数据失败",
            "data": None,
        }

    # 解析当前天气
    current = data.get("current_weather", {})
    current_weather = {
        "temperature": f"{current.get('temperature', 'N/A')}°C",
        "condition": _weather_code_to_desc(current.get("weathercode", 0)),
        "wind": f"{current.get('windspeed', 'N/A')} km/h",
    }

    # 解析预报
    daily = data.get("daily", {})
    forecast = []
    for i in range(min(days, len(daily.get("time", [])))):
        date = daily["time"][i]
        if i == 0:
            day_name = "今天"
        elif i == 1:
            day_name = "明天"
        elif i == 2:
            day_name = "后天"
        else:
            day_name = f"第{i+1}天"

        forecast.append({
            "day": day_name,
            "date": date,
            "temp_high": f"{daily['temperature_2m_max'][i]}°C",
            "temp_low": f"{daily['temperature_2m_min'][i]}°C",
            "condition": _weather_code_to_desc(daily["weathercode"][i]),
        })

    return {
        "success": True,
        "city": city,
        "days": days,
        "data": {
            "current": current_weather,
            "forecast": forecast,
        },
    }


def _weather_code_to_desc(code: int) -> str:
    """将 WMO 天气代码转换为中文描述"""
    weather_map = {
        0: "晴天",
        1: "晴",
        2: "多云",
        3: "阴天",
        45: "雾",
        48: "霜雾",
        51: "毛毛雨",
        53: "毛毛雨",
        55: "毛毛雨",
        56: "冻毛毛雨",
        57: "冻毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        66: "冻雨",
        67: "冻雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        77: "雪粒",
        80: "阵雨",
        81: "中阵雨",
        82: "强阵雨",
        85: "阵雪",
        86: "强阵雪",
        95: "雷暴",
        96: "雷暴伴冰雹",
        99: "雷暴伴强冰雹",
    }
    return weather_map.get(code, "未知")
