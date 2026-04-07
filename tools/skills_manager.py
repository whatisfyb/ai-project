"""Skills 管理器 - 动态加载和管理 AI Agent Skills

Skill 文件夹结构:
    skills/
      <skill_name>/
        skill.yaml       # 元数据 (name, description, version, author, parameters)
        prompt.md        # 系统提示词/知识文档
        main.py          # 主代码入口 (必须实现 run() 函数)

使用示例:
    >>> from tools.skills_manager import load_skills, list_skills
    >>>
    >>> # 列出所有可用 skills
    >>> available = list_skills()
    >>> print(available)
    [{'name': 'search', 'description': '...', ...}]
    >>>
    >>> # 加载所有 skills
    >>> skills = load_skills()
    >>>
    >>> # 加载指定 skills
    >>> skills = load_skills(['search', 'weather'])
    >>>
    >>> # 使用 skill
    >>> result = skills['search']['run'](query='Python教程')
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.tools import tool

# 配置日志
logger = logging.getLogger(__name__)

# 缓存
_skills_cache: dict[str, dict] | None = None


def _get_skills_dir() -> Path:
    """获取 skills 目录路径"""
    # 相对于项目根目录
    skills_dir = Path(__file__).parent.parent / "skills"
    return skills_dir


def _load_yaml(path: Path) -> dict[str, Any]:
    """加载 YAML 文件"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_markdown(path: Path) -> str:
    """加载 Markdown 文件"""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_python_module(path: Path) -> Any:
    """动态加载 Python 模块"""
    module_name = f"_skill_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_skill_structure(skill_path: Path) -> tuple[bool, str]:
    """验证 skill 文件夹结构是否完整

    Returns:
        (是否有效, 错误信息)
    """
    # 检查必要文件
    skill_yaml = skill_path / "skill.yaml"
    main_py = skill_path / "main.py"

    if not skill_yaml.exists():
        return False, f"缺少必要文件: skill.yaml"

    if not main_py.exists():
        return False, f"缺少必要文件: main.py"

    return True, ""


def _load_single_skill(skill_path: Path) -> dict[str, Any] | None:
    """加载单个 skill

    Args:
        skill_path: skill 文件夹路径

    Returns:
        skill 字典，加载失败返回 None
    """
    skill_name = skill_path.name

    # 验证结构
    is_valid, error_msg = _validate_skill_structure(skill_path)
    if not is_valid:
        logger.warning(f"Skill '{skill_name}' 结构无效: {error_msg}")
        return None

    try:
        # 加载元数据
        metadata = _load_yaml(skill_path / "skill.yaml")

        # 验证必要字段
        if "name" not in metadata:
            metadata["name"] = skill_name
        if "description" not in metadata:
            logger.warning(f"Skill '{skill_name}' 缺少 description 字段")
            metadata["description"] = ""

        # 加载 prompt
        prompt_path = skill_path / "prompt.md"
        prompt = _load_markdown(prompt_path) if prompt_path.exists() else ""

        # 加载 main.py
        module = _load_python_module(skill_path / "main.py")

        # 检查 run 函数
        if not hasattr(module, "run"):
            logger.warning(f"Skill '{skill_name}' 的 main.py 缺少 run() 函数")
            return None

        run_func = module.run
        if not callable(run_func):
            logger.warning(f"Skill '{skill_name}' 的 run 不是可调用对象")
            return None

        # 构建 skill 字典
        skill = {
            "name": metadata["name"],
            "description": metadata.get("description", ""),
            "version": metadata.get("version", "0.1.0"),
            "author": metadata.get("author", ""),
            "parameters": metadata.get("parameters", []),
            "prompt": prompt,
            "run": run_func,
            "_path": str(skill_path),
        }

        logger.info(f"成功加载 skill: {skill_name}")
        return skill

    except Exception as e:
        logger.warning(f"加载 skill '{skill_name}' 失败: {e}")
        return None


def _list_skills_raw() -> list[dict[str, Any]]:
    """列出所有可用的 skills（仅返回元数据，不加载代码）

    Returns:
        skill 元数据列表，每个元素包含 name, description, version, author
    """
    skills_dir = _get_skills_dir()

    if not skills_dir.exists():
        logger.warning(f"Skills 目录不存在: {skills_dir}")
        return []

    available_skills = []

    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue

        skill_yaml = skill_path / "skill.yaml"
        if not skill_yaml.exists():
            continue

        try:
            metadata = _load_yaml(skill_yaml)
            available_skills.append({
                "name": metadata.get("name", skill_path.name),
                "description": metadata.get("description", ""),
                "version": metadata.get("version", "0.1.0"),
                "author": metadata.get("author", ""),
            })
        except Exception as e:
            logger.debug(f"读取 skill '{skill_path.name}' 元数据失败: {e}")
            continue

    return available_skills


def _load_skills_raw(
    names: list[str] | None = None,
    cache: bool = True,
) -> dict[str, dict[str, Any]]:
    """加载 skills

    Args:
        names: 指定要加载的 skill 名称列表，None 表示加载所有
        cache: 是否使用缓存，默认 True

    Returns:
        skill 字典，key 为 skill 名称，value 为 skill 信息

    Example:
        >>> skills = load_skills()
        >>> skills = load_skills(['search', 'weather'])
        >>> skills = load_skills(cache=False)  # 强制重新加载
        >>>
        >>> # 使用 skill
        >>> result = skills['search']['run'](query='Python')
    """
    global _skills_cache

    # 检查缓存
    if cache and _skills_cache is not None:
        if names is None:
            return _skills_cache
        # 返回缓存中指定的 skills
        return {k: v for k, v in _skills_cache.items() if k in names}

    skills_dir = _get_skills_dir()

    if not skills_dir.exists():
        logger.warning(f"Skills 目录不存在: {skills_dir}")
        return {}

    loaded_skills: dict[str, dict[str, Any]] = {}

    # 遍历 skills 目录
    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue

        skill_name = skill_path.name

        # 如果指定了 names，只加载指定的
        if names is not None and skill_name not in names:
            continue

        # 加载 skill
        skill = _load_single_skill(skill_path)
        if skill is not None:
            loaded_skills[skill_name] = skill

    # 更新缓存
    if cache:
        _skills_cache = loaded_skills.copy()

    logger.info(f"共加载 {len(loaded_skills)} 个 skills")
    return loaded_skills


def clear_skills_cache() -> None:
    """清除 skills 缓存"""
    global _skills_cache
    _skills_cache = None
    logger.info("Skills 缓存已清除")


def get_skill(name: str, cache: bool = True) -> dict[str, Any] | None:
    """获取单个 skill

    Args:
        name: skill 名称
        cache: 是否使用缓存

    Returns:
        skill 字典，不存在返回 None
    """
    skills = _load_skills_raw(names=[name], cache=cache)
    return skills.get(name)


@tool
def list_skills() -> list[dict[str, Any]]:
    """列出所有可用技能，供LLM查看和选择需要加载哪些技能。"""
    return _list_skills_raw()


@tool
def load_skills(names: list[str]) -> dict[str, Any]:
    """加载指定的技能到当前上下文，使LLM可以调用这些技能。

    Args:
        names: 要加载的技能名称列表
    """
    # 调用原始加载函数，禁用缓存以确保重新加载
    loaded = _load_skills_raw(names=names, cache=False)
    # 返回简化信息（不包含函数引用）
    result = {}
    for name, skill in loaded.items():
        result[name] = {
            "name": skill["name"],
            "description": skill["description"],
            "version": skill["version"],
            "author": skill["author"],
            "parameters": skill["parameters"],
            "prompt_length": len(skill["prompt"]),
            "loaded": True
        }
    return result


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("=== 列出可用 Skills ===")
    available = _list_skills_raw()
    for s in available:
        print(f"  - {s['name']}: {s['description']}")

    print("\n=== 加载所有 Skills ===")
    skills = _load_skills_raw(cache=False)
    for name, skill in skills.items():
        print(f"  - {name}: {skill['description']}")
        print(f"    version: {skill['version']}")
        print(f"    prompt length: {len(skill['prompt'])} chars")
        print(f"    run callable: {callable(skill['run'])}")
