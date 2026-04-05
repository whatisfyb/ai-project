"""一次性规划 Agent — 将用户目标拆解为带依赖关系的任务计划"""

import json
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

from agents.planner.plan_structure import Plan, TaskStatus, PlanStatus
from utils.llm import get_llm_model

# ---------------------------------------------------------------------------
# Prompt (loaded from planner_prompt.txt)
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "planner_prompt.txt"
_PLANNER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Approach A: Structured output (preferred)
# ---------------------------------------------------------------------------


def _build_plan(goal: str, context: str = "") -> str:
    """构建 prompt 输入"""
    if context:
        instruction = f"目标：{goal}\n额外上下文：{context}"
    else:
        instruction = f"目标：{goal}"
    return _PLANNER_PROMPT.format(goal_instruction=instruction)


def _parse_json_from_text(text: str) -> dict:
    """从 LLM 输出中提取 JSON，支持多种情况"""
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试从文本中找到第一个 { 到最后一个 } 的范围
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从输出中解析出合法 JSON: {text[:300]}...")


def _dict_to_plan(data: dict) -> Plan:
    """将字典转换为 Plan，校验并补全默认值"""
    tasks = []
    for t in data.get("tasks", []):
        tasks.append({
            "id": t["id"],
            "description": t["description"],
            "status": t.get("status", "pending"),
            "dependencies": t.get("dependencies", []),
            "result": t.get("result"),
            "error": t.get("error"),
        })

    return Plan(
        id=data.get("id", "unnamed-plan"),
        goal=data.get("goal", ""),
        tasks=tasks,
        status=data.get("status", "running"),
    )


def _create_structured_llm(llm: ChatOpenAI):
    """创建带 structured output 的 LLM 实例"""

    structured_llm = llm.with_structured_output(Plan)

    return structured_llm


# ---------------------------------------------------------------------------
# Approach B: JSON fallback (pure prompt + manual parse)
# ---------------------------------------------------------------------------


def _create_plan_from_json(text: str) -> Plan:
    """从 JSON 文本创建 Plan（Approach B 的回退路径）"""
    data = _parse_json_from_text(text)
    return _dict_to_plan(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_plan(
    goal: str,
    context: str = "",
    llm: Optional[ChatOpenAI] = None,
    use_structured: bool = True,
) -> Plan:
    """创建计划

    Args:
        goal: 用户目标/任务描述
        context: 可选的额外上下文
        llm: 可选的 LLM 实例，不提供则用默认模型
        use_structured: 是否优先使用 structured output (Approach A),
                        为 False 则纯用 JSON prompt + 手动解析 (Approach B)

    Returns:
        Plan 实例，包含拆解后的任务列表

    Raises:
        ValueError: 当 LLM 输出无法解析为 Plan 时
    """
    if llm is None:
        llm = get_llm_model()

    instruction = _build_plan(goal, context)

    if use_structured:
        # Approach A: structured output
        try:
            structured_llm = _create_structured_llm(llm)
            plan = structured_llm.invoke(instruction)
            if isinstance(plan, Plan):
                plan.status = PlanStatus.RUNNING
                if plan.has_circular_dependency():
                    raise ValueError("LLM 输出的计划存在循环依赖")
                return plan
        except Exception:
            # 如果结构化输出失败，回退到 JSON 解析 (Approach B)
            response = llm.invoke(instruction)
            content = response.content if hasattr(response, "content") else str(response)
            return _create_plan_from_json(content)
        raise ValueError("LLM 输出格式不符合 Plan 结构")
    else:
        # Approach B: pure JSON via prompt
        response = llm.invoke(instruction)
        content = response.content if hasattr(response, "content") else str(response)
        return _create_plan_from_json(content)


def create_plan_from_json_text(json_text: str) -> Plan:
    """从 JSON 文本直接创建 Plan（用于独立测试或外部输入）

    Args:
        json_text: 包含计划结构的 JSON 字符串

    Returns:
        Plan 实例
    """
    return _create_plan_from_json(json_text)


# ---------------------------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.llm import reset_llm_models
    from utils.embedding import reset_embedding_model

    goals = [
        "写一篇关于人工智能在医疗领域应用的文章，并在社交媒体上分享",
        "搭建一个个人博客网站，支持文章发布和技术分享",
        "完成一个Python数据分析脚本，读取CSV并生成可视化报告",
    ]

    for i, g in enumerate(goals, 1):
        print(f"\n{'='*60}")
        approach = "structured" if i % 2 == 1 else "json"
        print(f"示例 {i}: {g}")
        print(f"方式: {approach}")
        print(f"{'='*60}")

        reset_llm_models()
        reset_embedding_model()

        plan = create_plan(
            goal=g,
            use_structured=(i % 2 == 1),
        )

        print(f"\n  Plan ID: {plan.id}")
        print(f"  Goal:    {plan.goal}")
        print(f"  Status:  {plan.status}")
        print(f"  Tasks ({len(plan.tasks)}):")
        for t in plan.tasks:
            deps = ", ".join(t.dependencies) if t.dependencies else "无"
            print(f"    [{t.id}] {t.description}")
            print(f"           依赖: {deps}")
        print()
