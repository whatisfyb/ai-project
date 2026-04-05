

from langchain.agents import create_agent

from tools import workflow_trigger
from utils.llm import get_llm_model
from utils.sqlite_checkpointer import SqliteCheckpointer


checkpointer = SqliteCheckpointer(db_path=".checkpoints/gateway.db")

llm = get_llm_model()

agent = create_agent(
    llm,
    tools=[workflow_trigger],
    system_prompt="你是一个科研助手，专门帮助用户完成各种科研相关的任务。当用户有明确的科研需求时，调用 workflow_trigger 工具来触发复杂工作流。",
    checkpointer=checkpointer,
)
