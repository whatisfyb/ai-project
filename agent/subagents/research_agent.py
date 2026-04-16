"""Research Agent - 信息收集与研究"""

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, END, START

from agent.core.base_agent import BaseAgent
from agent.a2a.models import AgentCard, AgentCapabilities, Skill
from agent.subagents.base import BaseSubagent
from tools.web import web


class ResearchAgentState(TypedDict):
    """Research Agent 状态"""
    query: str  # 研究问题
    search_results: Annotated[list[dict], operator.add]  # 搜索结果
    papers: Annotated[list[dict], operator.add]  # 论文列表
    rag_context: str | None  # RAG 检索结果
    summary: str | None  # 研究总结


class ResearchAgent(BaseSubagent[ResearchAgentState], BaseAgent):
    """Research Agent - 信息收集与研究

    职责：
    - 网络搜索
    - arXiv 论文搜索与下载
    - RAG 知识库检索
    - 信息整合
    """

    agent_id = "research-agent"
    agent_type = "Research"

    # ============ BaseSubagent 接口 ============

    @property
    def description(self) -> str:
        return (
            "研究代理，用于搜索信息、下载论文、使用RAG知识库。"
            "适用于需要收集资料、查找文献、检索知识库的任务。"
        )

    @property
    def tools(self) -> list:
        return [web]

    # ============ BaseAgent 接口 ============

    def get_card(self) -> AgentCard:
        """返回 Agent 能力声明"""
        return AgentCard(
            id=self.agent_id,
            name="Research Agent",
            description="研究代理，搜索信息、下载论文、使用知识库",
            capabilities=AgentCapabilities(text=True, files=True),
            skills=[
                Skill(name="research", description="研究分析"),
                Skill(name="search", description="网络搜索"),
                Skill(name="paper_kb", description="论文知识库"),
            ],
        )

    def handle_task(self, task) -> Any:
        """处理 A2A Task"""
        message = task.history[0].get_text() if task.history else ""
        return self.run(query=message, thread_id=task.metadata.get("thread_id", "default"))

    def _web_search_node(self, state: ResearchAgentState) -> dict:
        """网络搜索节点"""
        query = state["query"]

        prompt = f"""你是一个搜索专家。用户的问题是：{query}

请判断是否需要进行网络搜索，如果需要，生成 1-3 个搜索关键词。
格式：
NEED_SEARCH: yes/no
KEYWORDS: 关键词1, 关键词2, 关键词3
"""

        response = self.llm.invoke(prompt)
        content = response.content.lower()

        results = []
        if "need_search: yes" in content or "need_search:yes" in content:
            # 提取关键词
            keywords = []
            if "keywords:" in content:
                kw_line = [line for line in content.split("\n") if "keywords:" in line]
                if kw_line:
                    keywords = [k.strip() for k in kw_line[0].split(":", 1)[1].split(",")]

            if not keywords:
                keywords = [query]

            # 执行搜索
            for kw in keywords[:3]:
                try:
                    result = web.invoke({"action": "search", "query": kw, "max_results": 5})
                    if result.get("status") != "error":
                        results.append({
                            "keyword": kw,
                            "results": result.get("results", []),
                            "answer": result.get("answer", ""),
                        })
                    else:
                        results.append({"keyword": kw, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    results.append({"keyword": kw, "error": str(e)})

        return {"search_results": results}

    def _paper_search_node(self, state: ResearchAgentState) -> dict:
        """论文搜索节点"""
        query = state["query"]

        prompt = f"""你是一个学术研究专家。用户的问题是：{query}

请判断是否需要搜索学术论文，如果需要，生成 1-3 个学术搜索关键词。
格式：
NEED_PAPER: yes/no
KEYWORDS: 关键词1, 关键词2
"""

        response = self.llm.invoke(prompt)
        content = response.content.lower()

        papers = []
        if "need_paper: yes" in content or "need_paper:yes" in content:
            # 提取关键词
            keywords = []
            if "keywords:" in content:
                kw_line = [line for line in content.split("\n") if "keywords:" in line]
                if kw_line:
                    keywords = [k.strip() for k in kw_line[0].split(":", 1)[1].split(",")]

            if not keywords:
                keywords = [query]

            # 执行论文搜索
            for kw in keywords[:2]:
                try:
                    result = web.invoke({"action": "arxiv_search", "query": kw, "max_results": 5})
                    if isinstance(result, dict) and result.get("status") != "error":
                        papers.extend(result.get("papers", []))
                    elif isinstance(result, list):
                        papers.extend(result)
                    else:
                        papers.append({"error": result.get("error", "Unknown error"), "keyword": kw})
                except Exception as e:
                    papers.append({"error": str(e), "keyword": kw})

        return {"papers": papers}

    def _rag_search_node(self, state: ResearchAgentState) -> dict:
        """RAG 知识库检索节点"""
        query = state["query"]

        try:
            from utils.retrieval.retriever import Retriever

            retriever = Retriever()
            docs = retriever.retrieve(query, top_k=5)

            if docs:
                context = "\n\n---\n\n".join(
                    f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                    for doc in docs
                )
            else:
                context = None
        except Exception:
            context = None

        return {"rag_context": context}

    def _summarize_node(self, state: ResearchAgentState) -> dict:
        """整合总结节点"""
        query = state["query"]
        search_results = state["search_results"]
        papers = state["papers"]
        rag_context = state["rag_context"]

        # 构建上下文
        context_parts = []

        if search_results:
            context_parts.append("## 网络搜索结果")
            for sr in search_results:
                if "error" in sr:
                    context_parts.append(f"关键词 '{sr['keyword']}' 搜索出错: {sr['error']}")
                else:
                    context_parts.append(f"### 关键词: {sr['keyword']}")
                    if sr.get("answer"):
                        context_parts.append(f"摘要: {sr['answer']}")
                    for r in sr.get("results", [])[:3]:
                        context_parts.append(f"- [{r.get('title', 'No title')}]({r.get('url', '')})")
                        context_parts.append(f"  {r.get('content', '')[:200]}...")

        if papers:
            context_parts.append("\n## 相关论文")
            for p in papers[:5]:
                if isinstance(p, dict) and "title" in p:
                    context_parts.append(f"- [{p.get('title', 'No title')}]({p.get('url', '')})")
                    context_parts.append(f"  作者: {', '.join(p.get('authors', []))}")
                    context_parts.append(f"  摘要: {p.get('abstract', '')[:200]}...")

        if rag_context:
            context_parts.append("\n## 知识库检索结果")
            context_parts.append(rag_context[:2000])

        full_context = "\n".join(context_parts) if context_parts else "未找到相关信息"

        prompt = f"""你是一个研究总结专家。请根据以下信息回答用户的问题。

用户问题：{query}

收集到的信息：
{full_context}

请用中文撰写一份研究报告，包括：
1. 问题回答
2. 主要发现
3. 参考来源

如果信息不足，请诚实说明。"""

        response = self.llm.invoke(prompt)
        return {"summary": response.content}

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(ResearchAgentState)

        # 添加节点
        graph.add_node("web_search", self._web_search_node)
        graph.add_node("paper_search", self._paper_search_node)
        graph.add_node("rag_search", self._rag_search_node)
        graph.add_node("summarize", self._summarize_node)

        # 添加边 - 并行搜索后汇总
        graph.add_edge(START, "web_search")
        graph.add_edge(START, "paper_search")
        graph.add_edge(START, "rag_search")

        # 汇总节点需要等待所有搜索完成
        graph.add_edge("web_search", "summarize")
        graph.add_edge("paper_search", "summarize")
        graph.add_edge("rag_search", "summarize")

        graph.add_edge("summarize", END)

        return graph

    def run(self, query: str, thread_id: str = "default") -> dict:
        """运行 Research Agent

        Args:
            query: 研究问题
            thread_id: 会话 ID

        Returns:
            包含 search_results, papers, rag_context, summary 的结果
        """
        input_data = {
            "query": query,
            "search_results": [],
            "papers": [],
            "rag_context": None,
            "summary": None,
        }
        result = super().run(input_data, thread_id)
        return result
