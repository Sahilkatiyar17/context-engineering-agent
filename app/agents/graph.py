# app/agents/graph.py

from pathlib import Path

import sys, logging , sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.search.web_search import WebSearchClient
from app.utils.llm_client import LLMClient
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


DB_PATH = Path("data") / "checkpoints.db"


class ResearchAgentGraph:
    def __init__(self, search_client: WebSearchClient | None = None, llm_client: LLMClient | None = None,
                 summarize_after_n_messages: int = 6, db_path: Path = DB_PATH):
        self.nodes = AgentNodes(
            search_client=search_client or WebSearchClient(),
            llm_client=llm_client or LLMClient(),
            summarize_after_n_messages=summarize_after_n_messages,
        )
        self.checkpointer = self._build_checkpointer(db_path)
        self._graph = self._build_graph()

    def _build_checkpointer(self, db_path: Path) -> SqliteSaver:
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: SqliteSaver may be called from different threads
            # depending on how LangGraph schedules nodes -- safe here since we're not
            # sharing this connection across processes.
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            logger.info(f"Using SQLite checkpointer at {db_path}")
            return SqliteSaver(conn)
        except Exception as e:
            raise AgentException(e, sys) from e

    def _build_graph(self):
        try:
            builder = StateGraph(AgentState)
            builder.add_node("search", self.nodes.search_node)
            builder.add_node("chat", self.nodes.chat_node)
            builder.add_node("summarize", self.nodes.summarize_conversation)

            builder.set_entry_point("search")
            builder.add_edge("search", "chat")
            builder.add_conditional_edges("chat", self.nodes.should_summarize, {True: "summarize", False: END})
            builder.add_edge("summarize", END)

            return builder.compile(checkpointer=self.checkpointer)
        except Exception as e:
            raise AgentException(e, sys) from e

    def run(self, question: str, thread_id: str, experiment_id: str = "default", strategy: str = "unspecified") -> AgentState:
        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "tags": [experiment_id, strategy],
                "metadata": {"experiment_id": experiment_id, "strategy": strategy},
            }
            result = self._graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
            
            return AgentState(**result)
        except Exception as e:
            raise AgentException(e, sys) from e