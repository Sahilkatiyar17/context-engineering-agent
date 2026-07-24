import sys, sqlite3, logging
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.search.web_search import WebSearchClient
from app.utils.llm_client import LLMClient
from app.memory.long_term import LongTermMemory
from app.utils.exception import AgentException
from app.context.filters import ContextFilter, NoOpFilter
from app.context.ranking import NoOpRanker, ContextRanker
logger = logging.getLogger(__name__)

DB_PATH = Path("data") / "checkpoints.db"


class ResearchAgentGraph:
    def __init__(self, user_id: str, search_client: WebSearchClient | None = None,
                 llm_client: LLMClient | None = None, long_term_memory: LongTermMemory | None = None,
                 summarize_after_n_messages: int = 6, db_path: Path = DB_PATH,context_filter: ContextFilter | None = None,context_ranker: ContextRanker | None = None):
        self.nodes = AgentNodes(
            search_client=search_client or WebSearchClient(),
            llm_client=llm_client or LLMClient(),
            long_term_memory=long_term_memory or LongTermMemory(),
            user_id=user_id,
            summarize_after_n_messages=summarize_after_n_messages,
            context_filter=context_filter or NoOpFilter(),
            context_ranker=context_ranker or NoOpRanker(),
        )
        self.checkpointer = self._build_checkpointer(db_path)
        self._graph = self._build_graph()

    def _build_checkpointer(self, db_path: Path) -> SqliteSaver:
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            logger.info(f"Using SQLite checkpointer at {db_path}")
            return SqliteSaver(conn)
        except Exception as e:
            raise AgentException(e, sys) from e

    def _build_graph(self):
        try:
            builder = StateGraph(AgentState)

            builder.add_node("recall_memory", self.nodes.recall_memory_node)
            builder.add_node("search", self.nodes.search_node)
            builder.add_node("filter_context", self.nodes.filter_context_node)
            builder.add_node("rank_context", self.nodes.rank_context_node)
            builder.add_node("chat", self.nodes.chat_node)
            builder.add_node("remember", self.nodes.remember_node)
            builder.add_node("summarize", self.nodes.summarize_conversation)

            builder.set_entry_point("recall_memory")
            builder.add_edge("recall_memory", "search")
            builder.add_edge("search", "filter_context")
            builder.add_edge("filter_context", "rank_context")
            builder.add_edge("rank_context", "chat")
            builder.add_edge("chat", "remember")
            builder.add_conditional_edges("remember", self.nodes.should_summarize, {True: "summarize", False: END})
            builder.add_edge("summarize", END)

            return builder.compile(checkpointer=self.checkpointer)
        except Exception as e:
            raise AgentException(e, sys) from e

    def run(self, question: str, thread_id: str, experiment_id: str = "default", strategy: str = "unspecified") -> AgentState:
        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "tags": [experiment_id, strategy],
                "metadata": {"experiment_id": experiment_id, "strategy": strategy, "user_id": self.nodes.user_id},
            }
            result = self._graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
            return AgentState(**result)
        except Exception as e:
            raise AgentException(e, sys) from e