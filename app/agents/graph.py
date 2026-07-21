# app/agents/graph.py

import sys
import logging

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.search.web_search import WebSearchClient
from app.utils.llm_client import LLMClient
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


class ResearchAgentGraph:
    """
    Builds and compiles the Phase 1 LangGraph graph:

        search_node -> llm_node -> END

    Wrapped as a class so construction (wiring nodes, compiling) is
    separate from execution (run()), and so later phases can subclass
    or extend this rather than rewriting a module-level script.
    """

    def __init__(self, search_client: WebSearchClient | None = None, llm_client: LLMClient | None = None):
        self.nodes = AgentNodes(
            search_client=search_client or WebSearchClient(),
            llm_client=llm_client or LLMClient(),
        )
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            logger.info("Building StateGraph")
            builder = StateGraph(AgentState)

            builder.add_node("search", self.nodes.search_node)
            builder.add_node("llm", self.nodes.llm_node)

            builder.set_entry_point("search")
            builder.add_edge("search", "llm")
            builder.add_edge("llm", END)

            return builder.compile()
        except Exception as e:
            raise AgentException(e, sys) from e

    def run(self, question: str, experiment_id: str = "default", strategy: str = "unspecified") -> AgentState:
        try:
            logger.info(f"Running graph for question: {question!r}")
            initial_state = AgentState(question=question)

            config = {
                "tags": [experiment_id, strategy],
                "metadata": {"experiment_id": experiment_id, "strategy": strategy},
            }

            result = self._graph.invoke(initial_state, config=config)
            return AgentState(**result)
        except Exception as e:
            raise AgentException(e, sys) from e