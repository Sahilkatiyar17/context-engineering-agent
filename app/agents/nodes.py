# app/agents/nodes.py

import sys
import logging
import time
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.search.web_search import WebSearchClient
from app.utils.llm_client import LLMClient
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


class AgentNodes:
    """
    Holds the LangGraph node functions for Phase 1's baseline agent
    (search -> LLM). Each public method matches the signature LangGraph
    expects: (state: AgentState) -> dict of updated fields.

    Wrapping nodes as bound methods (instead of free functions with
    module-level clients) is what makes this OOP: dependencies are
    injected once in __init__, not re-created or globally imported
    inside each node.
    """

    def __init__(self, search_client: WebSearchClient, llm_client: LLMClient):
        self.search_client = search_client
        self.llm_client = llm_client

    def search_node(self, state: AgentState) -> dict:
        """Runs web search for the question and attaches results to state."""
        try:
            logger.info(f"[search_node] question={state.question!r}")
            results = self.search_client.search(state.question)
            return {"search_results": results}
        except Exception as e:
            raise AgentException(e, sys) from e

    

    def llm_node(self, state: AgentState) -> dict:
        try:
            logger.info(f"[llm_node] building answer from {len(state.search_results)} results")
            messages = self._build_messages(state)

            start = time.perf_counter()
            result = self.llm_client.invoke_with_usage(messages)
            elapsed = time.perf_counter() - start

            return {
                "answer": result["text"],
                "total_tokens": result["total_tokens"],
                "latency_seconds": round(elapsed, 3),
            }
        except Exception as e:
            raise AgentException(e, sys) from e

    def _build_messages(self, state: AgentState) -> list:
        """
        Assembles the prompt. Deliberately inline and simple for Phase 1 --
        app/prompts/templates.py (next in the flow) will formalize this
        once we're comparing prompt versions.
        """
        context_block = "\n\n".join(
            f"Source: {r.title} ({r.url})\n{r.content}" for r in state.search_results
        )
        system = SystemMessage(
            content=(
                "You are a research assistant. Answer the user's question using "
                "only the provided sources. Be concise and factual."
            )
        )
        human = HumanMessage(
            content=f"Question: {state.question}\n\nSources:\n{context_block}"
        )
        return [system, human]