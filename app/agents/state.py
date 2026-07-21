# app/agents/state.py

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single normalized search result. Keeps Tavily's raw shape out of the rest of the app."""
    title: str
    url: str
    content: str


class AgentState(BaseModel):
    """
    Shared state passed between every node in the LangGraph graph.

    Phase 1 keeps this deliberately minimal: a question comes in,
    search results get attached, an answer comes out.
    Later phases (memory, context pipeline, planner) will extend this
    class rather than bolt fields onto a dict.
    """

    question: str
    search_results: list[SearchResult] = Field(default_factory=list)
    answer: Optional[str] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None

    def has_results(self) -> bool:
        return len(self.search_results) > 0