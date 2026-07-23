# app/agents/state.py

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SearchResult(BaseModel):
    """A single normalized search result. Keeps Tavily's raw shape out of the rest of the app."""
    title: str
    url: str
    content: str


from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SearchResult(BaseModel):
    title: str
    url: str
    content: str


class AgentState(BaseModel):
    # add_messages reducer means new messages APPEND instead of overwrite,
    # and it auto-assigns .id to each message (needed for RemoveMessage later)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    summary: str = ""
    long_term_memories: list[str] = Field(default_factory=list)
    search_results: list[dict] = Field(default_factory=list)
    answer: Optional[str] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None

    @property
    def latest_question(self) -> str:
        """The most recent user message -- replaces the old standalone `question` field."""
        for m in reversed(self.messages):
            if m.type == "human":
                return m.content
        return ""