# app/utils/llm_client.py

import sys
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from app.utils.config import get_settings
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin OOP wrapper around LangChain's ChatGroq.

    This is the only class in the app that instantiates ChatGroq directly.
    Nodes and chains should depend on LLMClient, not on langchain_groq —
    so swapping providers later means changing this one class.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        settings = get_settings()
        self.model_name = model or settings.groq_model
        self.temperature = temperature
        self._llm = self._build_llm(settings.groq_api_key)

    def _build_llm(self, api_key: str) -> ChatGroq:
        try:
            logger.info(f"Initializing ChatGroq with model={self.model_name}")
            return ChatGroq(
                api_key=api_key,
                model=self.model_name,
                temperature=self.temperature,
            )
        except Exception as e:
            raise AgentException(e, sys) from e

    @property
    def llm(self) -> ChatGroq:
        """Expose the raw LangChain model — needed when building chains/graphs that expect a Runnable."""
        return self._llm

    def invoke(self, messages: list[BaseMessage]) -> str:
        """Single call, returns just the text content."""
        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            raise AgentException(e, sys) from e