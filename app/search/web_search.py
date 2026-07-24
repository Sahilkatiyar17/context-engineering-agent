# app/search/web_search.py

import sys
import logging

from langchain_tavily import TavilySearch

from app.utils.config import get_settings
from app.utils.exception import AgentException
from app.agents.state import SearchResult

logger = logging.getLogger(__name__)


class WebSearchClient:
    """
    Thin OOP wrapper around LangChain's Tavily search tool.

    This is the only class in the app that talks to Tavily directly.
    Every node calls WebSearchClient.search() and gets back a clean
    list[SearchResult] — never Tavily's raw dict shape.
    """

    def __init__(self, max_results: int = 2):
        settings = get_settings()
        self.max_results = max_results
        self._tool = self._build_tool(settings.tavily_api_key)

    def _build_tool(self, api_key: str) -> TavilySearch:
        try:
            logger.info(f"Initializing TavilySearch with max_results={self.max_results}")
            return TavilySearch(
                tavily_api_key=api_key,
                max_results=self.max_results,
            )
        except Exception as e:
            raise AgentException(e, sys) from e

    def search(self, query: str) -> list[SearchResult]:
        """Runs a search and returns normalized SearchResult objects."""
        try:
            logger.info(f"Searching Tavily for query: {query!r}")
            raw_response = self._tool.invoke({"query": query})
            return self._normalize(raw_response)
        except Exception as e:
            raise AgentException(e, sys) from e

    def _normalize(self, raw_response: dict) -> list[SearchResult]:
        raw_results = raw_response.get("results", [])
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
            )
            for item in raw_results
        ]
        return results