import sys, logging
import cohere
from abc import ABC, abstractmethod
from app.utils.config import get_settings
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


class ContextRanker(ABC):
    @abstractmethod
    def apply(self, question: str, search_results: list[dict]) -> list[dict]:
        raise NotImplementedError


class NoOpRanker(ContextRanker):
    def apply(self, question: str, search_results: list[dict]) -> list[dict]:
        return search_results


class ScoreRanker(ContextRanker):
    """Cheap baseline -- reuses Tavily's own relevance score."""
    def apply(self, question: str, search_results: list[dict]) -> list[dict]:
        return sorted(search_results, key=lambda r: r.get("score", 0.0), reverse=True)


class CohereRerankRanker(ContextRanker):
    """Real cross-encoder reranking -- scores (question, content) pairs jointly."""

    def __init__(self, model: str = "rerank-v3.5"):
        settings = get_settings()
        self.model = model
        self._client = self._build_client(settings.cohere_api_key)

    def _build_client(self, api_key: str) -> cohere.Client:
        try:
            logger.info(f"Initializing Cohere client with model={self.model}")
            return cohere.Client(api_key=api_key)
        except Exception as e:
            raise AgentException(e, sys) from e

    def apply(self, question: str, search_results: list[dict]) -> list[dict]:
        try:
            if not search_results:
                return search_results
            documents = [r["content"] for r in search_results]
            response = self._client.rerank(
                model=self.model,
                query=question,
                documents=documents,
            )
            reranked = [search_results[result.index] for result in response.results]
            logger.info(f"Cohere reranked {len(reranked)} search results")
            return reranked
        except Exception as e:
            raise AgentException(e, sys) from e