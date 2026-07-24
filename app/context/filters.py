from abc import ABC, abstractmethod


class ContextFilter(ABC):
    @abstractmethod
    def apply(self, search_results: list[dict]) -> list[dict]:
        raise NotImplementedError


class NoOpFilter(ContextFilter):
    """Baseline -- passes everything through, unchanged."""
    def apply(self, search_results: list[dict]) -> list[dict]:
        return search_results


class RelevanceScoreFilter(ContextFilter):
    """Drops search results below a minimum Tavily relevance score."""
    def __init__(self, min_score: float = 0.3):
        self.min_score = min_score

    def apply(self, search_results: list[dict]) -> list[dict]:
        return [r for r in search_results if r.get("score", 0.0) >= self.min_score]