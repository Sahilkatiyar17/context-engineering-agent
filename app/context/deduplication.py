import difflib
from abc import ABC, abstractmethod


class ContextDeduplicator(ABC):
    @abstractmethod
    def apply(self, search_results: list[dict]) -> list[dict]:
        raise NotImplementedError


class NoOpDeduplicator(ContextDeduplicator):
    def apply(self, search_results: list[dict]) -> list[dict]:
        return search_results


class SimilarityDeduplicator(ContextDeduplicator):
    """
    Drops results whose content is near-identical to one already kept.
    Uses difflib's ratio (0-1 text similarity) -- no embeddings/API call needed,
    cheap enough to run on every search result set.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def apply(self, search_results: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for result in search_results:
            if not self._is_duplicate(result, kept):
                kept.append(result)
        return kept

    def _is_duplicate(self, candidate: dict, kept: list[dict]) -> bool:
        for existing in kept:
            ratio = difflib.SequenceMatcher(None, candidate["content"], existing["content"]).ratio()
            if ratio >= self.similarity_threshold:
                return True
        return False