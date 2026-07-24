from abc import ABC, abstractmethod
from app.utils.compression_llm_client import CompressionLLMClient


class ContextCompressor(ABC):
    @abstractmethod
    def apply(self, search_results: list[dict]) -> list[dict]:
        raise NotImplementedError


class NoOpCompressor(ContextCompressor):
    def apply(self, search_results: list[dict]) -> list[dict]:
        return search_results


class LLMCompressor(ContextCompressor):
    """Compresses all surviving search results into one dense, fact-preserving block."""

    def __init__(self, compression_client: CompressionLLMClient | None = None):
        self.compression_client = compression_client or CompressionLLMClient()

    def apply(self, search_results: list[dict]) -> list[dict]:
        if not search_results:
            return search_results

        combined = "\n\n".join(f"Source: {r['title']}\n{r['content']}" for r in search_results)
        prompt = (
            "Compress the following sources into a dense summary. "
            "Preserve all specific facts, numbers, and technical terms. "
            "Remove redundancy and filler. Do not add commentary.\n\n"
            f"{combined}"
        )
        compressed_text = self.compression_client.invoke(prompt)

        return [{"title": "Compressed Sources", "url": "", "content": compressed_text, "score": 1.0}]