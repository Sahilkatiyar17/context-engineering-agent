import sys
import logging

from mem0 import MemoryClient

from app.utils.config import get_settings
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Thin OOP wrapper around Mem0's MemoryClient.

    Unlike short-term memory (scoped to one thread_id / conversation),
    this is scoped to user_id -- it persists across every conversation
    that user ever has, which is the whole point of "long-term."
    """

    def __init__(self, top_k: int = 5):
        settings = get_settings()
        self.top_k = top_k
        self._client = self._build_client(settings.mem0_api_key)

    def _build_client(self, api_key: str) -> MemoryClient:
        try:
            logger.info("Initializing Mem0 MemoryClient")
            return MemoryClient(api_key=api_key)
        except Exception as e:
            raise AgentException(e, sys) from e

    def remember(self, user_id: str, question: str, answer: str) -> None:
        """Extraction: hands the turn to Mem0, which decides what's worth keeping long-term."""
        try:
            self._client.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                user_id=user_id,
            )
            logger.info(f"Stored turn to long-term memory for user_id={user_id}")
        except Exception as e:
            raise AgentException(e, sys) from e

    def recall(self, user_id: str, query: str) -> list[str]:
        """Retrieval: top-k relevant memories for this query, as plain strings."""
        try:
            response = self._client.search(
                query=query,
                filters={"user_id": user_id},
                limit=self.top_k,
            )
            memories = self._extract_memories(response)
            logger.info(f"Retrieved {len(memories)} long-term memories for user_id={user_id}")
            return memories
        except Exception as e:
            raise AgentException(e, sys) from e

    def _extract_memories(self, response) -> list[str]:
        """
        Mem0's response shape has changed across SDK versions, so this
        normalizes whatever comes back into a flat list[str]. Keeping this
        logic isolated here means future SDK changes only touch this method.
        """
        # Some versions wrap results under a "results" key
        items = response.get("results", response) if isinstance(response, dict) else response

        memories = []
        for item in items:
            if isinstance(item, str):
                memories.append(item)
            elif isinstance(item, dict):
                memories.append(item.get("memory", str(item)))
        return memories