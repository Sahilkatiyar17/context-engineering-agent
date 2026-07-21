# main.py

import sys
import logging

from app.agents.graph import ResearchAgentGraph
from app.utils.exception import AgentException

# Importing this configures logging as a side effect (see app/utils/logger.py)
import app.utils.logger  # noqa: F401

logger = logging.getLogger(__name__)


class ResearchAgentApp:
    """
    Top-level application wrapper. Keeps main.py itself as a thin
    entrypoint -- the actual orchestration logic lives here as a class,
    so it's importable/testable without running __main__.
    """

    def __init__(self):
        self.graph = ResearchAgentGraph()

    def ask(self, question: str) -> str:
        try:
            logger.info(f"Received question: {question!r}")
            result_state = self.graph.run(question)
            logger.info("Question answered successfully")
            return result_state.answer
        except AgentException as e:
            logger.error(str(e))
            raise


if __name__ == "__main__":
    app = ResearchAgentApp()
    question = "What is PPO in reinforcement learning?"
    answer = app.ask(question)
    print("\n--- ANSWER ---")
    print(answer)