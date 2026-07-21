# main.py

import sys
import logging

from app.agents.graph import ResearchAgentGraph
from app.utils.exception import AgentException
from app.utils.tracing import TracingConfigurator
from app.evaluation.experiment_logger import ExperimentLogger

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
    TracingConfigurator.configure()

    app = ResearchAgentApp()
    exp_logger = ExperimentLogger()

    question = "What is PPO in reinforcement learning?"
    result_state = app.graph.run(question)  # use graph.run directly to get full state, not just .answer

    exp_logger.log(experiment_id="phase1_baseline", strategy="raw_search_no_memory", state=result_state)

    print("\n--- ANSWER ---")
    print(result_state.answer)