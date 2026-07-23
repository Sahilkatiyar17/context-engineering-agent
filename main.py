# main.py

import sys
import logging

from app.agents.graph import ResearchAgentGraph
from app.utils.exception import AgentException
from app.utils.tracing import TracingConfigurator
from app.evaluation.experiment_logger import ExperimentLogger

from app.agents.graph import ResearchAgentGraph
from app.utils.tracing import TracingConfigurator
from app.evaluation.experiment_logger import ExperimentLogger
import app.utils.logger  # noqa: F401

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

    USER_ID = "sahil"                                  # long-term memory scope -- persists across ALL threads
    EXPERIMENT_ID = "long_term_memory"
    STRATEGY = "mem0_top5_plus_summary"
    THREAD_ID = "long_term_memory-session-2"            # conversation scope -- change manually for a fresh thread

    graph = ResearchAgentGraph(user_id=USER_ID, summarize_after_n_messages=6)
    exp_logger = ExperimentLogger()

    conversation = [
    "Can you remember that i like python codes.",]

    for question in conversation:
        result_state = graph.run(question, thread_id=THREAD_ID, experiment_id=EXPERIMENT_ID, strategy=STRATEGY)
        exp_logger.log(experiment_id=EXPERIMENT_ID, strategy=STRATEGY, state=result_state)
        print(f"\nQ: {question}\nA: {result_state.answer}")
        print(f"[long_term_memories_used={len(result_state.long_term_memories)}]")