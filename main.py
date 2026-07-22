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

    EXPERIMENT_ID = "short_term_memory"
    STRATEGY = "summary_after_6"

    graph = ResearchAgentGraph(summarize_after_n_messages=6)
    exp_logger = ExperimentLogger()
    thread_id = f"{EXPERIMENT_ID}-session-2"

    conversation = [
        # "What is PPO in reinforcement learning?",
        # "How does it compare to TRPO?",
        # "Can you explain the clipping objective in more detail?",
        # "Can summarize everything we learned so far?",
        # "can you explain me how DQN is different from above topics?",
        # "what is a policy gradient in reinforcement learning?",
        "what we discussed today? give me the topics name only",
        
    ]

    for question in conversation:
        print(f"[DEBUG] thread_id={thread_id}")
        result_state = graph.run(question, thread_id=thread_id, experiment_id=EXPERIMENT_ID, strategy=STRATEGY)
        print(f"[DEBUG] summary_present={bool(result_state.summary)} messages_count={len(result_state.messages)}")
        exp_logger.log(experiment_id=EXPERIMENT_ID, strategy=STRATEGY, state=result_state)
        print(f"\nQ: {question}\nA: {result_state.answer}")