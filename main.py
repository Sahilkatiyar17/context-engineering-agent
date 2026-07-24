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
from app.context.filters import NoOpFilter, RelevanceScoreFilter
from app.context.ranking import NoOpRanker, CohereRerankRanker
from app.context.deduplication import SimilarityDeduplicator
from app.context.compression import NoOpCompressor, LLMCompressor

from experiments.dataset import EXPERIMENT_DATASET



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

    USER_ID = "sahil"
    EXPERIMENT_ID = "context_engineering"
    STRATEGY = "tight_filter_wide_memory"
    THREAD_ID = "tight_filter_wide_memory-1"

    graph = ResearchAgentGraph(
        user_id=USER_ID,
        context_filter=RelevanceScoreFilter(min_score=0.5),
        context_deduplicator=SimilarityDeduplicator(similarity_threshold=0.75),
        context_ranker=CohereRerankRanker(),
        context_compressor=LLMCompressor(),
    )
    exp_logger = ExperimentLogger()

    print(f"Running dataset -- experiment_id={EXPERIMENT_ID} strategy={STRATEGY} thread_id={THREAD_ID}\n")

    for item in EXPERIMENT_DATASET:
        result_state = graph.run(
            item["question"],
            thread_id=THREAD_ID,
            experiment_id=EXPERIMENT_ID,
            strategy=STRATEGY,
        )
        exp_logger.log(
            experiment_id=EXPERIMENT_ID,
            strategy=STRATEGY,
            state=result_state,
            notes=item["id"],
        )
        print(f"[{item['id']}] Q: {item['question']}")
        print(f"[{item['id']}] A: {result_state.answer}")
        print(f"[{item['id']}] tokens={result_state.total_tokens} latency={result_state.latency_seconds}s\n")

    print("Dataset run complete.")