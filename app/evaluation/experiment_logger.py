# app/evaluation/experiment_logger.py

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from pydantic import BaseModel,Field

from app.agents.state import AgentState
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"


import uuid
# ... existing imports stay

class ExperimentRecord(BaseModel):
    timestamp: str
    experiment_id: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str
    question: str
    answer: str
    total_tokens: int | None
    latency_seconds: float | None
    notes: str = ""


class ExperimentLogger:
    """
    Appends structured experiment results to experiments/results.jsonl.

    This is deliberately separate from app/utils/logger.py:
    that one is for debugging (rotating .log files), this one is
    permanent research data -- one JSON line per run, easy to load
    into pandas later for the blog post / comparison tables.
    """

    def __init__(self, results_file: Path = RESULTS_FILE):
        self.results_file = results_file
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, experiment_id: str, strategy: str, state: AgentState, notes: str = "") -> None:
        try:
            record = ExperimentRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                experiment_id=experiment_id,
                strategy=strategy,
                question=state.question,
                answer=state.answer or "",
                total_tokens=state.total_tokens,
                latency_seconds=state.latency_seconds,
                notes=notes,
            )
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            logger.info(f"Logged experiment: {experiment_id} / {strategy}")
        except Exception as e:
            raise AgentException(e, sys) from e